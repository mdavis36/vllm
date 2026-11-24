# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register device KV cache memory with the Mooncake store in pieces.

By default each KV cache storage is one `register_buffer` call, which can be
100+ GiB. Two things go wrong with a device region that large:

- Mooncake's EFA transport budgets every NIC at 22 Mi PTEs and counts a CUDA
  buffer in 4 KiB pages, so it splits any device region over 88 GiB into
  chunks. Up to Mooncake 0.3.14 the topology-local NIC selection
  (`MC_EFA_NIC_SELECTION=local`) is applied only to a single-chunk device
  buffer; the chunks of a split one are partitioned across all NICs by index,
  so part of the GPU's KV is served by NICs on the other CPU socket
  (kvcache-ai/Mooncake#5684 fixes this in the transfer engine).
- RDMA stacks without linux 15fe76e23615 ("RDMA/umem: Fix truncation for block
  sizes >= 4G") can wrap a dma-buf MR of 4 GiB or more (vllm-project/vllm#64048,
  ai-dynamo/nixl#2512; the NIXL connector registers in pieces for the same
  reason, see nixl/registration.py).

With `VLLM_MOONCAKE_STORE_CHUNKED_KV_REGISTRATION=1`, each device KV storage is
registered as consecutive pieces of at most
`VLLM_MOONCAKE_STORE_KV_REGISTRATION_PIECE_BYTES` (default 2 GiB). Every piece
is then a single chunk on the GPU's own NICs and a short MR. Transfers keep
their addresses: Mooncake's EFA and RDMA transports cap each slice at the end
of the registered buffer that holds it, so a transfer that spans two pieces is
posted as two slices (`efaBytesUntilChunkEnd` and `bytesUntilBufferEnd` in the
transfer engine).
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

CHUNKED_KV_REGISTRATION_ENV = "VLLM_MOONCAKE_STORE_CHUNKED_KV_REGISTRATION"
KV_REGISTRATION_PIECE_BYTES_ENV = "VLLM_MOONCAKE_STORE_KV_REGISTRATION_PIECE_BYTES"
DEFAULT_KV_REGISTRATION_PIECE_BYTES = 1 << 31
# Pieces stay page-aligned relative to the storage base and under 4 GiB, with
# room for the dma-buf range to be rounded to pages at both ends.
_PIECE_ALIGN_BYTES = 1 << 12
MAX_KV_REGISTRATION_PIECE_BYTES = (1 << 32) - (1 << 22)

_ON = ("1", "y", "yes", "on", "true")


def chunked_kv_registration_piece_bytes() -> int | None:
    """Piece size for device KV registrations, or None when registering each
    storage whole (the default)."""
    if os.environ.get(CHUNKED_KV_REGISTRATION_ENV, "0").strip().lower() not in _ON:
        return None
    raw = os.environ.get(KV_REGISTRATION_PIECE_BYTES_ENV, "").strip()
    piece_bytes = int(raw) if raw else DEFAULT_KV_REGISTRATION_PIECE_BYTES
    if (
        not _PIECE_ALIGN_BYTES <= piece_bytes <= MAX_KV_REGISTRATION_PIECE_BYTES
        or piece_bytes % _PIECE_ALIGN_BYTES
    ):
        raise ValueError(
            f"{KV_REGISTRATION_PIECE_BYTES_ENV}={piece_bytes} must be a multiple "
            f"of {_PIECE_ALIGN_BYTES} between {_PIECE_ALIGN_BYTES} and "
            f"{MAX_KV_REGISTRATION_PIECE_BYTES} bytes"
        )
    return piece_bytes


def plan_registration_pieces(
    base: int, length: int, piece_bytes: int | None
) -> list[tuple[int, int]]:
    """(addr, length) pieces tiling [base, base + length), each at most
    `piece_bytes`; the whole region when `piece_bytes` is None."""
    if piece_bytes is None or length <= piece_bytes:
        return [(base, length)]
    return [
        (base + offset, min(piece_bytes, length - offset))
        for offset in range(0, length, piece_bytes)
    ]


def register_device_region(
    store: Any, base: int, length: int, piece_bytes: int | None
) -> tuple[int, list[tuple[int, int]]]:
    """Register [base, base + length) with `store` in pieces.

    Returns (0, pieces) on success, or (the failing call's return code, [])
    after unregistering the pieces registered before it."""
    pieces = plan_registration_pieces(base, length, piece_bytes)
    for i, (addr, size) in enumerate(pieces):
        ret = store.register_buffer(addr, size)
        if ret != 0:
            for done_addr, _ in pieces[:i]:
                store.unregister_buffer(done_addr)
            return ret, []
    if piece_bytes is not None:
        logger.info(
            "Mooncake store KV registration: device storage 0x%x (%d bytes) "
            "registered as %d pieces of at most %d bytes (%s)",
            base,
            length,
            len(pieces),
            max(size for _, size in pieces),
            CHUNKED_KV_REGISTRATION_ENV,
        )
    return 0, pieces
