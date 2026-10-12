# SPDX-License-Identifier: Apache-2.0
"""Metadata for exact skipping of fully excluded broadcast-mask tiles."""

import torch

from . import _backend


def prepare_key_mask_tiles(mask, key_block_size, query_length, scale, is_causal):
    if (
        mask is None
        or mask.stride(-2) != 0
        or mask.shape[-1] <= 1024
        or query_length < 256
        or scale <= 0
        or is_causal
    ):
        return None
    batch, heads, _, length = mask.shape
    width = (length + key_block_size - 1) // key_block_size + 1
    tiles = torch.empty((batch, heads, width), device=mask.device, dtype=torch.int32)
    _backend._C.prepare_key_mask_tiles(
        _backend.wrap_for_dlpack(mask),
        _backend.wrap_for_dlpack(tiles),
        key_block_size,
        torch.cuda.current_stream(mask.device).cuda_stream,
    )
    return tiles
