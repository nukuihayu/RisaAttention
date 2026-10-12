# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Q/K/V preparation and optional fused permutation."""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from . import _backend
from ._types import PrequantizedInt8Attention
from ._validation import (
    DTYPE_CODES,
    pad_key_length,
    select_key_block_size,
    uses_centered_values,
    validate_inputs,
)


def prequantize_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    v_quantization: str = "absmax",
    is_causal: bool = False,
    return_lse: bool = False,
    kv_permutation=None,
    fuse_gather: bool = True,
) -> PrequantizedInt8Attention:
    """Quantize Q, K, and V without allocating the attention output.

    The returned object does not retain the floating-point inputs, so model
    code can delete those tensors before calling
    :func:`int8_attention_from_prequantized`. Quantization and consumption use
    the current CUDA stream and preserve normal PyTorch stream ordering; no
    host synchronization is introduced.

    This is an inference and peak-memory API. The regular :func:`int8_attention`
    remains the lower-overhead single-call path when early Q/K/V release is not
    needed.
    """
    attn_mask = validate_inputs(q, k, v, attn_mask)
    transform_pointers = ()
    if kv_permutation is not None:
        from .permutation import prepare_inputs

        q, k, v, pointers = prepare_inputs(
            q,
            k,
            v,
            permutation=kv_permutation,
            mask=attn_mask,
            causal=is_causal,
            return_lse=return_lse,
            fuse=fuse_gather,
        )
        if any(pointers):
            transform_pointers = pointers
    use_midpoint_affine = uses_centered_values(v_quantization)

    original_head_dim = q.shape[-1]
    input_dtype = q.dtype
    if original_head_dim <= 64:
        kernel_head_dim = 64
    elif original_head_dim <= 128:
        kernel_head_dim = 128
    else:
        kernel_head_dim = 256
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = functional.pad(q, padding)
        k = functional.pad(k, padding)
        v = functional.pad(v, padding)

    attention_scale = original_head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError(f"scale must be finite, got {attention_scale}")

    batch_size, query_heads, query_length, _ = q.shape
    _, key_value_heads, key_length, _ = k.shape
    key_block_size = select_key_block_size(
        kernel_head_dim,
        key_length,
        has_mask=attn_mask is not None and attn_mask.stride(-2) != 0,
    )
    padded_key_length = pad_key_length(key_length, key_block_size)
    query_codes = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    key_codes = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    query_scales_per_block = 64 if kernel_head_dim == 256 else 32
    query_scales = torch.empty(
        batch_size,
        query_heads,
        ((query_length + 127) // 128) * query_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    key_scales = torch.empty(
        batch_size,
        key_value_heads,
        ((key_length + key_block_size - 1) // key_block_size) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    value_codes = torch.empty(
        batch_size * key_value_heads * kernel_head_dim,
        padded_key_length,
        dtype=torch.int8,
        device=q.device,
    )
    value_scales = torch.empty(
        batch_size * key_value_heads * kernel_head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    value_centers = torch.empty_like(value_scales)
    anchor_indices = torch.empty(batch_size, key_value_heads, dtype=torch.int32, device=q.device)
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _backend._C.risa_sdpa_quantize(
        _backend.wrap_for_dlpack(q),
        _backend.wrap_for_dlpack(k),
        _backend.wrap_for_dlpack(v),
        _backend.wrap_for_dlpack(query_codes),
        _backend.wrap_for_dlpack(query_scales),
        _backend.wrap_for_dlpack(key_codes),
        _backend.wrap_for_dlpack(key_scales),
        _backend.wrap_for_dlpack(value_codes),
        _backend.wrap_for_dlpack(value_scales),
        _backend.wrap_for_dlpack(value_centers),
        key_block_size,
        DTYPE_CODES[input_dtype],
        use_midpoint_affine,
        stream_ptr,
        anchor_indices_ptr,
        *transform_pointers,
    )

    lse_shift = None
    if return_lse and hasattr(_backend._C, "risa_lse_shift"):
        lse_shift = torch.empty(
            (batch_size, query_heads, query_length), device=q.device, dtype=torch.float32
        )
        _backend._C.risa_lse_shift(
            _backend.wrap_for_dlpack(q),
            _backend.wrap_for_dlpack(k),
            _backend.wrap_for_dlpack(anchor_indices),
            _backend.wrap_for_dlpack(lse_shift),
            attention_scale,
            stream_ptr,
        )
    elif return_lse:
        # K is shifted by a sampled anchor before quantization. Restore the
        # row-wise logit offset so LSE from different windows can be combined.
        indices = anchor_indices.clamp_min(0).long()
        anchors = (
            k.detach()
            .gather(
                2, indices[..., None, None].expand(batch_size, key_value_heads, 1, kernel_head_dim)
            )
            .squeeze(2)
            .float()
        )
        anchors = anchors * (anchor_indices >= 0)[..., None]
        grouped_q = q.detach().reshape(
            batch_size,
            key_value_heads,
            query_heads // key_value_heads,
            query_length,
            kernel_head_dim,
        )
        lse_shift = (grouped_q.float() * anchors[:, :, None, None, :]).sum(-1)
        lse_shift = lse_shift.reshape(batch_size, query_heads, query_length) * attention_scale

    return PrequantizedInt8Attention(
        q=query_codes,
        k=key_codes,
        v=value_codes,
        q_scale=query_scales,
        k_scale=key_scales,
        v_scale=value_scales,
        v_center=value_centers,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        cta_k=key_block_size,
        attn_mask=attn_mask,
        v_quantization=v_quantization,
        is_causal=is_causal,
        lse_shift=lse_shift,
        kv_permuted=kv_permutation is not None,
    )
