# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Native dense and prequantized attention execution."""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from . import _backend
from ._key_mask import prepare_key_mask_tiles
from ._types import PrequantizedInt8Attention, RetainedMassPattern
from ._validation import (
    DTYPE_CODES,
    MIN_PATTERN_SPARSITY,
    OUTPUT_LAYOUTS,
    is_available,
    mask_signature,
    pad_key_length,
    select_key_block_size,
    validate_inputs,
    wrap_optional_tensor,
)


def execute_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    use_midpoint_affine: bool = False,
    is_causal: bool = False,
) -> torch.Tensor:
    attn_mask = validate_inputs(q, k, v, attn_mask)

    original_head_dim = q.shape[-1]
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
        batch_size * key_value_heads * kernel_head_dim, dtype=torch.float32, device=q.device
    )
    value_centers = torch.empty_like(value_scales)

    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    output = torch.empty(
        batch_size, query_heads, query_length, kernel_head_dim, dtype=output_dtype, device=q.device
    )

    anchor_indices = torch.empty(batch_size, key_value_heads, dtype=torch.int32, device=q.device)
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    mask_tiles = prepare_key_mask_tiles(
        attn_mask, key_block_size, query_length, attention_scale, is_causal
    )
    mask_arguments = (
        {"key_mask_tiles": _backend.wrap_for_dlpack(mask_tiles)} if mask_tiles is not None else {}
    )
    _backend._C.risa_sdpa(
        _backend.wrap_for_dlpack(q),
        _backend.wrap_for_dlpack(k),
        _backend.wrap_for_dlpack(v),
        _backend.wrap_for_dlpack(output),
        _backend.wrap_for_dlpack(query_codes),
        _backend.wrap_for_dlpack(query_scales),
        _backend.wrap_for_dlpack(key_codes),
        _backend.wrap_for_dlpack(key_scales),
        _backend.wrap_for_dlpack(value_codes),
        _backend.wrap_for_dlpack(value_scales),
        _backend.wrap_for_dlpack(value_centers),
        attention_scale,
        DTYPE_CODES[q.dtype],
        DTYPE_CODES[output_dtype],
        use_midpoint_affine,
        stream_ptr,
        anchor_indices_ptr,
        None if attn_mask is None else _backend.wrap_for_dlpack(attn_mask),
        cta_k=key_block_size,
        is_causal=is_causal,
        **mask_arguments,
    )

    if original_head_dim != kernel_head_dim:
        output = output[..., :original_head_dim]
    return output.float() if q.dtype == torch.float32 else output


def int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
    *,
    sparse_pattern: RetainedMassPattern | None = None,
    output_layout: str = "hnd",
    _block_mass: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run INT8 attention after the floating-point Q/K/V inputs are released.

    Both layouts return a logical ``[batch, heads, sequence, head_dim]``
    tensor. ``"nhd"`` stores sequence before heads, so the common
    ``output.transpose(1, 2).reshape(batch, sequence, -1)`` is a view rather
    than a full-tensor copy when the head dimension needs no padding. The
    default ``"hnd"`` remains contiguous.
    """
    if not isinstance(quantized, PrequantizedInt8Attention):
        raise TypeError(
            "quantized must be returned by prequantize_int8_attention, got "
            f"{type(quantized).__name__}"
        )
    if output_layout not in OUTPUT_LAYOUTS:
        raise ValueError(f"output_layout must be one of {OUTPUT_LAYOUTS}, got {output_layout!r}")
    if quantized.kv_permuted and (sparse_pattern is not None or _block_mass is not None):
        raise ValueError("permuted K/V is supported only by dense attention")
    if return_lse and quantized.lse_shift is None:
        raise ValueError("prequantize with return_lse=True to retain the K-anchor correction")

    packed_tensors = (
        quantized.q,
        quantized.k,
        quantized.v,
        quantized.q_scale,
        quantized.k_scale,
        quantized.v_scale,
        quantized.v_center,
    )
    if not quantized.q.is_cuda:
        raise ValueError("prequantized INT8 attention tensors must be on a CUDA device")
    if any(tensor.device != quantized.q.device for tensor in packed_tensors[1:]):
        raise ValueError("prequantized INT8 attention tensors must be on the same CUDA device")
    if quantized.attn_mask is not None and quantized.attn_mask.device != quantized.q.device:
        raise ValueError("attn_mask must be on the same CUDA device as the packed tensors")
    if sparse_pattern is not None:
        if not isinstance(sparse_pattern, RetainedMassPattern):
            raise TypeError("sparse_pattern must be returned by build_retained_mass_pattern")
        if sparse_pattern.is_causal != quantized.is_causal:
            raise ValueError("sparse pattern and attention call must use the same is_causal")
        if sparse_pattern.mask_signature != mask_signature(quantized.attn_mask):
            raise ValueError("sparse pattern mask has changed; rebuild the pattern")
        expected = (
            quantized.q.shape[0],
            quantized.q.shape[1],
            quantized.q.shape[2],
            quantized.k.shape[2],
            128,
            quantized.cta_k,
        )
        actual = (
            sparse_pattern.batch_size,
            sparse_pattern.q_heads,
            sparse_pattern.q_length,
            sparse_pattern.kv_length,
            sparse_pattern.query_block_size,
            sparse_pattern.key_block_size,
        )
        if actual != expected:
            raise ValueError(
                "sparse pattern metadata does not match packed attention: "
                f"expected {expected}, got {actual}"
            )
        if not math.isclose(
            sparse_pattern.attention_scale,
            quantized.attention_scale,
            rel_tol=1e-7,
            abs_tol=0.0,
        ):
            raise ValueError("sparse pattern and attention call must use the same scale")
        if (
            sparse_pattern.row_offsets.device != quantized.q.device
            or sparse_pattern.block_indices.device != quantized.q.device
        ):
            raise ValueError("sparse pattern and packed tensors must be on the same CUDA device")
        if (
            sparse_pattern.row_offsets.dtype != torch.int32
            or sparse_pattern.block_indices.dtype != torch.int32
            or not sparse_pattern.row_offsets.is_contiguous()
            or not sparse_pattern.block_indices.is_contiguous()
        ):
            raise TypeError("sparse CSR row offsets and block indices must be contiguous int32")
        if sparse_pattern.sparsity < MIN_PATTERN_SPARSITY:
            sparse_pattern = None
    if _block_mass is not None:
        if sparse_pattern is not None:
            raise ValueError("block mass requires dense attention")
        mass_rows = (
            quantized.q.shape[0] * quantized.q.shape[1] * math.ceil(quantized.q.shape[2] / 128)
        )
        mass_columns = math.ceil(quantized.k.shape[2] / quantized.cta_k)
        query_rows = quantized.q.shape[0] * quantized.q.shape[1] * quantized.q.shape[2]
        stats_elements = query_rows * (mass_columns * 2 + 2)
        expected_workspace_elements = mass_rows * mass_columns + stats_elements
        if (
            _block_mass.ndim != 1
            or _block_mass.numel() != expected_workspace_elements
            or _block_mass.dtype != torch.float32
            or _block_mass.device != quantized.q.device
            or not _block_mass.is_contiguous()
        ):
            raise ValueError(
                "block mass workspace must be contiguous CUDA float32 with "
                f"{expected_workspace_elements} elements"
            )
    if not is_available(quantized.q.device):
        raise RuntimeError(
            "INT8 attention requires the risa_attention CUDA extension and an NVIDIA SM80+ GPU"
        )

    batch_size, query_heads, query_length, kernel_head_dim = quantized.q.shape
    output_dtype = (
        torch.bfloat16 if quantized.input_dtype == torch.float32 else quantized.input_dtype
    )

    if output_layout == "nhd":
        output = torch.empty(
            batch_size,
            query_length,
            query_heads,
            kernel_head_dim,
            dtype=output_dtype,
            device=quantized.q.device,
        ).permute(0, 2, 1, 3)
    else:
        output = torch.empty(
            batch_size,
            query_heads,
            query_length,
            kernel_head_dim,
            dtype=output_dtype,
            device=quantized.q.device,
        )

    stream_ptr = torch.cuda.current_stream(quantized.q.device).cuda_stream
    arguments = (
        _backend.wrap_for_dlpack(quantized.q),
        _backend.wrap_for_dlpack(quantized.k),
        _backend.wrap_for_dlpack(quantized.v),
        _backend.wrap_for_dlpack(output),
        _backend.wrap_for_dlpack(quantized.q_scale),
        _backend.wrap_for_dlpack(quantized.k_scale),
        _backend.wrap_for_dlpack(quantized.v_scale),
        _backend.wrap_for_dlpack(quantized.v_center),
        quantized.cta_k,
        quantized.attention_scale,
        DTYPE_CODES[output_dtype],
        stream_ptr,
    )
    lse = (
        torch.empty(
            (batch_size, query_heads, query_length), dtype=torch.float32, device=quantized.q.device
        )
        if return_lse
        else None
    )
    wrap = wrap_optional_tensor
    mask_tiles = None
    if sparse_pattern is None and _block_mass is None:
        mask_tiles = prepare_key_mask_tiles(
            quantized.attn_mask,
            quantized.cta_k,
            query_length,
            quantized.attention_scale,
            quantized.is_causal,
        )
    mask_arguments = {"key_mask_tiles": wrap(mask_tiles)} if mask_tiles is not None else {}
    _backend._C.risa_sdpa_prequantized(
        *arguments,
        attn_mask=wrap(quantized.attn_mask),
        sparse_row_offsets=wrap(None if sparse_pattern is None else sparse_pattern.row_offsets),
        sparse_block_indices=wrap(None if sparse_pattern is None else sparse_pattern.block_indices),
        block_mass=wrap(_block_mass),
        is_causal=quantized.is_causal,
        lse=wrap(lse),
        **mask_arguments,
    )

    if quantized.original_head_dim != kernel_head_dim:
        output = output[..., : quantized.original_head_dim]
    output = output.float() if quantized.input_dtype == torch.float32 else output
    if return_lse:
        lse.add_(quantized.lse_shift)
    return (output, lse) if return_lse else output
