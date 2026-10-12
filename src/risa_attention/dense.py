# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Public dense attention API and torch.compile operator registration."""

from __future__ import annotations

import torch

from ._execution import execute_dense_attention, int8_attention_from_prequantized
from ._quantization import prequantize_int8_attention
from ._validation import uses_centered_values


@torch.library.custom_op("risa_attention::int8_attention", mutates_args=())
def _op_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    use_midpoint_affine: bool,
    is_causal: bool,
) -> torch.Tensor:
    return execute_dense_attention(
        q,
        k,
        v,
        scale=scale,
        attn_mask=None,
        use_midpoint_affine=use_midpoint_affine,
        is_causal=is_causal,
    )


@_op_int8_attention.register_fake
def _op_int8_attention_fake(
    q,
    k,
    v,
    scale,
    use_midpoint_affine,
    is_causal,
):
    return q.new_empty(q.shape)


@torch.library.custom_op("risa_attention::int8_attention_masked", mutates_args=())
def _op_int8_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None,
    use_midpoint_affine: bool,
    is_causal: bool,
) -> torch.Tensor:
    return execute_dense_attention(
        q,
        k,
        v,
        scale=scale,
        attn_mask=attn_mask,
        use_midpoint_affine=use_midpoint_affine,
        is_causal=is_causal,
    )


@_op_int8_attention_masked.register_fake
def _op_int8_attention_masked_fake(
    q,
    k,
    v,
    attn_mask,
    scale,
    use_midpoint_affine,
    is_causal,
):
    return q.new_empty(q.shape)


def int8_attention(
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute inference SDPA with signed INT8 Q/K/V and unsigned INT8 P.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout. Grouped-query
    attention and unequal non-causal Q/K sequence lengths are supported. Head
    dimensions are padded to the kernel's 64-, 128-, or 256-wide tile and
    sliced back on return; 64, 128, and 256 take the zero-copy dimension path.
    Q and K receive the same fused block-Hadamard rotation before INT8
    quantization, preserving their exact dot products while reducing
    quantization outliers. K lengths up to 256 use low-overhead H4 blocks and
    longer D64 attention uses H64. The common D128 path uses a fixed signed H128
    transform, while padded D256 uses plain H128. The kernel samples K on the
    GPU and subtracts a representative key only when doing so improves the
    quantization range. This model-independent shift is exactly
    softmax-invariant and uses one int32 of temporary storage per batch/KV-head.
    V defaults to per-channel symmetric absmax INT8 quantization. Explicit
    midpoint_affine adds a residual-corrected center after normalization.
    The denominator uses the same rounded U8 probabilities as P*V. Softmax
    score, maximum, exponential, denominator, reciprocal, and V dequantization
    arithmetic is FP32. This path does not allocate FP8 tensors or execute FP8
    MMA instructions.
    """
    if return_lse or kv_permutation is not None:
        packed = prequantize_int8_attention(
            q,
            k,
            v,
            scale=scale,
            attn_mask=attn_mask,
            v_quantization=v_quantization,
            is_causal=is_causal,
            return_lse=return_lse,
            kv_permutation=kv_permutation,
            fuse_gather=fuse_gather,
        )
        return int8_attention_from_prequantized(packed, return_lse=return_lse)
    use_midpoint_affine = uses_centered_values(v_quantization)
    if attn_mask is None:
        return torch.ops.risa_attention.int8_attention(
            q,
            k,
            v,
            scale,
            use_midpoint_affine,
            is_causal,
        )
    return torch.ops.risa_attention.int8_attention_masked(
        q,
        k,
        v,
        attn_mask,
        scale,
        use_midpoint_affine,
        is_causal,
    )
