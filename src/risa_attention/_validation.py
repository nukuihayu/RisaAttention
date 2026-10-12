# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Input contracts, masks, and kernel tile selection."""

from __future__ import annotations

import torch

from . import _backend

DTYPE_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
KEY_BLOCK_SIZE = 64
LARGE_KEY_BLOCK_SIZE = 128
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MIN_PATTERN_SPARSITY = 0.05
OUTPUT_LAYOUTS = ("hnd", "nhd")
VALUE_QUANTIZATION_MODES = ("midpoint_affine", "absmax")


def uses_centered_values(v_quantization: str) -> bool:
    if v_quantization not in VALUE_QUANTIZATION_MODES:
        raise ValueError(
            f"v_quantization must be one of {VALUE_QUANTIZATION_MODES}, got {v_quantization!r}"
        )
    return v_quantization == "midpoint_affine"


def mask_signature(mask: torch.Tensor | None) -> tuple | None:
    if mask is None:
        return None
    # Inference tensors lack a version counter: their contents must remain
    # immutable for the lifetime of a retained pattern.
    try:
        version = mask._version
    except RuntimeError:
        version = None
    return (
        mask.data_ptr(),
        version,
        tuple(mask.shape),
        tuple(mask.stride()),
        mask.dtype,
        mask.device,
    )


def apply_score_mask(scores, attn_mask, is_causal, start):
    if attn_mask is not None:
        bias = attn_mask[..., start : start + scores.shape[-2], :]
        scores = (
            scores.masked_fill(~bias, -torch.inf) if bias.dtype == torch.bool else scores + bias
        )
    if is_causal:
        queries = torch.arange(start, start + scores.shape[-2], device=scores.device)
        keys = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores.masked_fill(keys[None, :] > queries[:, None], -torch.inf)
    return scores


def pad_key_length(length: int, key_block_size: int = KEY_BLOCK_SIZE) -> int:
    return ((length + key_block_size - 1) // key_block_size) * key_block_size


def select_key_block_size(
    kernel_head_dim: int,
    key_length: int,
    *,
    has_mask: bool,
) -> int:
    if not has_mask and kernel_head_dim >= 128 and key_length > 1024:
        return LARGE_KEY_BLOCK_SIZE
    return KEY_BLOCK_SIZE


def wrap_optional_tensor(tensor):
    return None if tensor is None else _backend.wrap_for_dlpack(tensor)


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether the compiled INT8 attention kernel supports this GPU."""
    return _backend.is_available(device)


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    if q.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"q, k, and v must be float32, float16, or bfloat16, got {q.dtype}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, and {v.dtype}"
        )
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "INT8 attention requires the risa_attention CUDA extension and an NVIDIA SM80+ GPU"
        )

    batch_size, query_heads, query_length, head_dim = q.shape
    k_batch, key_value_heads, key_length, k_head_dim = k.shape
    if v.shape != (batch_size, key_value_heads, key_length, head_dim):
        raise ValueError(
            f"v must have shape [q.batch, k.heads, k.sequence, q.head_dim], got {tuple(v.shape)}"
        )
    if k_batch != batch_size or k_head_dim != head_dim:
        raise ValueError(
            f"q and k batch/head dimensions must match, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if (
        batch_size == 0
        or query_heads == 0
        or key_value_heads == 0
        or query_length == 0
        or key_length == 0
    ):
        raise ValueError("batch, head counts, and sequence lengths must be positive")
    if query_heads % key_value_heads != 0:
        raise ValueError(
            f"q head count ({query_heads}) must be divisible by k/v head count ({key_value_heads})"
        )
    if head_dim <= 0 or head_dim > 256:
        raise ValueError(f"head_dim must be in [1, 256], got {head_dim}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last dimension of q, k, and v must be contiguous")
    if attn_mask is None:
        return None
    if attn_mask.device != q.device:
        raise ValueError("attn_mask must be on the same CUDA device as q, k, and v")
    if attn_mask.dtype not in (
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("attn_mask must be bool, float16, bfloat16, or float32")
    try:
        return torch.broadcast_to(attn_mask, (batch_size, query_heads, query_length, key_length))
    except RuntimeError as error:
        raise ValueError(
            "attn_mask must be broadcastable to "
            f"[{batch_size}, {query_heads}, {query_length}, {key_length}], got {tuple(attn_mask.shape)}"
        ) from error
