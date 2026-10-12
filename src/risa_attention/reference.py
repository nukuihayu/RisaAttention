# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""FP32 reference sparse selection and retained-mass metrics."""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from ._types import RetainedMassPattern
from ._validation import (
    SUPPORTED_DTYPES,
    apply_score_mask,
    is_available,
    mask_signature,
    select_key_block_size,
    validate_inputs,
)


@torch.inference_mode()
def build_retained_mass_pattern(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    theta: float = 0.99,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> RetainedMassPattern:
    """Construct frozen retained-mass support from exact dense block masses.

    The construction is chunked by 128 query tokens, so it never materializes
    the full attention matrix. It is intended for one early dense diffusion
    step and is then amortized across later calls to :func:`sparse_int8_attention`.
    Key blocks follow the selected Sage kernel tile (64 or 128 tokens), rather
    than FlashInfer's paper configuration of 32, to keep sparse loads aligned
    with the INT8 MMA pipeline.

    ``theta`` is the retained-mass threshold described by LoSA.
    """
    if not 0.0 < theta <= 1.0 or not math.isfinite(theta):
        raise ValueError("theta must be finite and in (0, 1]")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, sequence, head_dim]")
    if q.dtype not in SUPPORTED_DTYPES or q.dtype != k.dtype:
        raise TypeError("q and k must have the same float32, float16, or bfloat16 dtype")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError("sparse INT8 attention requires the CUDA extension and an SM80+ GPU")
    batch_size, query_heads, query_length, head_dim = q.shape
    if k.shape[0] != batch_size or k.shape[3] != head_dim:
        raise ValueError("sparse attention requires self-attention with equal Q/K lengths")
    key_length = k.shape[2]
    attn_mask = validate_inputs(q, k, k, attn_mask)
    key_value_heads = k.shape[1]
    if batch_size == 0 or query_heads == 0 or key_value_heads == 0 or query_length == 0:
        raise ValueError("batch, head counts, and sequence length must be positive")
    if query_heads % key_value_heads:
        raise ValueError("q head count must be divisible by k head count")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1, 256]")
    attention_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError("scale must be finite")

    kernel_head_dim = 64 if head_dim <= 64 else 128 if head_dim <= 128 else 256
    key_block_size = select_key_block_size(
        kernel_head_dim,
        key_length,
        has_mask=attn_mask is not None and attn_mask.stride(-2) != 0,
    )
    query_block_size = 128
    num_q_blocks = math.ceil(query_length / query_block_size)
    num_k_blocks = math.ceil(key_length / key_block_size)
    group_size = query_heads // key_value_heads
    k_float_t = k.float().transpose(-1, -2).unsqueeze(2)
    rows: list[list[int] | None] = [None] * (batch_size * query_heads * num_q_blocks)
    retained_total = 0.0
    mass_total = 0.0

    for query_block in range(num_q_blocks):
        start = query_block * query_block_size
        end = min(start + query_block_size, query_length)
        q_chunk = q[:, :, start:end].reshape(
            batch_size, key_value_heads, group_size, end - start, head_dim
        )
        scores = torch.matmul(q_chunk.float(), k_float_t).mul_(attention_scale)
        scores = apply_score_mask(
            scores.reshape(batch_size, query_heads, end - start, key_length),
            attn_mask,
            is_causal,
            start,
        )
        probabilities = torch.softmax(scores, dim=-1).nan_to_num_(nan=0.0)
        if key_length % key_block_size:
            probabilities = functional.pad(
                probabilities, (0, key_block_size - key_length % key_block_size)
            )
        masses = (
            probabilities.reshape(
                batch_size,
                key_value_heads,
                group_size,
                end - start,
                num_k_blocks,
                key_block_size,
            )
            .sum(dim=(3, 5))
            .reshape(batch_size, query_heads, num_k_blocks)
        )
        sorted_mass, descending_indices = masses.sort(dim=-1, descending=True)
        target = theta * masses.sum(dim=-1, keepdim=True)
        counts = (sorted_mass.cumsum(dim=-1) < target).sum(dim=-1).add_(1)
        counts.clamp_(max=num_k_blocks)

        masses_cpu = masses.cpu()
        indices_cpu = descending_indices.cpu()
        counts_cpu = counts.cpu()
        for batch_index in range(batch_size):
            for head_index in range(query_heads):
                count = int(counts_cpu[batch_index, head_index])
                selected = indices_cpu[batch_index, head_index, :count].tolist()
                selected.sort()
                row = (batch_index * query_heads + head_index) * num_q_blocks + query_block
                rows[row] = selected
                retained_total += float(masses_cpu[batch_index, head_index, selected].sum())
                mass_total += float(masses_cpu[batch_index, head_index].sum())

    row_offsets = [0]
    flat_indices: list[int] = []
    for selected in rows:
        if not selected:
            raise RuntimeError("retained-mass construction produced an empty support row")
        flat_indices.extend(selected)
        row_offsets.append(len(flat_indices))

    return RetainedMassPattern(
        row_offsets=torch.tensor(row_offsets, dtype=torch.int32, device=q.device),
        block_indices=torch.tensor(flat_indices, dtype=torch.int32, device=q.device),
        batch_size=batch_size,
        q_heads=query_heads,
        q_length=query_length,
        kv_length=key_length,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
        attention_scale=attention_scale,
        retained_mass_target=float(theta),
        measured_retained_mass=retained_total / max(mass_total, 1e-20),
        is_causal=is_causal,
        mask_signature=mask_signature(attn_mask),
        mask_reference=attn_mask,
    )


@torch.inference_mode()
def measure_pattern_recall(
    q: torch.Tensor,
    k: torch.Tensor,
    sparse_pattern: RetainedMassPattern,
    *,
    attn_mask: torch.Tensor | None = None,
) -> float:
    """Measure exact dense attention mass retained by a frozen pattern."""
    if not isinstance(sparse_pattern, RetainedMassPattern):
        raise TypeError("sparse_pattern must be returned by build_retained_mass_pattern")
    if q.ndim != 4 or k.ndim != 4 or q.device != k.device or not q.is_cuda:
        raise ValueError("q and k must be 4D tensors on the same CUDA device")
    attn_mask = validate_inputs(q, k, k, attn_mask)
    if mask_signature(attn_mask) != sparse_pattern.mask_signature:
        raise ValueError("pattern mask has changed")
    key_length = k.shape[2]
    batch_size, query_heads, query_length, head_dim = q.shape
    key_value_heads = k.shape[1] if k.ndim == 4 else 0
    if (
        k.shape != (batch_size, key_value_heads, key_length, head_dim)
        or q.dtype != k.dtype
        or query_heads % key_value_heads
    ):
        raise ValueError("q and k shapes/dtypes are incompatible")
    expected = (
        sparse_pattern.batch_size,
        sparse_pattern.q_heads,
        sparse_pattern.q_length,
        sparse_pattern.kv_length,
    )
    if (batch_size, query_heads, query_length, key_length) != expected:
        raise ValueError("q and k do not match the frozen retained-mass pattern")
    if sparse_pattern.row_offsets.device != q.device:
        raise ValueError("pattern and q/k must be on the same CUDA device")

    query_block_size = sparse_pattern.query_block_size
    key_block_size = sparse_pattern.key_block_size
    num_q_blocks = math.ceil(query_length / query_block_size)
    num_k_blocks = math.ceil(key_length / key_block_size)
    group_size = query_heads // key_value_heads
    k_float_t = k.float().transpose(-1, -2).unsqueeze(2)
    offsets = sparse_pattern.row_offsets.cpu()
    indices = sparse_pattern.block_indices.cpu()
    retained_total = 0.0
    mass_total = 0.0

    for query_block in range(num_q_blocks):
        start = query_block * query_block_size
        end = min(start + query_block_size, query_length)
        q_chunk = q[:, :, start:end].reshape(
            batch_size, key_value_heads, group_size, end - start, head_dim
        )
        scores = torch.matmul(q_chunk.float(), k_float_t).mul_(sparse_pattern.attention_scale)
        scores = apply_score_mask(
            scores.reshape(batch_size, query_heads, end - start, key_length),
            attn_mask,
            sparse_pattern.is_causal,
            start,
        )
        probabilities = torch.softmax(scores, dim=-1).nan_to_num_(nan=0.0)
        mass_total += float(probabilities.sum())
        if key_length % key_block_size:
            probabilities = functional.pad(
                probabilities, (0, key_block_size - key_length % key_block_size)
            )
        masses = (
            probabilities.reshape(
                batch_size,
                key_value_heads,
                group_size,
                end - start,
                num_k_blocks,
                key_block_size,
            )
            .sum(dim=(3, 5))
            .reshape(batch_size, query_heads, num_k_blocks)
            .cpu()
        )
        for batch_index in range(batch_size):
            for head_index in range(query_heads):
                row = (batch_index * query_heads + head_index) * num_q_blocks + query_block
                row_start = int(offsets[row])
                row_end = int(offsets[row + 1])
                retained_total += float(
                    masses[batch_index, head_index, indices[row_start:row_end]].sum()
                )
    return retained_total / max(mass_total, 1e-20)
