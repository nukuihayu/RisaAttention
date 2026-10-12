# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""CUDA retained-mass support construction and sparse reuse."""

from __future__ import annotations

import math

import torch

from . import _backend
from ._execution import int8_attention_from_prequantized
from ._quantization import prequantize_int8_attention
from ._types import PrequantizedInt8Attention, RetainedMassPattern
from ._validation import mask_signature


def select_retained_mass_csr(
    masses: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Select the shortest retained-mass prefix and pack it as ordered CSR."""
    row_count, key_blocks = masses.shape
    sorted_mass, descending_indices = masses.sort(dim=-1, descending=True)
    counts = torch.empty(row_count, dtype=torch.int32, device=masses.device)
    support = torch.empty_like(masses, dtype=torch.uint8)
    row_totals = torch.empty(row_count, 2, dtype=torch.float64, device=masses.device)
    row_offsets = torch.empty(row_count + 1, dtype=torch.int32, device=masses.device)
    block_storage = torch.empty(row_count * key_blocks, dtype=torch.int32, device=masses.device)
    summary = torch.empty(3, dtype=torch.float64, device=masses.device)
    scan_workspace = torch.empty(
        _backend._C.retained_mass_selector_workspace_size(row_count),
        dtype=torch.uint8,
        device=masses.device,
    )
    _backend._C.select_retained_mass_csr(
        _backend.wrap_for_dlpack(sorted_mass),
        _backend.wrap_for_dlpack(descending_indices),
        _backend.wrap_for_dlpack(counts),
        _backend.wrap_for_dlpack(support),
        _backend.wrap_for_dlpack(row_totals),
        _backend.wrap_for_dlpack(row_offsets),
        _backend.wrap_for_dlpack(block_storage),
        _backend.wrap_for_dlpack(summary),
        _backend.wrap_for_dlpack(scan_workspace),
        theta,
        torch.cuda.current_stream(masses.device).cuda_stream,
    )
    selected_mass, total_mass, selected_blocks = summary.tolist()
    block_indices = block_storage[: int(selected_blocks)]
    measured_mass = selected_mass / max(total_mass, 1e-20)
    return row_offsets, block_indices, measured_mass


@torch.inference_mode()
def construct_sparse_int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
    *,
    theta: float = 0.99,
    output_layout: str = "hnd",
) -> tuple[torch.Tensor, RetainedMassPattern]:
    """Construct reusable retained-mass support while consuming packed Q/K/V.

    The CUDA attention kernel captures its online-softmax sufficient statistics
    and reconstructs block mass without replaying QK. Selection and CSR
    construction stay on the GPU. The masses follow Sage's quantized U8
    probability path; use :func:`measure_pattern_recall` when exact FP
    attention recall is required for evaluation.

    ``theta`` is the retained-mass threshold.
    """
    if not 0.0 < theta <= 1.0 or not math.isfinite(theta):
        raise ValueError("theta must be finite and in (0, 1]")
    if not isinstance(quantized, PrequantizedInt8Attention):
        raise TypeError(
            "quantized must be returned by prequantize_int8_attention, got "
            f"{type(quantized).__name__}"
        )

    batch_size, query_heads, query_length, _ = quantized.q.shape
    num_q_blocks = math.ceil(query_length / 128)
    num_k_blocks = math.ceil(quantized.k.shape[2] / quantized.cta_k)
    mass_elements = batch_size * query_heads * num_q_blocks * num_k_blocks
    stats_elements = batch_size * query_heads * query_length * (num_k_blocks * 2 + 2)
    workspace = torch.empty(
        mass_elements + stats_elements,
        dtype=torch.float32,
        device=quantized.q.device,
    )
    masses = workspace[:mass_elements].view(batch_size * query_heads * num_q_blocks, num_k_blocks)
    output = int8_attention_from_prequantized(
        quantized,
        output_layout=output_layout,
        _block_mass=workspace,
    )

    row_offsets, block_indices, measured_mass = select_retained_mass_csr(masses, theta)

    pattern = RetainedMassPattern(
        row_offsets=row_offsets,
        block_indices=block_indices,
        batch_size=batch_size,
        q_heads=query_heads,
        q_length=query_length,
        kv_length=quantized.k.shape[2],
        query_block_size=128,
        key_block_size=quantized.cta_k,
        attention_scale=quantized.attention_scale,
        retained_mass_target=float(theta),
        measured_retained_mass=measured_mass,
        is_causal=quantized.is_causal,
        mask_signature=mask_signature(quantized.attn_mask),
        mask_reference=quantized.attn_mask,
    )
    return output, pattern


@torch.inference_mode()
def construct_sparse_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    theta: float = 0.99,
    scale: float | None = None,
    output_layout: str = "hnd",
    v_quantization: str = "absmax",
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RetainedMassPattern]:
    """Run dense INT8 attention while constructing reusable sparse support."""
    quantized = prequantize_int8_attention(
        q,
        k,
        v,
        scale=scale,
        v_quantization=v_quantization,
        is_causal=is_causal,
        attn_mask=attn_mask,
    )
    return construct_sparse_int8_attention_from_prequantized(
        quantized,
        theta=theta,
        output_layout=output_layout,
    )


def sparse_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_pattern: RetainedMassPattern,
    *,
    scale: float | None = None,
    output_layout: str = "hnd",
    v_quantization: str = "absmax",
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run rotation-stabilized INT8 attention over frozen block support.

    Pattern construction is intentionally separate so diffusion pipelines can
    build once at an early dense step and reuse the indices thereafter.
    """
    quantized = prequantize_int8_attention(
        q,
        k,
        v,
        scale=scale,
        v_quantization=v_quantization,
        is_causal=is_causal,
        attn_mask=attn_mask,
    )
    return int8_attention_from_prequantized(
        quantized,
        sparse_pattern=sparse_pattern,
        output_layout=output_layout,
    )
