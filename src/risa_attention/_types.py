# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Packed tensors and immutable sparse support metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PrequantizedInt8Attention:
    """Packed Q/K/V and immutable launch metadata for split INT8 attention.

    Instances own only the quantized tensors, their scales, V midpoint centers,
    and an optional attention mask. They never retain the floating-point Q, K,
    or V inputs.
    Create instances with :func:`prequantize_int8_attention` rather than
    constructing them directly.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    v_center: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    attn_mask: torch.Tensor | None
    v_quantization: str = "absmax"
    is_causal: bool = False
    lse_shift: torch.Tensor | None = None
    kv_permuted: bool = False


@dataclass(frozen=True, slots=True)
class RetainedMassPattern:
    """Frozen per-head block support in CUDA CSR form.

    Build instances with :func:`build_retained_mass_pattern`. Rows are ordered by
    ``[batch, query_head, query_block]`` and contain sorted key tile indices.
    """

    row_offsets: torch.Tensor
    block_indices: torch.Tensor
    batch_size: int
    q_heads: int
    q_length: int
    kv_length: int
    query_block_size: int
    key_block_size: int
    attention_scale: float
    retained_mass_target: float
    measured_retained_mass: float
    is_causal: bool = False
    mask_signature: tuple | None = None
    mask_reference: torch.Tensor | None = None

    @property
    def coverage(self) -> float:
        rows = self.batch_size * self.q_heads * math.ceil(self.q_length / self.query_block_size)
        key_blocks = math.ceil(self.kv_length / self.key_block_size)
        return self.block_indices.numel() / (rows * key_blocks)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.coverage

    @property
    def index_bytes(self) -> int:
        return self.row_offsets.numel() * self.row_offsets.element_size() + (
            self.block_indices.numel() * self.block_indices.element_size()
        )
