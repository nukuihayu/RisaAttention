# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure INT8 scaled dot-product attention for NVIDIA tensor-core GPUs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from . import _backend

DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

CTA_K = 64
LARGE_CTA_K = 128
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_NATIVE_MINIMUM_CAPABILITY = (7, 5)
_MIN_PATTERN_SPARSITY = 0.05
_OUTPUT_LAYOUTS = ("hnd", "nhd")


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

    @property
    def coverage(self) -> float:
        rows = (
            self.batch_size
            * self.q_heads
            * math.ceil(self.q_length / self.query_block_size)
        )
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


def _pad_to_cta_k(length: int, cta_k: int = CTA_K) -> int:
    return ((length + cta_k - 1) // cta_k) * cta_k


def _select_cta_k(
    kernel_head_dim: int,
    kv_length: int,
    *,
    has_mask: bool,
) -> int:
    if not has_mask and kernel_head_dim >= 128 and kv_length > 1024:
        return LARGE_CTA_K
    return CTA_K


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether the compiled INT8 attention kernel supports this GPU."""
    if not torch.cuda.is_available():
        return False
    return _backend.is_available(device)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape [batch, heads, sequence, head_dim]"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"q, k, and v must be float32, float16, or bfloat16, got {q.dtype}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, and {v.dtype}"
        )
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "INT8 attention requires the risa_attention CUDA extension and an NVIDIA SM75+ GPU"
        )

    batch, q_heads, q_length, head_dim = q.shape
    k_batch, kv_heads, kv_length, k_head_dim = k.shape
    if v.shape != (batch, kv_heads, kv_length, head_dim):
        raise ValueError(
            f"v must have shape [q.batch, k.heads, k.sequence, q.head_dim], got {tuple(v.shape)}"
        )
    if k_batch != batch or k_head_dim != head_dim:
        raise ValueError(
            f"q and k batch/head dimensions must match, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if batch == 0 or q_heads == 0 or kv_heads == 0 or q_length == 0 or kv_length == 0:
        raise ValueError("batch, head counts, and sequence lengths must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"q head count ({q_heads}) must be divisible by k/v head count ({kv_heads})"
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
        return torch.broadcast_to(attn_mask, (batch, q_heads, q_length, kv_length))
    except RuntimeError as error:
        raise ValueError(
            "attn_mask must be broadcastable to "
            f"[{batch}, {q_heads}, {q_length}, {kv_length}], got {tuple(attn_mask.shape)}"
        ) from error


def _int8_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_mask = _validate_inputs(q, k, v, attn_mask)

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

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None,
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim, dtype=torch.float32, device=q.device
    )
    v_center = torch.empty_like(v_scale)

    output_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    output = torch.empty(
        batch, q_heads, q_length, kernel_head_dim, dtype=output_dtype, device=q.device
    )

    anchor_indices = torch.empty(batch, kv_heads, dtype=torch.int32, device=q.device)
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    if attn_mask is None:
        _backend._C.risa_sdpa(
            _backend.wrap_for_dlpack(q),
            _backend.wrap_for_dlpack(k),
            _backend.wrap_for_dlpack(v),
            _backend.wrap_for_dlpack(output),
            _backend.wrap_for_dlpack(q_int8),
            _backend.wrap_for_dlpack(q_scale),
            _backend.wrap_for_dlpack(k_int8),
            _backend.wrap_for_dlpack(k_scale),
            _backend.wrap_for_dlpack(v_int8),
            _backend.wrap_for_dlpack(v_scale),
            _backend.wrap_for_dlpack(v_center),
            attention_scale,
            DTYPE_TO_CODE[q.dtype],
            DTYPE_TO_CODE[output_dtype],
            stream_ptr,
            anchor_indices_ptr,
            cta_k=cta_k,
        )
    else:
        _backend._C.risa_sdpa(
            _backend.wrap_for_dlpack(q),
            _backend.wrap_for_dlpack(k),
            _backend.wrap_for_dlpack(v),
            _backend.wrap_for_dlpack(output),
            _backend.wrap_for_dlpack(q_int8),
            _backend.wrap_for_dlpack(q_scale),
            _backend.wrap_for_dlpack(k_int8),
            _backend.wrap_for_dlpack(k_scale),
            _backend.wrap_for_dlpack(v_int8),
            _backend.wrap_for_dlpack(v_scale),
            _backend.wrap_for_dlpack(v_center),
            attention_scale,
            DTYPE_TO_CODE[q.dtype],
            DTYPE_TO_CODE[output_dtype],
            stream_ptr,
            anchor_indices_ptr,
            _backend.wrap_for_dlpack(attn_mask),
            cta_k=cta_k,
        )

    output = output[..., :original_head_dim]
    return output.float() if q.dtype == torch.float32 else output


@torch.inference_mode()
def build_retained_mass_pattern(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    theta: float = 0.99,
    scale: float | None = None,
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
    if q.dtype not in _SUPPORTED_DTYPES or q.dtype != k.dtype:
        raise TypeError(
            "q and k must have the same float32, float16, or bfloat16 dtype"
        )
    if not q.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "sparse INT8 attention requires the CUDA extension and an SM75+ GPU"
        )
    batch, q_heads, q_length, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2] != q_length or k.shape[3] != head_dim:
        raise ValueError(
            "sparse attention requires self-attention with equal Q/K lengths"
        )
    kv_heads = k.shape[1]
    if batch == 0 or q_heads == 0 or kv_heads == 0 or q_length == 0:
        raise ValueError("batch, head counts, and sequence length must be positive")
    if q_heads % kv_heads:
        raise ValueError("q head count must be divisible by k head count")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1, 256]")
    attention_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError("scale must be finite")

    kernel_head_dim = 64 if head_dim <= 64 else 128 if head_dim <= 128 else 256
    key_block_size = _select_cta_k(kernel_head_dim, q_length, has_mask=False)
    query_block_size = 128
    num_q_blocks = math.ceil(q_length / query_block_size)
    num_k_blocks = math.ceil(q_length / key_block_size)
    group_size = q_heads // kv_heads
    k_float_t = k.float().transpose(-1, -2).unsqueeze(2)
    rows: list[list[int] | None] = [None] * (batch * q_heads * num_q_blocks)
    retained_total = 0.0
    mass_total = 0.0

    for query_block in range(num_q_blocks):
        start = query_block * query_block_size
        end = min(start + query_block_size, q_length)
        q_chunk = q[:, :, start:end].reshape(
            batch, kv_heads, group_size, end - start, head_dim
        )
        scores = torch.matmul(q_chunk.float(), k_float_t).mul_(attention_scale)
        probabilities = torch.softmax(scores, dim=-1)
        if q_length % key_block_size:
            probabilities = functional.pad(
                probabilities, (0, key_block_size - q_length % key_block_size)
            )
        masses = (
            probabilities.reshape(
                batch,
                kv_heads,
                group_size,
                end - start,
                num_k_blocks,
                key_block_size,
            )
            .sum(dim=(3, 5))
            .reshape(batch, q_heads, num_k_blocks)
        )
        sorted_mass, descending_indices = masses.sort(dim=-1, descending=True)
        target = theta * (end - start)
        counts = (sorted_mass.cumsum(dim=-1) < target).sum(dim=-1).add_(1)
        counts.clamp_(max=num_k_blocks)

        masses_cpu = masses.cpu()
        indices_cpu = descending_indices.cpu()
        counts_cpu = counts.cpu()
        for batch_index in range(batch):
            for head_index in range(q_heads):
                count = int(counts_cpu[batch_index, head_index])
                selected = indices_cpu[batch_index, head_index, :count].tolist()
                selected.sort()
                row = (batch_index * q_heads + head_index) * num_q_blocks + query_block
                rows[row] = selected
                retained_total += float(
                    masses_cpu[batch_index, head_index, selected].sum()
                )
                mass_total += end - start

    row_offsets = [0]
    flat_indices: list[int] = []
    for selected in rows:
        if not selected:
            raise RuntimeError(
                "retained-mass construction produced an empty support row"
            )
        flat_indices.extend(selected)
        row_offsets.append(len(flat_indices))

    return RetainedMassPattern(
        row_offsets=torch.tensor(row_offsets, dtype=torch.int32, device=q.device),
        block_indices=torch.tensor(flat_indices, dtype=torch.int32, device=q.device),
        batch_size=batch,
        q_heads=q_heads,
        q_length=q_length,
        kv_length=q_length,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
        attention_scale=attention_scale,
        retained_mass_target=float(theta),
        measured_retained_mass=retained_total / mass_total,
    )


@torch.inference_mode()
def measure_pattern_recall(
    q: torch.Tensor,
    k: torch.Tensor,
    sparse_pattern: RetainedMassPattern,
) -> float:
    """Measure exact dense attention mass retained by a frozen pattern."""
    if not isinstance(sparse_pattern, RetainedMassPattern):
        raise TypeError(
            "sparse_pattern must be returned by build_retained_mass_pattern"
        )
    if q.ndim != 4 or k.ndim != 4 or q.device != k.device or not q.is_cuda:
        raise ValueError("q and k must be 4D tensors on the same CUDA device")
    batch, q_heads, q_length, head_dim = q.shape
    kv_heads = k.shape[1] if k.ndim == 4 else 0
    if (
        k.shape != (batch, kv_heads, q_length, head_dim)
        or q.dtype != k.dtype
        or q_heads % kv_heads
    ):
        raise ValueError("q and k shapes/dtypes are incompatible")
    expected = (
        sparse_pattern.batch_size,
        sparse_pattern.q_heads,
        sparse_pattern.q_length,
        sparse_pattern.kv_length,
    )
    if (batch, q_heads, q_length, q_length) != expected:
        raise ValueError("q and k do not match the frozen retained-mass pattern")
    if sparse_pattern.row_offsets.device != q.device:
        raise ValueError("pattern and q/k must be on the same CUDA device")

    query_block_size = sparse_pattern.query_block_size
    key_block_size = sparse_pattern.key_block_size
    num_q_blocks = math.ceil(q_length / query_block_size)
    num_k_blocks = math.ceil(q_length / key_block_size)
    group_size = q_heads // kv_heads
    k_float_t = k.float().transpose(-1, -2).unsqueeze(2)
    offsets = sparse_pattern.row_offsets.cpu()
    indices = sparse_pattern.block_indices.cpu()
    retained_total = 0.0
    mass_total = float(batch * q_heads * q_length)

    for query_block in range(num_q_blocks):
        start = query_block * query_block_size
        end = min(start + query_block_size, q_length)
        q_chunk = q[:, :, start:end].reshape(
            batch, kv_heads, group_size, end - start, head_dim
        )
        probabilities = torch.softmax(
            torch.matmul(q_chunk.float(), k_float_t).mul_(
                sparse_pattern.attention_scale
            ),
            dim=-1,
        )
        if q_length % key_block_size:
            probabilities = functional.pad(
                probabilities, (0, key_block_size - q_length % key_block_size)
            )
        masses = (
            probabilities.reshape(
                batch,
                kv_heads,
                group_size,
                end - start,
                num_k_blocks,
                key_block_size,
            )
            .sum(dim=(3, 5))
            .reshape(batch, q_heads, num_k_blocks)
            .cpu()
        )
        for batch_index in range(batch):
            for head_index in range(q_heads):
                row = (batch_index * q_heads + head_index) * num_q_blocks + query_block
                row_start = int(offsets[row])
                row_end = int(offsets[row + 1])
                retained_total += float(
                    masses[batch_index, head_index, indices[row_start:row_end]].sum()
                )
    return retained_total / mass_total


def prequantize_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
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
    attn_mask = _validate_inputs(q, k, v, attn_mask)

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

    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    cta_k = _select_cta_k(
        kernel_head_dim,
        kv_length,
        has_mask=attn_mask is not None,
    )
    padded_k_length = _pad_to_cta_k(kv_length, cta_k)
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scales_per_block = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        q_heads,
        ((q_length + 127) // 128) * q_scales_per_block,
        dtype=torch.float32,
        device=q.device,
    )
    k_scale = torch.empty(
        batch,
        kv_heads,
        ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32,
        device=q.device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim,
        padded_k_length,
        dtype=torch.int8,
        device=q.device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    v_center = torch.empty_like(v_scale)
    anchor_indices = torch.empty(batch, kv_heads, dtype=torch.int32, device=q.device)
    anchor_indices_ptr = anchor_indices.data_ptr()

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _backend._C.risa_sdpa_quantize(
        _backend.wrap_for_dlpack(q),
        _backend.wrap_for_dlpack(k),
        _backend.wrap_for_dlpack(v),
        _backend.wrap_for_dlpack(q_int8),
        _backend.wrap_for_dlpack(q_scale),
        _backend.wrap_for_dlpack(k_int8),
        _backend.wrap_for_dlpack(k_scale),
        _backend.wrap_for_dlpack(v_int8),
        _backend.wrap_for_dlpack(v_scale),
        _backend.wrap_for_dlpack(v_center),
        cta_k,
        DTYPE_TO_CODE[input_dtype],
        stream_ptr,
        anchor_indices_ptr,
    )

    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        v_center=v_center,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        cta_k=cta_k,
        attn_mask=attn_mask,
    )


def int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
    *,
    sparse_pattern: RetainedMassPattern | None = None,
    output_layout: str = "hnd",
    _block_mass: torch.Tensor | None = None,
) -> torch.Tensor:
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
    if output_layout not in _OUTPUT_LAYOUTS:
        raise ValueError(
            f"output_layout must be one of {_OUTPUT_LAYOUTS}, got {output_layout!r}"
        )

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
        raise ValueError(
            "prequantized INT8 attention tensors must be on the same CUDA device"
        )
    if (
        quantized.attn_mask is not None
        and quantized.attn_mask.device != quantized.q.device
    ):
        raise ValueError(
            "attn_mask must be on the same CUDA device as the packed tensors"
        )
    if sparse_pattern is not None:
        if not isinstance(sparse_pattern, RetainedMassPattern):
            raise TypeError(
                "sparse_pattern must be returned by build_retained_mass_pattern"
            )
        if quantized.attn_mask is not None:
            raise ValueError("sparse support cannot be combined with an attention mask")
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
            raise ValueError(
                "sparse pattern and attention call must use the same scale"
            )
        if (
            sparse_pattern.row_offsets.device != quantized.q.device
            or sparse_pattern.block_indices.device != quantized.q.device
        ):
            raise ValueError(
                "sparse pattern and packed tensors must be on the same CUDA device"
            )
        if (
            sparse_pattern.row_offsets.dtype != torch.int32
            or sparse_pattern.block_indices.dtype != torch.int32
            or not sparse_pattern.row_offsets.is_contiguous()
            or not sparse_pattern.block_indices.is_contiguous()
        ):
            raise TypeError(
                "sparse CSR row offsets and block indices must be contiguous int32"
            )
        if sparse_pattern.sparsity < _MIN_PATTERN_SPARSITY:
            sparse_pattern = None
    if _block_mass is not None:
        if sparse_pattern is not None or quantized.attn_mask is not None:
            raise ValueError("block mass requires dense unmasked attention")
        expected_mass_shape = (
            quantized.q.shape[0]
            * quantized.q.shape[1]
            * math.ceil(quantized.q.shape[2] / 128),
            math.ceil(quantized.k.shape[2] / quantized.cta_k),
        )
        if (
            _block_mass.shape != expected_mass_shape
            or _block_mass.dtype != torch.float32
            or _block_mass.device != quantized.q.device
            or not _block_mass.is_contiguous()
        ):
            raise ValueError(
                f"block mass must be contiguous CUDA float32 {expected_mass_shape}"
            )
    if not is_available(quantized.q.device):
        raise RuntimeError(
            "INT8 attention requires the risa_attention CUDA extension and an NVIDIA SM75+ GPU"
        )

    batch, q_heads, q_length, kernel_head_dim = quantized.q.shape
    output_dtype = (
        torch.bfloat16
        if quantized.input_dtype == torch.float32
        else quantized.input_dtype
    )

    if output_layout == "nhd":
        output = torch.empty(
            batch,
            q_length,
            q_heads,
            kernel_head_dim,
            dtype=output_dtype,
            device=quantized.q.device,
        ).permute(0, 2, 1, 3)
    else:
        output = torch.empty(
            batch,
            q_heads,
            q_length,
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
        DTYPE_TO_CODE[output_dtype],
        stream_ptr,
    )
    if _block_mass is not None:
        _backend._C.risa_sdpa_prequantized(
            *arguments,
            None,
            None,
            None,
            _backend.wrap_for_dlpack(_block_mass),
        )
    elif sparse_pattern is not None:
        _backend._C.risa_sdpa_prequantized(
            *arguments,
            None,
            _backend.wrap_for_dlpack(sparse_pattern.row_offsets),
            _backend.wrap_for_dlpack(sparse_pattern.block_indices),
        )
    elif quantized.attn_mask is None:
        _backend._C.risa_sdpa_prequantized(*arguments)
    else:
        _backend._C.risa_sdpa_prequantized(
            *arguments,
            _backend.wrap_for_dlpack(quantized.attn_mask),
        )

    output = output[..., : quantized.original_head_dim]
    return output.float() if quantized.input_dtype == torch.float32 else output


@torch.inference_mode()
def construct_sparse_int8_attention_from_prequantized(
    quantized: PrequantizedInt8Attention,
    *,
    theta: float = 0.99,
    output_layout: str = "hnd",
) -> tuple[torch.Tensor, RetainedMassPattern]:
    """Construct reusable retained-mass support while consuming packed Q/K/V.

    The CUDA attention kernel reuses its final online-softmax state and
    recomputes QK once to emit block masses. Selection and CSR construction
    stay on the GPU. The masses follow Sage's quantized U8 probability path;
    use :func:`measure_pattern_recall` when exact FP attention recall is
    required for evaluation.

    ``theta`` is the retained-mass threshold.
    """
    if not 0.0 < theta <= 1.0 or not math.isfinite(theta):
        raise ValueError("theta must be finite and in (0, 1]")
    if not isinstance(quantized, PrequantizedInt8Attention):
        raise TypeError(
            "quantized must be returned by prequantize_int8_attention, got "
            f"{type(quantized).__name__}"
        )
    if quantized.q.shape[2] != quantized.k.shape[2]:
        raise ValueError(
            "sparse attention requires self-attention with equal Q/K lengths"
        )

    batch, q_heads, q_length, _ = quantized.q.shape
    num_q_blocks = math.ceil(q_length / 128)
    num_k_blocks = math.ceil(quantized.k.shape[2] / quantized.cta_k)
    masses = torch.empty(
        batch * q_heads * num_q_blocks,
        num_k_blocks,
        dtype=torch.float32,
        device=quantized.q.device,
    )
    output = int8_attention_from_prequantized(
        quantized,
        output_layout=output_layout,
        _block_mass=masses,
    )

    sorted_mass, descending_indices = masses.sort(dim=-1, descending=True)
    row_mass = masses.sum(dim=-1)
    thresholds = row_mass * theta
    counts = (sorted_mass.cumsum(dim=-1) < thresholds[:, None]).sum(dim=-1) + 1
    counts.clamp_(max=num_k_blocks)
    slots = torch.arange(num_k_blocks, device=masses.device)[None, :]
    selected = (
        torch.where(slots < counts[:, None], descending_indices, num_k_blocks)
        .sort(dim=-1)
        .values
    )
    selected_mask = selected < num_k_blocks
    block_indices = selected[selected_mask].to(torch.int32)
    row_offsets = torch.empty(
        masses.shape[0] + 1, dtype=torch.int32, device=masses.device
    )
    row_offsets[0] = 0
    row_offsets[1:] = counts.cumsum(dim=0).to(torch.int32)
    selected_mass = (
        masses.gather(1, descending_indices)
        .masked_fill(slots >= counts[:, None], 0.0)
        .sum()
    )
    measured_mass = float((selected_mass / row_mass.sum().clamp_min(1e-20)).item())

    pattern = RetainedMassPattern(
        row_offsets=row_offsets,
        block_indices=block_indices,
        batch_size=batch,
        q_heads=q_heads,
        q_length=q_length,
        kv_length=quantized.k.shape[2],
        query_block_size=128,
        key_block_size=quantized.cta_k,
        attention_scale=quantized.attention_scale,
        retained_mass_target=float(theta),
        measured_retained_mass=measured_mass,
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
) -> tuple[torch.Tensor, RetainedMassPattern]:
    """Run dense INT8 attention while constructing reusable sparse support."""
    quantized = prequantize_int8_attention(q, k, v, scale=scale)
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
) -> torch.Tensor:
    """Run rotation-stabilized INT8 attention over frozen block support.

    Pattern construction is intentionally separate so diffusion pipelines can
    build once at an early dense step and reuse the indices thereafter.
    """
    quantized = prequantize_int8_attention(q, k, v, scale=scale)
    return int8_attention_from_prequantized(
        quantized,
        sparse_pattern=sparse_pattern,
        output_layout=output_layout,
    )


@torch.library.custom_op("risa_attention::int8_attention", mutates_args=())
def _op_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=None,
    )


@_op_int8_attention.register_fake
def _op_int8_attention_fake(
    q,
    k,
    v,
    scale,
):
    return q.new_empty(q.shape)


@torch.library.custom_op("risa_attention::int8_attention_masked", mutates_args=())
def _op_int8_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _int8_attention_cuda(
        q,
        k,
        v,
        scale=scale,
        attn_mask=attn_mask,
    )


@_op_int8_attention_masked.register_fake
def _op_int8_attention_masked_fake(
    q,
    k,
    v,
    attn_mask,
    scale,
):
    return q.new_empty(q.shape)


def int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
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
    V uses per-channel midpoint-centered INT8 quantization; adding the center
    after probability normalization preserves the effective U8 attention mass
    while reducing the quantization step for asymmetric value ranges. Softmax
    score, maximum, exponential, denominator, reciprocal, and V dequantization
    arithmetic is FP32. This path does not allocate FP8 tensors or execute FP8
    MMA instructions.
    """
    if attn_mask is None:
        return torch.ops.risa_attention.int8_attention(
            q,
            k,
            v,
            scale,
        )
    return torch.ops.risa_attention.int8_attention_masked(
        q,
        k,
        v,
        attn_mask,
        scale,
    )
