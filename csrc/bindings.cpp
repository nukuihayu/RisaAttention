/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernels/quantization_input.cuh"
#include <climits>
#include <cstring>
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

// Helper: Map nanobind dtype to internal dtype code
// Returns: 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
  if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Float) {
    if (dtype.bits == 32)
      return 0; // float32
    if (dtype.bits == 16)
      return 1; // float16
    if (dtype.bits == 8)
      return 5; // float8_e4m3fn (default)
  } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Bfloat && dtype.bits == 16) {
    return 2; // bfloat16
  } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::UInt && dtype.bits == 8) {
    return 3; // uint8
  } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Int && dtype.bits == 8) {
    return 4; // int8
  }
  return -1; // unsupported
}

#include "launchers.h"

const int32_t* validate_key_mask_tiles(const std::optional<nb::ndarray<nb::device::cuda>>& tiles,
                                       int batch_size, int query_heads, int key_length,
                                       int tile_size) {
  if (!tiles.has_value())
    return nullptr;
  const auto& value = *tiles;
  const int width = (key_length + tile_size - 1) / tile_size + 1;
  if (value.ndim() != 3 || value.shape(0) != static_cast<size_t>(batch_size) ||
      value.shape(1) != static_cast<size_t>(query_heads) ||
      value.shape(2) != static_cast<size_t>(width) ||
      value.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
      value.dtype().bits != 32 || value.stride(2) != 1 || value.stride(1) != width ||
      value.stride(0) != int64_t(query_heads) * width) {
    throw std::runtime_error("key mask tiles must be contiguous CUDA int32 [B,Hq,tiles+1]");
  }
  return static_cast<const int32_t*>(value.data());
}

void prepare_key_mask_tiles(nb::ndarray<nb::device::cuda> mask, nb::ndarray<nb::device::cuda> tiles,
                            int tile_size, uintptr_t stream_ptr) {
  if (mask.ndim() != 4 || mask.stride(2) != 0 || (tile_size != 64 && tile_size != 128)) {
    throw std::runtime_error("prepare_key_mask_tiles requires an expanded broadcast key mask");
  }
  const int batch_size = mask.shape(0), heads = mask.shape(1), length = mask.shape(3);
  const auto* output = validate_key_mask_tiles(tiles, batch_size, heads, length, tile_size);
  const int dtype = mask.dtype().code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool)
                        ? 3
                        : map_dtype_to_code(mask.dtype());
  if (dtype < 0 || dtype > 3 || mask.device_id() != tiles.device_id()) {
    throw std::runtime_error(
        "prepare_key_mask_tiles requires matching CUDA devices and a supported mask dtype");
  }
  launch_prepare_key_mask_tiles(mask.data(), const_cast<int32_t*>(output), batch_size, heads,
                                length, tile_size, mask.stride(0), mask.stride(1), mask.stride(3),
                                dtype, reinterpret_cast<cudaStream_t>(stream_ptr));
}

void select_retained_mass_csr(
    nb::ndarray<nb::device::cuda> sorted_mass, nb::ndarray<nb::device::cuda> descending_indices,
    nb::ndarray<nb::device::cuda> counts, nb::ndarray<nb::device::cuda> support,
    nb::ndarray<nb::device::cuda> row_totals, nb::ndarray<nb::device::cuda> row_offsets,
    nb::ndarray<nb::device::cuda> block_indices, nb::ndarray<nb::device::cuda> summary,
    nb::ndarray<nb::device::cuda> scan_workspace, double theta, uintptr_t stream_ptr) {
  if (sorted_mass.ndim() != 2 || descending_indices.ndim() != 2) {
    throw std::runtime_error("select_retained_mass_csr: masses and indices must be 2D");
  }
  const size_t rows = sorted_mass.shape(0);
  const size_t columns = sorted_mass.shape(1);
  const auto float_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
  const auto int_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
  const auto uint_code = static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
  const bool valid =
      rows > 0 && columns > 0 && sorted_mass.dtype().code == float_code &&
      sorted_mass.dtype().bits == 32 && sorted_mass.stride(1) == 1 &&
      sorted_mass.stride(0) == static_cast<int64_t>(columns) &&
      descending_indices.shape(0) == rows && descending_indices.shape(1) == columns &&
      descending_indices.dtype().code == int_code && descending_indices.dtype().bits == 64 &&
      descending_indices.stride(1) == 1 &&
      descending_indices.stride(0) == static_cast<int64_t>(columns) && counts.ndim() == 1 &&
      counts.shape(0) == rows && counts.dtype().code == int_code && counts.dtype().bits == 32 &&
      counts.stride(0) == 1 && support.ndim() == 2 && support.shape(0) == rows &&
      support.shape(1) == columns && support.dtype().code == uint_code &&
      support.dtype().bits == 8 && support.stride(1) == 1 &&
      support.stride(0) == static_cast<int64_t>(columns) && row_totals.ndim() == 2 &&
      row_totals.shape(0) == rows && row_totals.shape(1) == 2 &&
      row_totals.dtype().code == float_code && row_totals.dtype().bits == 64 &&
      row_totals.stride(1) == 1 && row_totals.stride(0) == 2 && row_offsets.ndim() == 1 &&
      row_offsets.shape(0) == rows + 1 && row_offsets.dtype().code == int_code &&
      row_offsets.dtype().bits == 32 && row_offsets.stride(0) == 1 && block_indices.ndim() == 1 &&
      block_indices.shape(0) == rows * columns && block_indices.dtype().code == int_code &&
      block_indices.dtype().bits == 32 && block_indices.stride(0) == 1 && summary.ndim() == 1 &&
      summary.shape(0) == 3 && summary.dtype().code == float_code && summary.dtype().bits == 64 &&
      summary.stride(0) == 1 && scan_workspace.ndim() == 1 &&
      scan_workspace.dtype().code == uint_code && scan_workspace.dtype().bits == 8 &&
      scan_workspace.stride(0) == 1;
  if (!valid) {
    throw std::runtime_error("select_retained_mass_csr: invalid shape, dtype, or stride");
  }
  if (rows > static_cast<size_t>(INT_MAX) || columns > static_cast<size_t>(INT_MAX) ||
      rows > static_cast<size_t>(INT_MAX) / columns) {
    throw std::runtime_error("select_retained_mass_csr: matrix is too large");
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const size_t required_workspace = retained_mass_selector_workspace_size(static_cast<int>(rows));
  if (scan_workspace.shape(0) < required_workspace) {
    throw std::runtime_error("select_retained_mass_csr: scan workspace is too small");
  }
  launch_select_retained_mass(
      static_cast<const float*>(sorted_mass.data()),
      static_cast<const int64_t*>(descending_indices.data()), static_cast<int32_t*>(counts.data()),
      static_cast<uint8_t*>(support.data()), static_cast<double*>(row_totals.data()),
      static_cast<int32_t*>(row_offsets.data()), static_cast<int32_t*>(block_indices.data()),
      static_cast<double*>(summary.data()), scan_workspace.data(), scan_workspace.shape(0),
      static_cast<int>(rows), static_cast<int>(columns), theta, stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("retained-mass selector launch failed: ") +
                             cudaGetErrorString(err));
  }
}

// Nanobind wrapper: signed INT8 V quantization

void quantize_v(nb::ndarray<nb::device::cuda> v, nb::ndarray<nb::device::cuda> out,
                nb::ndarray<nb::device::cuda> scale, nb::ndarray<nb::device::cuda> center,
                int padded_n, int input_dtype_code, bool use_midpoint_affine,
                uintptr_t stream_ptr) {
  if (v.ndim() != 4) {
    throw std::runtime_error(
        "quantize_v: v must be 4D [batch_size,num_heads,sequence_length,head_dim]");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_quantize_v(v.data(), out.data(), scale.data(), center.data(), static_cast<int>(v.shape(0)),
                    static_cast<int>(v.shape(1)), static_cast<int>(v.shape(2)),
                    static_cast<int>(v.shape(3)), padded_n, v.stride(0), v.stride(1), v.stride(2),
                    input_dtype_code, use_midpoint_affine, stream);
}

// Nanobind wrapper: stabilized INT8 Q/K per-thread quant (contiguous HND layout)
void quant_qk_per_thread_int8(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> q_int8,
                              nb::ndarray<nb::device::cuda> q_scale,
                              nb::ndarray<nb::device::cuda> k, nb::ndarray<nb::device::cuda> k_int8,
                              nb::ndarray<nb::device::cuda> k_scale, int BLKQ, int WARPQ, int BLKK,
                              int WARPK, int input_dtype_code, uintptr_t stream_ptr,
                              uintptr_t anchor_indices_ptr) {
  if (q.ndim() != 4 || k.ndim() != 4) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: q and k must be 4D [batch_size,num_heads,L,head_dim]");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_quantize_qk(
      q.data(), q_int8.data(), q_scale.data(), k.data(), k_int8.data(), k_scale.data(),
      static_cast<int>(q.shape(0)), static_cast<int>(q.shape(1)), static_cast<int>(q.shape(2)),
      static_cast<int>(k.shape(1)), static_cast<int>(k.shape(2)), static_cast<int>(q.shape(3)),
      BLKQ, WARPQ, BLKK, WARPK, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
      k.stride(2), input_dtype_code, reinterpret_cast<void*>(anchor_indices_ptr), stream);
}

// Quantization half of the split INT8 SDPA API.  This deliberately launches
// the same Q/K and V kernels with the same tiling as risa_sdpa below, so moving
// the attention launch after the caller releases its input tensors does not
// change any numerical results.
void risa_sdpa_quantize(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
                        nb::ndarray<nb::device::cuda> v, nb::ndarray<nb::device::cuda> q_int8,
                        nb::ndarray<nb::device::cuda> q_scale, nb::ndarray<nb::device::cuda> k_int8,
                        nb::ndarray<nb::device::cuda> k_scale, nb::ndarray<nb::device::cuda> v_int8,
                        nb::ndarray<nb::device::cuda> v_scale,
                        nb::ndarray<nb::device::cuda> v_center, int cta_k, int input_dtype_code,
                        bool use_midpoint_affine, uintptr_t stream_ptr,
                        uintptr_t anchor_indices_ptr, uintptr_t permutation_ptr) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::runtime_error(
        "risa_sdpa_quantize: q, k, and v must be 4D [batch_size,num_heads,L,head_dim]");
  }
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("risa_sdpa_quantize: cta_k must be 64 or 128");
  }
  if (input_dtype_code < 0 || input_dtype_code > 2) {
    throw std::runtime_error(
        "risa_sdpa_quantize: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
  }
  if (!anchor_indices_ptr) {
    throw std::runtime_error("risa_sdpa_quantize: anchor_indices scratch is required");
  }

  const int batch_size = static_cast<int>(q.shape(0));
  const int query_heads = static_cast<int>(q.shape(1));
  const int query_length = static_cast<int>(q.shape(2));
  const int head_dim = static_cast<int>(q.shape(3));
  const int key_value_heads = static_cast<int>(k.shape(1));
  const int key_length = static_cast<int>(k.shape(2));
  const int padded_Lk = ((key_length + cta_k - 1) / cta_k) * cta_k;

  const bool transformed = permutation_ptr != 0;
  if (transformed && (head_dim != 128 || input_dtype_code != 2)) {
    throw std::runtime_error("fused gather requires BF16 head_dim 128");
  }
  risa::QuantInput qt{};
  risa::QuantInput kt{reinterpret_cast<const int32_t*>(permutation_ptr)};
  risa::QuantInput vt = kt;

  if (cta_k == 128 && head_dim == 64) {
    throw std::runtime_error("risa_sdpa_quantize: cta_k 128 is unsupported for head_dim 64");
  }

  if (k.shape(0) != batch_size || v.shape(0) != batch_size || v.shape(1) != key_value_heads ||
      v.shape(2) != key_length || k.shape(3) != head_dim || v.shape(3) != head_dim) {
    throw std::runtime_error("risa_sdpa_quantize: incompatible q, k, and v shapes");
  }
  if (q_int8.ndim() != 4 || k_int8.ndim() != 4 || v_int8.ndim() != 2 ||
      q_int8.shape(0) != batch_size || q_int8.shape(1) != query_heads ||
      q_int8.shape(2) != query_length || q_int8.shape(3) != head_dim ||
      k_int8.shape(0) != batch_size || k_int8.shape(1) != key_value_heads ||
      k_int8.shape(2) != key_length || k_int8.shape(3) != head_dim ||
      v_int8.shape(0) != static_cast<size_t>(batch_size) * key_value_heads * head_dim ||
      v_int8.shape(1) != padded_Lk) {
    throw std::runtime_error("risa_sdpa_quantize: incompatible INT8 output shapes");
  }

  constexpr int BLKQ = 128;
  const int WARPQ = head_dim == 256 ? 16 : 32;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  (transformed ? launch_quantize_qk_gather : launch_quantize_qk)(
      q.data(), q_int8.data(), q_scale.data(), k.data(), k_int8.data(), k_scale.data(), batch_size,
      query_heads, query_length, key_value_heads, key_length, head_dim, BLKQ, WARPQ, cta_k, cta_k,
      q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
      input_dtype_code, reinterpret_cast<void*>(anchor_indices_ptr), stream, qt, kt);

  (permutation_ptr ? launch_quantize_v_gather : launch_quantize_v)(
      v.data(), v_int8.data(), v_scale.data(), v_center.data(), batch_size, key_value_heads,
      key_length, head_dim, padded_Lk, v.stride(0), v.stride(1), v.stride(2), input_dtype_code,
      use_midpoint_affine, stream, vt);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("risa_sdpa_quantize kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

// Attention half of the split INT8 SDPA API.  The input tensors use the exact
// packed layouts produced by risa_sdpa_quantize; no floating-point Q/K/V
// tensor is retained or reconstructed.
void risa_sdpa_prequantized(
    nb::ndarray<nb::device::cuda> q_int8, nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> v_int8, nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale, nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale, nb::ndarray<nb::device::cuda> v_center, int cta_k,
    float sm_scale, int output_dtype_code, uintptr_t stream_ptr,
    std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> sparse_row_offsets = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> sparse_block_indices = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> block_mass = std::nullopt, bool is_causal = false,
    std::optional<nb::ndarray<nb::device::cuda>> lse = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> key_mask_tiles = std::nullopt) {
  if (q_int8.ndim() != 4 || k_int8.ndim() != 4 || v_int8.ndim() != 2 || o.ndim() != 4) {
    throw std::runtime_error("risa_sdpa_prequantized: q/k/o must be 4D and packed v must be 2D");
  }
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("risa_sdpa_prequantized: cta_k must be 64 or 128");
  }
  if (output_dtype_code != 1 && output_dtype_code != 2) {
    throw std::runtime_error(
        "risa_sdpa_prequantized: output_dtype_code must be 1 (fp16) or 2 (bf16)");
  }

  const int batch_size = static_cast<int>(q_int8.shape(0));
  const int query_heads = static_cast<int>(q_int8.shape(1));
  const int query_length = static_cast<int>(q_int8.shape(2));
  const int head_dim = static_cast<int>(q_int8.shape(3));
  const int key_value_heads = static_cast<int>(k_int8.shape(1));
  const int key_length = static_cast<int>(k_int8.shape(2));
  const int padded_Lk = ((key_length + cta_k - 1) / cta_k) * cta_k;

  if (cta_k == 128 && head_dim == 64) {
    throw std::runtime_error("risa_sdpa_prequantized: cta_k 128 requires head_dim 128 or 256");
  }
  if (sparse_row_offsets.has_value() != sparse_block_indices.has_value()) {
    throw std::runtime_error(
        "risa_sdpa_prequantized: sparse row offsets and block indices must be provided together");
  }

  if (block_mass.has_value() && sparse_row_offsets.has_value()) {
    throw std::runtime_error("risa_sdpa_prequantized: block mass requires dense attention");
  }
  if (k_int8.shape(0) != batch_size || k_int8.shape(3) != head_dim || o.shape(0) != batch_size ||
      o.shape(1) != query_heads || o.shape(2) != query_length || o.shape(3) != head_dim ||
      v_int8.shape(0) != static_cast<size_t>(batch_size) * key_value_heads * head_dim ||
      v_int8.shape(1) != padded_Lk) {
    throw std::runtime_error("risa_sdpa_prequantized: incompatible quantized tensor shapes");
  }
  if (lse.has_value()) {
    const auto& value = lse.value();
    if (value.ndim() != 3 || value.shape(0) != batch_size || value.shape(1) != query_heads ||
        value.shape(2) != query_length || map_dtype_to_code(value.dtype()) != 0 ||
        (query_length > 1 && value.stride(2) != 1) ||
        (query_heads > 1 && value.stride(1) != query_length) ||
        (batch_size > 1 && value.stride(0) != static_cast<int64_t>(query_heads) * query_length)) {
      throw std::runtime_error(
          "risa_sdpa_prequantized: LSE must be contiguous float32 [batch_size,Hq,query_length]");
    }
  }

  const bool output_hnd =
      (query_length <= 1 || o.stride(2) == head_dim) &&
      (query_heads <= 1 || o.stride(1) == static_cast<int64_t>(query_length) * head_dim);
  const bool output_nhd =
      (query_heads <= 1 || o.stride(1) == head_dim) &&
      (query_length <= 1 || o.stride(2) == static_cast<int64_t>(query_heads) * head_dim);
  if (q_int8.stride(3) != 1 || q_int8.stride(2) != head_dim ||
      q_int8.stride(1) != static_cast<int64_t>(query_length) * head_dim || k_int8.stride(3) != 1 ||
      k_int8.stride(2) != head_dim ||
      k_int8.stride(1) != static_cast<int64_t>(key_length) * head_dim || v_int8.stride(1) != 1 ||
      v_int8.stride(0) != padded_Lk || o.stride(3) != 1 || (!output_hnd && !output_nhd)) {
    throw std::runtime_error("risa_sdpa_prequantized: quantized tensors must be contiguous and "
                             "output must use HND- or NHD-contiguous storage");
  }

  const void* mask_ptr = nullptr;
  int64_t mask_stride_b = 0;
  int64_t mask_stride_h = 0;
  int64_t mask_stride_q = 0;
  int64_t mask_stride_k = 0;
  int mask_dtype_code = -1;
  if (attn_mask.has_value()) {
    const auto& mask = attn_mask.value();
    if (mask.ndim() != 4 || mask.shape(0) != batch_size || mask.shape(1) != query_heads ||
        mask.shape(2) != query_length || mask.shape(3) != key_length) {
      throw std::runtime_error("risa_sdpa_prequantized: attention mask must be expanded to "
                               "[batch_size,query_heads,query_length,key_length]");
    }
    if (mask.dtype().code == (uint8_t)nb::dlpack::dtype_code::Bool) {
      mask_dtype_code = 3;
    } else {
      mask_dtype_code = map_dtype_to_code(mask.dtype());
    }
    if (mask_dtype_code < 0 || mask_dtype_code > 3) {
      throw std::runtime_error(
          "risa_sdpa_prequantized: attention mask must be bool, float16, bfloat16, or float32");
    }
    mask_ptr = mask.data();
    mask_stride_b = mask.stride(0);
    mask_stride_h = mask.stride(1);
    mask_stride_q = mask.stride(2);
    mask_stride_k = mask.stride(3);
  }

  const void* sparse_row_offsets_ptr = nullptr;
  const void* sparse_block_indices_ptr = nullptr;
  if (sparse_row_offsets.has_value()) {
    const auto& row_offsets = sparse_row_offsets.value();
    const auto& block_indices = sparse_block_indices.value();
    const int num_q_blocks = (query_length + 127) / 128;
    const int num_k_blocks = (key_length + cta_k - 1) / cta_k;
    const int64_t expected_rows = static_cast<int64_t>(batch_size) * query_heads * num_q_blocks;
    const auto int_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    if (row_offsets.ndim() != 1 || row_offsets.shape(0) != static_cast<size_t>(expected_rows + 1) ||
        block_indices.ndim() != 1 || row_offsets.dtype().code != int_code ||
        row_offsets.dtype().bits != 32 || block_indices.dtype().code != int_code ||
        block_indices.dtype().bits != 32) {
      throw std::runtime_error("risa_sdpa_prequantized: sparse support must be contiguous int32 "
                               "CSR with batch_size*Hq*ceil(query_length/128)+1 row offsets");
    }
    if (row_offsets.stride(0) != 1 || block_indices.stride(0) != 1 || block_indices.shape(0) == 0 ||
        num_k_blocks <= 0) {
      throw std::runtime_error(
          "risa_sdpa_prequantized: sparse CSR tensors must be non-empty and contiguous");
    }
    sparse_row_offsets_ptr = row_offsets.data();
    sparse_block_indices_ptr = block_indices.data();
  }

  void* block_mass_ptr = nullptr;
  if (block_mass.has_value()) {
    auto& mass = block_mass.value();
    const int num_q_blocks = (query_length + 127) / 128;
    const int num_k_blocks = (key_length + cta_k - 1) / cta_k;
    const size_t mass_elements =
        static_cast<size_t>(batch_size) * query_heads * num_q_blocks * num_k_blocks;
    const size_t stats_elements =
        static_cast<size_t>(batch_size) * query_heads * query_length * (num_k_blocks * 2 + 2);
    const auto float_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
    if (mass.ndim() != 1 || mass.shape(0) != mass_elements + stats_elements ||
        mass.dtype().code != float_code || mass.dtype().bits != 32 || mass.stride(0) != 1) {
      throw std::runtime_error("risa_sdpa_prequantized: block_mass workspace must contain "
                               "mass rows followed by captured tile statistics and softmax state");
    }
    block_mass_ptr = mass.data();
  }

  const int64_t qi_st_bz64 = static_cast<int64_t>(query_heads) * query_length * head_dim;
  const int64_t ki_st_bz64 = static_cast<int64_t>(key_value_heads) * key_length * head_dim;
  const int64_t v_st_bz64 = static_cast<int64_t>(key_value_heads) * head_dim * padded_Lk;
  const int64_t o_st_bz64 =
      batch_size <= 1 ? static_cast<int64_t>(query_heads) * query_length * head_dim : o.stride(0);
  const int64_t o_st_n64 = query_length <= 1 ? head_dim : o.stride(2);
  const int64_t o_st_h64 =
      query_heads <= 1 ? static_cast<int64_t>(query_length) * head_dim : o.stride(1);

  const int64_t qi_st_h = static_cast<int64_t>(query_length) * head_dim;
  const int64_t qi_st_n = head_dim;
  const int64_t qi_st_bz = qi_st_bz64;
  const int64_t ki_st_h = static_cast<int64_t>(key_length) * head_dim;
  const int64_t ki_st_n = head_dim;
  const int64_t ki_st_bz = ki_st_bz64;
  const int64_t v_st_d = padded_Lk;
  const int64_t v_st_h = static_cast<int64_t>(head_dim) * padded_Lk;
  const int64_t v_st_bz = v_st_bz64;
  const int64_t o_st_h = o_st_h64;
  const int64_t o_st_n = o_st_n64;
  const int64_t o_st_bz = o_st_bz64;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_attention(
      q_int8.data(), k_int8.data(), v_int8.data(), o.data(), q_scale.data(), k_scale.data(),
      v_scale.data(), v_center.data(), mask_ptr, mask_stride_b, mask_stride_h, mask_stride_q,
      mask_stride_k, mask_dtype_code, cta_k, batch_size, query_length, key_length, query_heads,
      key_value_heads, head_dim, qi_st_bz, qi_st_n, qi_st_h, ki_st_bz, ki_st_n, ki_st_h, v_st_bz,
      v_st_h, v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale, output_dtype_code, sparse_row_offsets_ptr,
      sparse_block_indices_ptr, block_mass_ptr, is_causal, lse.has_value() ? lse->data() : nullptr,
      stream, validate_key_mask_tiles(key_mask_tiles, batch_size, query_heads, key_length, cta_k));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("risa_sdpa_prequantized kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

// Nanobind wrapper: pure INT8 QK / U8-softmax / INT8-V attention kernel
void risa_attention(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
                    nb::ndarray<nb::device::cuda> v, nb::ndarray<nb::device::cuda> o,
                    nb::ndarray<nb::device::cuda> q_scale, nb::ndarray<nb::device::cuda> k_scale,
                    nb::ndarray<nb::device::cuda> v_scale, nb::ndarray<nb::device::cuda> v_center,
                    float sm_scale, int output_dtype_code, uintptr_t stream_ptr) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
    throw std::runtime_error("risa_attention: q, k, v, o must be 4D");
  }

  if (output_dtype_code != 1 && output_dtype_code != 2) {
    throw std::runtime_error("risa_attention: output_dtype_code must be 1 (fp16) or 2 (bf16)");
  }

  constexpr int CTA_K = 64;
  const int64_t padded_k_length = ((static_cast<int64_t>(k.shape(2)) + CTA_K - 1) / CTA_K) * CTA_K;
  if (v.shape(3) < padded_k_length || v.shape(3) % CTA_K != 0) {
    throw std::runtime_error(
        "risa_attention: packed V sequence extent must cover K and be a multiple of 64");
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_attention(q.data(), k.data(), v.data(), o.data(), q_scale.data(), k_scale.data(),
                   v_scale.data(), v_center.data(), nullptr, 0, 0, 0, 0, -1, CTA_K,
                   static_cast<int>(q.shape(0)), static_cast<int>(q.shape(2)),
                   static_cast<int>(k.shape(2)), static_cast<int>(q.shape(1)),
                   static_cast<int>(k.shape(1)), static_cast<int>(q.shape(3)), q.stride(0),
                   q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0),
                   v.stride(1), v.stride(2), o.stride(0), o.stride(2), o.stride(1), sm_scale,
                   output_dtype_code, nullptr, nullptr, nullptr, false, nullptr, stream);
}

// Fused RISA SDPA: Q/K quantization, V quantization, and attention in one call.
// All scratch buffers are pre-allocated by the caller (Python frontend).
void risa_sdpa(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
               nb::ndarray<nb::device::cuda> v, nb::ndarray<nb::device::cuda> o,
               nb::ndarray<nb::device::cuda> q_int8, nb::ndarray<nb::device::cuda> q_scale,
               nb::ndarray<nb::device::cuda> k_int8, nb::ndarray<nb::device::cuda> k_scale,
               nb::ndarray<nb::device::cuda> v_int8, nb::ndarray<nb::device::cuda> v_scale,
               nb::ndarray<nb::device::cuda> v_center, float sm_scale, int input_dtype_code,
               int output_dtype_code, bool use_midpoint_affine, uintptr_t stream_ptr,
               uintptr_t anchor_indices_ptr,
               std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt, int cta_k = 0,
               bool is_causal = false,
               std::optional<nb::ndarray<nb::device::cuda>> key_mask_tiles = std::nullopt) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
    throw std::runtime_error("risa_sdpa: q, k, v, o must be 4D [batch_size,num_heads,L,head_dim]");
  }

  const int batch_size = static_cast<int>(q.shape(0));
  const int query_heads = static_cast<int>(q.shape(1));
  const int query_length = static_cast<int>(q.shape(2));
  const int head_dim = static_cast<int>(q.shape(3));
  const int key_value_heads = static_cast<int>(k.shape(1));
  const int key_length = static_cast<int>(k.shape(2));

  const void* mask_ptr = nullptr;
  int64_t mask_stride_b = 0;
  int64_t mask_stride_h = 0;
  int64_t mask_stride_q = 0;
  int64_t mask_stride_k = 0;
  int mask_dtype_code = -1;
  if (attn_mask.has_value()) {
    const auto& mask = attn_mask.value();
    if (mask.ndim() != 4 || mask.shape(0) != batch_size || mask.shape(1) != query_heads ||
        mask.shape(2) != query_length || mask.shape(3) != key_length) {
      throw std::runtime_error("risa_sdpa: attention mask must be expanded to "
                               "[batch_size,query_heads,query_length,key_length]");
    }
    if (mask.dtype().code == (uint8_t)nb::dlpack::dtype_code::Bool) {
      mask_dtype_code = 3;
    } else {
      mask_dtype_code = map_dtype_to_code(mask.dtype());
    }
    if (mask_dtype_code < 0 || mask_dtype_code > 3) {
      throw std::runtime_error(
          "risa_sdpa: attention mask must be bool, float16, bfloat16, or float32");
    }
    mask_ptr = mask.data();
    mask_stride_b = mask.stride(0);
    mask_stride_h = mask.stride(1);
    mask_stride_q = mask.stride(2);
    mask_stride_k = mask.stride(3);
  }

  if (input_dtype_code < 0 || input_dtype_code > 2) {
    throw std::runtime_error("risa_sdpa: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
  }
  if (output_dtype_code != 1 && output_dtype_code != 2) {
    throw std::runtime_error("risa_sdpa: output_dtype_code must be 1 (fp16) or 2 (bf16)");
  }
  if (cta_k == 0) {
    cta_k = !attn_mask.has_value() && head_dim >= 128 && key_length > 1024 ? 128 : 64;
  }
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("risa_sdpa: cta_k must be 64 or 128");
  }
  if (cta_k == 128 && head_dim == 64) {
    throw std::runtime_error("risa_sdpa: cta_k 128 requires head_dim 128 or 256");
  }
  if (!anchor_indices_ptr) {
    throw std::runtime_error("risa_sdpa: anchor_indices scratch is required");
  }
  constexpr int BLKQ = 128;
  const int WARPQ = head_dim == 256 ? 16 : 32;
  const int BLKK = cta_k;
  const int WARPK = cta_k;
  const int padded_Lk = ((key_length + cta_k - 1) / cta_k) * cta_k;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  launch_quantize_qk(q.data(), q_int8.data(), q_scale.data(), k.data(), k_int8.data(),
                     k_scale.data(), batch_size, query_heads, query_length, key_value_heads,
                     key_length, head_dim, BLKQ, WARPQ, BLKK, WARPK, q.stride(0), q.stride(1),
                     q.stride(2), k.stride(0), k.stride(1), k.stride(2), input_dtype_code,
                     reinterpret_cast<void*>(anchor_indices_ptr), stream);

  launch_quantize_v(v.data(), v_int8.data(), v_scale.data(), v_center.data(), batch_size,
                    key_value_heads, key_length, head_dim, padded_Lk, v.stride(0), v.stride(1),
                    v.stride(2), input_dtype_code, use_midpoint_affine, stream);

  // Keep derived strides 64-bit through the launcher and device addressing.
  const int64_t qi_st_bz64 =
      static_cast<int64_t>(query_heads) * static_cast<int64_t>(query_length) * head_dim;
  const int64_t ki_st_bz64 = static_cast<int64_t>(key_value_heads) * key_length * head_dim;
  const int64_t v_st_bz64 = static_cast<int64_t>(key_value_heads) * head_dim * padded_Lk;

  const int64_t qi_st_h = static_cast<int64_t>(query_length) * head_dim, qi_st_n = head_dim,
                qi_st_bz = qi_st_bz64;
  const int64_t ki_st_h = static_cast<int64_t>(key_length) * head_dim, ki_st_n = head_dim,
                ki_st_bz = ki_st_bz64;
  const int64_t o_st_h = static_cast<int64_t>(query_length) * head_dim, o_st_n = head_dim,
                o_st_bz = qi_st_bz64;
  // v_int8 is [batch_size*key_value_heads*head_dim, padded_Lk] (2D from quant kernel).
  // Attention expects V as [batch_size, num_heads, head_dim, padded_length].
  const int64_t v_st_d = padded_Lk;
  const int64_t v_st_h = static_cast<int64_t>(head_dim) * padded_Lk;
  const int64_t v_st_bz = v_st_bz64;

  launch_attention(
      q_int8.data(), k_int8.data(), v_int8.data(), o.data(), q_scale.data(), k_scale.data(),
      v_scale.data(), v_center.data(), mask_ptr, mask_stride_b, mask_stride_h, mask_stride_q,
      mask_stride_k, mask_dtype_code, cta_k, batch_size, query_length, key_length, query_heads,
      key_value_heads, head_dim, qi_st_bz, qi_st_n, qi_st_h, ki_st_bz, ki_st_n, ki_st_h, v_st_bz,
      v_st_h, v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale, output_dtype_code, nullptr, nullptr,
      nullptr, is_causal, nullptr, stream,
      validate_key_mask_tiles(key_mask_tiles, batch_size, query_heads, key_length, cta_k));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("risa_sdpa kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

void risa_lse_shift(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
                    nb::ndarray<nb::device::cuda> anchors, nb::ndarray<nb::device::cuda> out,
                    float scale, uintptr_t stream_ptr) {
  const int dtype = map_dtype_to_code(q.dtype());
  if (q.ndim() != 4 || k.ndim() != 4 || anchors.ndim() != 2 || out.ndim() != 3 || dtype < 0 ||
      dtype > 2 || map_dtype_to_code(k.dtype()) != dtype || q.shape(0) != k.shape(0) ||
      q.shape(3) != k.shape(3) || !k.shape(1) || q.shape(1) % k.shape(1) || q.stride(3) != 1 ||
      k.stride(3) != 1 || anchors.shape(0) != k.shape(0) || anchors.shape(1) != k.shape(1) ||
      anchors.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
      anchors.dtype().bits != 32 || anchors.stride(1) != 1 ||
      (anchors.shape(0) > 1 && anchors.stride(0) != k.shape(1)) ||
      map_dtype_to_code(out.dtype()) != 0 || out.shape(0) != q.shape(0) ||
      out.shape(1) != q.shape(1) || out.shape(2) != q.shape(2) || out.stride(2) != 1 ||
      (out.shape(1) > 1 && out.stride(1) != q.shape(2)) ||
      (out.shape(0) > 1 && out.stride(0) != q.shape(1) * q.shape(2))) {
    throw std::runtime_error("LSE shift: incompatible shapes, dtypes or layout");
  }
  launch_correct_lse(q.data(), k.data(), static_cast<const int*>(anchors.data()),
                     static_cast<float*>(out.data()), q.shape(0), q.shape(1), k.shape(1),
                     q.shape(2), q.shape(3), q.stride(0), q.stride(1), q.stride(2), k.stride(0),
                     k.stride(1), k.stride(2), scale, dtype,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
}

NB_MODULE(_C, m) {
  m.def("risa_lse_shift", &risa_lse_shift, nb::arg("q"), nb::arg("k"), nb::arg("anchors"),
        nb::arg("out"), nb::arg("scale"), nb::arg("stream_ptr"));
  m.doc() = "RISA Attention CUDA kernels";
  m.def("risa_sdpa_quantize", &risa_sdpa_quantize, nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"), nb::arg("k_scale"),
        nb::arg("v_int8"), nb::arg("v_scale"), nb::arg("v_center"), nb::arg("cta_k"),
        nb::arg("input_dtype_code"), nb::arg("use_midpoint_affine"), nb::arg("stream_ptr"),
        nb::arg("anchor_indices_ptr"), nb::arg("permutation_ptr") = 0);
  m.attr("fused_gather_version") = 1;
  m.def("prepare_key_mask_tiles", &prepare_key_mask_tiles, nb::arg("mask"), nb::arg("tiles"),
        nb::arg("tile_size"), nb::arg("stream_ptr"));
  m.def("risa_sdpa_prequantized", &risa_sdpa_prequantized, nb::arg("q_int8"), nb::arg("k_int8"),
        nb::arg("v_int8"), nb::arg("o"), nb::arg("q_scale"), nb::arg("k_scale"), nb::arg("v_scale"),
        nb::arg("v_center"), nb::arg("cta_k"), nb::arg("sm_scale"), nb::arg("output_dtype_code"),
        nb::arg("stream_ptr"), nb::arg("attn_mask") = nb::none(),
        nb::arg("sparse_row_offsets") = nb::none(), nb::arg("sparse_block_indices") = nb::none(),
        nb::arg("block_mass") = nb::none(), nb::arg("is_causal") = false,
        nb::arg("lse") = nb::none(), nb::arg("key_mask_tiles") = nb::none());
  m.def("select_retained_mass_csr", &select_retained_mass_csr, nb::arg("sorted_mass"),
        nb::arg("descending_indices"), nb::arg("counts"), nb::arg("support"), nb::arg("row_totals"),
        nb::arg("row_offsets"), nb::arg("block_indices"), nb::arg("summary"),
        nb::arg("scan_workspace"), nb::arg("theta"), nb::arg("stream_ptr"));
  m.def("retained_mass_selector_workspace_size", &retained_mass_selector_workspace_size,
        nb::arg("row_count"));
  m.def("risa_sdpa", &risa_sdpa, nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("o"),
        nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"), nb::arg("k_scale"),
        nb::arg("v_int8"), nb::arg("v_scale"), nb::arg("v_center"), nb::arg("sm_scale"),
        nb::arg("input_dtype_code"), nb::arg("output_dtype_code"), nb::arg("use_midpoint_affine"),
        nb::arg("stream_ptr"), nb::arg("anchor_indices_ptr"), nb::arg("attn_mask") = nb::none(),
        nb::arg("cta_k") = 0, nb::arg("is_causal") = false, nb::arg("key_mask_tiles") = nb::none());
}
