// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2024 by SageAttention team.
// SPDX-FileContributor: Modified by NVIDIA CORPORATION & AFFILIATES, 2025.
// Derived from SageAttention (https://github.com/thu-ml/SageAttention)
// commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.
// Modifications: removed torch/extension.h dependency, flattened include paths.

/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "attention_types.cuh"

template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row,
          uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA,
          typename T>
__device__ __forceinline__ void copy_global_to_shared(T** lane_ptr, uint32_t& smem_offset,
                                                      const int64_t& gmem_stride,
                                                      const smem_t<swizzle_mode, stride>& smem) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes ==
                WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
                  (smem_iters_row * global_to_shared_line_lanes * pack_size));
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

// with predicate
template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row,
          uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA,
          typename T>
__device__ __forceinline__ void copy_global_to_shared(T** lane_ptr, uint32_t& smem_offset,
                                                      const int64_t& gmem_stride,
                                                      const smem_t<swizzle_mode, stride>& smem,
                                                      uint32_t base_idx, uint32_t max_len) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes ==
                WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async<cp_async::SharedMemFillMode::kNoFill>(smem_offset, *lane_ptr,
                                                                 base_idx < max_len);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
                  (smem_iters_row * global_to_shared_line_lanes * pack_size));
    base_idx += global_to_shared_copy_lines_per_warp_per_iter;
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row,
          uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA>
__device__ __forceinline__ void copy_values_to_shared(int8_t** lane_ptr, uint32_t& smem_offset,
                                                      const int64_t& gmem_stride,
                                                      const smem_t<swizzle_mode, stride>& smem) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes ==
                WARP_SIZE);
  constexpr uint32_t pack_size_int8 = 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size_int8);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
                  (smem_iters_row * global_to_shared_line_lanes * pack_size_int8));
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  // for QK: *lane_ptr += (CTA - smem_iters_col *
  // global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
  *lane_ptr += CTA; // ! prevent underflow
  *lane_ptr -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

#pragma nv_diag_suppress 174
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_qk_inner, SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void compute_qk_product(const smem_t<swizzle_mode, stride>& smem_Q,
                                                   const smem_t<swizzle_mode, stride>& smem_K,
                                                   int32_t RS[][num_tiles_k][8], uint32_t& offset_Q,
                                                   uint32_t& offset_K) {
  static_assert(DTypeQK == DataType::kInt8);

  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++) {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    // ! using permutation invariance
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[fq],
                                                                                RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_Q -= (2 * num_tiles_qk_inner);
  offset_K -= (2 * num_tiles_qk_inner);
}
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_qk_inner, SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void
compute_qk_with_cached_query(const smem_t<swizzle_mode, stride>& smem_K,
                             int32_t RS[][num_tiles_k][8], uint32_t RQ[][num_tiles_qk_inner][4],
                             uint32_t& offset_K) {
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq][iter],
                                                                       RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk],
                                                                                RQ[fq][iter], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_K -= (2 * num_tiles_qk_inner);
}
#pragma nv_diag_default 174

// for case when num_tiles_qk_inner = 1
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_qk_inner, SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void compute_qk_product(const smem_t<swizzle_mode, stride>& smem_K,
                                                   int32_t RS[][num_tiles_k][8], uint32_t RQ[][4],
                                                   uint32_t offset_K) {
  static_assert(DTypeQK == DataType::kInt8);
  static_assert(num_tiles_qk_inner == 1);

  uint32_t RK[4];

  // mma mode is kInit
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
    // load RK
    smem_K.ldmatrix_m8n8x4(offset_K, RK);
    offset_K = smem_K.advance_offset_by_row<16>(offset_K);

    // mma
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      if constexpr (DTypeQK == DataType::kInt8) {
        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
      }
    }
  }
}
