// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2024 by SageAttention team.
// SPDX-FileContributor: Modified by NVIDIA CORPORATION & AFFILIATES, 2025.
// Derived from SageAttention (https://github.com/thu-ml/SageAttention)
// commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.
// Modifications: removed torch/extension.h dependency, flattened include paths.

#pragma once

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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "math.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"

#include "attn_utils.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // int8
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// unsigned INT8 softmax x signed INT8 V tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN,
          QuantGranularity K_GRAN, typename DTypeSVAccum = float,
          bool use_inst_buffer = false, typename DTypeOut = half,
          ComputeUnit DenominatorAccumUnit,
          MaskMode mask_mode = MaskMode::kNone, bool return_lse = false,
          bool fuse_v_scale = false, bool use_pv_fp16_accu = false,
          bool use_sparse_support = false,
          bool return_block_mass = false>
__global__ void qk_int_sv_i8_attn_kernel(
    int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V,
    DTypeOut *__restrict__ O, float *__restrict__ Lse,
    float *__restrict__ Q_scale, float *__restrict__ K_scale,
    float *__restrict__ V_scale, float *__restrict__ V_center,
    const void *__restrict__ AttnMask,
    const int64_t mask_stride_b,
    const int64_t mask_stride_h, const int64_t mask_stride_q,
    const int64_t mask_stride_k, const int mask_dtype_code,
    const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q,
    const uint32_t stride_h_q, const uint32_t stride_bz_k,
    const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_h_v,
    const uint32_t stride_d_v, const uint32_t stride_bz_o,
    const uint32_t stride_seq_o, const uint32_t stride_h_o, float sm_scale,
    const int32_t *__restrict__ sparse_row_offsets,
    const int32_t *__restrict__ sparse_block_indices,
    float *__restrict__ block_mass) {
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4,
                "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock ||
                    Q_GRAN == QuantGranularity::kPerWarp ||
                    Q_GRAN == QuantGranularity::kPerThread,
                "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock ||
                    K_GRAN == QuantGranularity::kPerWarp ||
                    K_GRAN == QuantGranularity::kPerThread,
                "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(std::is_same<DTypeSVAccum, float>::value,
                "DTypeSVAccum must be float, half is WIP");
  static_assert(DenominatorAccumUnit == ComputeUnit::kCudaCore,
                "pure INT8 attention accumulates the softmax denominator on CUDA cores");
  static_assert(std::is_same<DTypeOut, half>::value ||
                    std::is_same<DTypeOut, nv_bfloat16>::value,
                "DTypeOut must be half or nv_bfloat16");
  static_assert(CTA_K % 64 == 0);
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation
  static_assert(!use_sparse_support || mask_mode == MaskMode::kNone,
                "sparse support cannot be combined with an attention mask");
  static_assert(!return_block_mass ||
                    (!use_sparse_support && mask_mode == MaskMode::kNone),
                "block mass is only supported by dense unmasked attention");

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8)
                                              ? (head_dim / MMA_QK_K)
                                              : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;
  constexpr bool custom_mask = mask_mode == MaskMode::kCustom ||
                               mask_mode == MaskMode::kCustomKey;
  // For unmasked and causal FP32 kernels, retain raw scores until update_mdo
  // fuses score scaling, max subtraction, and conversion to the exp2 domain.
  // Custom masks keep pre-scaled scores so additive bias values retain their
  // existing semantics.
  constexpr bool pre_scale_scores = custom_mask;
  constexpr uint32_t QK_SMEM_STRIDE =
      (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator
  bool valid[num_tiles_q][2];

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx =
        batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  } else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q +
                  head_id * num_warp_block_q + bx * num_warps_q +
                  get_warp_idx_q<num_warps_q, num_warps_k>();
  } else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
    if constexpr (head_dim == 128 && WARP_Q == 16) {
      constexpr uint32_t quant_warps_q = CTA_Q / 32;
      const uint32_t num_warp_block_q = gridDim.x * quant_warps_q;
      q_scale_idx =
          batch_id * num_qo_heads * (num_warp_block_q * 8) +
          head_id * (num_warp_block_q * 8) + bx * (quant_warps_q * 8) +
          (get_warp_idx_q<num_warps_q, num_warps_k>() / 2) * 8 + lane_id / 4;
    } else {
      const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
      q_scale_idx =
          batch_id * num_qo_heads * (num_warp_block_q * 8) +
          head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) +
          get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
    }
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock) {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k +
                  (head_id / num_kv_groups) * num_block_k;
  } else if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
    const uint32_t num_warp_block_k =
        div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k +
                  (head_id / num_kv_groups) * num_warp_block_k +
                  get_warp_idx_k<num_warps_q, num_warps_k>();
  } else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
    const uint32_t num_warp_block_k =
        div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx =
        batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) +
        (head_id / num_kv_groups) * (num_warp_block_k * 4) +
        get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset =
      (K_GRAN == QuantGranularity::kPerBlock)  ? 1
      : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K)
                                               : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      if constexpr (std::is_same<DTypeSVAccum, float>::value) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          RO[fq][fv][k] = 0.0f;
        }
      } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
          ((int32_t *)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      m[fq][k] = -50000.0f;
      d[fq][k] = 1.0f;
      valid[fq][k] = false;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK =
      (QK_SMEM_STRIDE == 32)   ? SwizzleMode::k32B
      : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B
                               : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(
      smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V =
      (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(
      smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O =
      (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2
                                                      : (QK_SMEM_STRIDE == 64)
                                                          ? 4
                                                          : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK =
      (QK_SMEM_STRIDE == 32)   ? 16
      : (QK_SMEM_STRIDE == 64) ? 8
                               : 4;
  constexpr uint32_t global_to_shared_line_lanes_V =
      (V_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V =
      (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O =
      (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O =
      (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row =
      QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col =
      CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row =
      V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col =
      head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row =
      O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr =
      Q + batch_id * stride_bz_q + head_id * stride_h_q +
      (bx * CTA_Q + CTA_Q / num_warps * warp_id +
       lane_id / global_to_shared_line_lanes_QK) *
          stride_seq_q +
      (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr =
      K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k +
      (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) *
          stride_seq_k +
      (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *V_lane_base_ptr =
      V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v +
      head_dim / num_warps * warp_id * stride_d_v +
      lane_id / global_to_shared_line_lanes_V * stride_d_v +
      (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  int8_t *K_sparse_lane_base_ptr = K_lane_base_ptr;
  int8_t *V_sparse_lane_base_ptr = V_lane_base_ptr;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col +
          lane_id / global_to_shared_line_lanes_QK,
      lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col +
          lane_id / global_to_shared_line_lanes_QK,
      lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col +
          lane_id / global_to_shared_line_lanes_V,
      lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16,
      lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(
      get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 +
          (lane_id / 16) * 8,
      (lane_id / 8) % 2);
  // for causal masking
  uint32_t Q_idx_lane_base =
      bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
      lane_id / 4;
  const uint32_t K_idx_lane_offset =
      get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);
  uint32_t K_idx_lane_base = K_idx_lane_offset;

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_QK;
  const uint32_t K_load_idx_lane_offset =
      CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = K_load_idx_lane_offset;

  uint32_t sparse_row_start = 0;
  uint32_t num_iterations;
  if constexpr (use_sparse_support) {
    const uint32_t sparse_row =
        (batch_id * num_qo_heads + head_id) * gridDim.x + bx;
    sparse_row_start = sparse_row_offsets[sparse_row];
    num_iterations = sparse_row_offsets[sparse_row + 1] - sparse_row_start;
    const uint32_t first_kv_tile = sparse_block_indices[sparse_row_start];
    K_lane_base_ptr =
        K_sparse_lane_base_ptr + first_kv_tile * CTA_K * stride_seq_k;
    V_lane_base_ptr = V_sparse_lane_base_ptr + first_kv_tile * CTA_K;
    K_idx_lane_base = first_kv_tile * CTA_K + K_idx_lane_offset;
    K_load_idx_lane_base = first_kv_tile * CTA_K + K_load_idx_lane_offset;
  } else {
    num_iterations = div_ceil(
        mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len,
        CTA_K);
  }

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK,
                       global_to_shared_copy_lines_per_warp_QK,
                       QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
      &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q,
      Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK,
                       global_to_shared_copy_lines_per_warp_QK,
                       QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K,
      K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  const uint32_t first_scale_tile = use_sparse_support
                                        ? sparse_block_indices[sparse_row_start]
                                        : 0;
  float dequant_scale = q_scale *
                        K_scale[k_scale_idx +
                                first_scale_tile * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V
  // V is padded to a complete CTA_K tile by the quantizer.
  load_int8_V_global_to_share<global_to_shared_line_lanes_V,
                             global_to_shared_copy_lines_per_warp_V,
                             V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
                             V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
  cp_async::commit_group();

  if constexpr (!use_sparse_support) {
    K_load_idx_lane_base += CTA_K;
  }

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++) {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1) {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                             K_smem_offset_mma);
    } else {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }
    uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
    if constexpr (mask_mode == MaskMode::kNone) {
      update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
          RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8);
    } else {
      float pv_scale[num_tiles_q][2];
      float RS_soft[num_tiles_q][num_tiles_k][8];
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) {
            const float score = __int2float_rz(RS[fq][fk][k]);
            RS_soft[fq][fk][k] =
                pre_scale_scores ? score * sm_scale : score;
          }
        }
      }

      if constexpr (mask_mode == MaskMode::kCustom) {
        apply_custom_mask<num_tiles_q, num_tiles_k>(
            Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
            mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
            batch_id, head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
      } else if constexpr (mask_mode == MaskMode::kCustomKey) {
        apply_custom_key_mask<num_tiles_q, num_tiles_k>(
            K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
            mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
            mask_dtype_code, 1.0f);
      }

      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
                 pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                   S_U8_OFFSET);
      RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

      if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
        accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
      }
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++)
          RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
      }
    }
    if constexpr (!use_sparse_support) {
      K_idx_lane_base += CTA_K;
    }

    __syncthreads();

    uint32_t scale_tile = iter;
    if constexpr (use_sparse_support) {
      scale_tile = sparse_block_indices[sparse_row_start + iter];
      K_lane_base_ptr =
          K_sparse_lane_base_ptr + scale_tile * CTA_K * stride_seq_k;
      V_lane_base_ptr = V_sparse_lane_base_ptr + scale_tile * CTA_K;
      K_idx_lane_base = scale_tile * CTA_K + K_idx_lane_offset;
      K_load_idx_lane_base = scale_tile * CTA_K + K_load_idx_lane_offset;
    }

    // load K without predicate; sparse indices always name padded CTA_K tiles.
    load_global_to_share<global_to_shared_line_lanes_QK,
                         global_to_shared_copy_lines_per_warp_QK,
                         QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale *
                    K_scale[k_scale_idx + scale_tile * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                    num_tiles_v, swizzle_mode_V,
                    V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS, RS_u8, RO);
    __syncthreads();
    // load V
    load_int8_V_global_to_share<
        global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V,
        V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
        V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();

    if constexpr (!use_sparse_support) {
      K_load_idx_lane_base += CTA_K;
    }
  }

  // second last iter, apply causal mask
  if (num_iterations > 1) {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1) {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                             K_smem_offset_mma);
    } else {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
    if constexpr (mask_mode == MaskMode::kNone) {
      update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
          RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8);
    } else {
      float pv_scale[num_tiles_q][2];
      float RS_soft[num_tiles_q][num_tiles_k][8];
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) {
            const float score = __int2float_rz(RS[fq][fk][k]);
            RS_soft[fq][fk][k] =
                pre_scale_scores ? score * sm_scale : score;
          }
        }
      }

      if constexpr (mask_mode == MaskMode::kCausal) {
        apply_causal_mask<num_tiles_q, num_tiles_k>(
            Q_idx_lane_base, K_idx_lane_base, RS_soft,
            pre_scale_scores ? -50000.0f : -1.0e30f);
      } else if constexpr (mask_mode == MaskMode::kCustom) {
        apply_custom_mask<num_tiles_q, num_tiles_k>(
            Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
            mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
            batch_id, head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
      } else if constexpr (mask_mode == MaskMode::kCustomKey) {
        apply_custom_key_mask<num_tiles_q, num_tiles_k>(
            K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
            mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
            mask_dtype_code, 1.0f);
      }

      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
                 pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                   S_U8_OFFSET);
      RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

      if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
        accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
      }
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++)
          RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
      }
    }
    if constexpr (!use_sparse_support) {
      K_idx_lane_base += CTA_K;
    }

    __syncthreads();

    uint32_t last_scale_tile = num_iterations - 1;
    if constexpr (use_sparse_support) {
      last_scale_tile = sparse_block_indices[sparse_row_start + num_iterations - 1];
      K_lane_base_ptr =
          K_sparse_lane_base_ptr + last_scale_tile * CTA_K * stride_seq_k;
      V_lane_base_ptr = V_sparse_lane_base_ptr + last_scale_tile * CTA_K;
      K_idx_lane_base = last_scale_tile * CTA_K + K_idx_lane_offset;
      K_load_idx_lane_base = last_scale_tile * CTA_K + K_load_idx_lane_offset;
    }

    // The selected indices are sorted, so a partial tail tile is last when kept.
    load_global_to_share<global_to_shared_line_lanes_QK,
                         global_to_shared_copy_lines_per_warp_QK,
                         QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K,
        K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale =
        q_scale * K_scale[k_scale_idx + last_scale_tile * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                    num_tiles_v, swizzle_mode_V,
                    V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS, RS_u8, RO);

    __syncthreads();
    // load V
    load_int8_V_global_to_share<
        global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V,
        V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
        V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();
    if constexpr (!use_sparse_support) {
      K_load_idx_lane_base += CTA_K;
    }
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1) {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                             K_smem_offset_mma);
    } else {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_soft[num_tiles_q][num_tiles_k][8];
    float pv_scale[num_tiles_q][2];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const float score = __int2float_rz(RS[fq][fk][k]);
          RS_soft[fq][fk][k] =
              pre_scale_scores ? score * sm_scale : score;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal) {
      apply_causal_mask<num_tiles_q, num_tiles_k>(
          Q_idx_lane_base, K_idx_lane_base, RS_soft,
          pre_scale_scores ? -50000.0f : -1.0e30f);
    } else if constexpr (mask_mode == MaskMode::kCustom) {
      apply_custom_mask<num_tiles_q, num_tiles_k>(
          Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
          mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k, batch_id,
          head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
    } else if constexpr (mask_mode == MaskMode::kCustomKey) {
      apply_custom_key_mask<num_tiles_q, num_tiles_k>(
          K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
          mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
          mask_dtype_code, 1.0f);
    }
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
        K_idx_lane_base, RS_soft, kv_len,
        pre_scale_scores ? -50000.0f : -1.0e30f);

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
               pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                 S_U8_OFFSET);

    uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
    RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
      accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++)
        RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
    }
    K_idx_lane_base += CTA_K;

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                    num_tiles_v, swizzle_mode_V,
                    V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS, RS_u8, RO);

    __syncthreads();
  }

  // TODO: thread block sync mdo state for num_warps_k > 0. Then only one thread
  // block needs to do the final saving.

  if constexpr (return_block_mass) {
    // Recompute QK after online softmax has produced the final row max and
    // denominator. This emits one normalized mass per CTA_Q x CTA_K tile
    // without storing token-level probabilities in global memory.
    const uint32_t num_mass_blocks = div_ceil(kv_len, CTA_K);
    const uint32_t mass_row =
        (batch_id * num_qo_heads + head_id) * gridDim.x + bx;
    float cta_mass_total = 0.0f;
    float *mass_smem = reinterpret_cast<float *>(smem);
    for (uint32_t mass_block = 0; mass_block < num_mass_blocks; mass_block++) {
      K_lane_base_ptr =
          K_sparse_lane_base_ptr + mass_block * CTA_K * stride_seq_k;
      K_load_idx_lane_base =
          mass_block * CTA_K + K_load_idx_lane_offset;
      load_global_to_share<global_to_shared_line_lanes_QK,
                           global_to_shared_copy_lines_per_warp_QK,
                           QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                           QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
          &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K,
          K_load_idx_lane_base, kv_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

      if constexpr (num_tiles_qk_inner == 1) {
        compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                       num_tiles_qk_inner, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
            smem_K, RS, RQ, K_smem_offset_mma);
      } else {
        compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                       num_tiles_qk_inner, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
            smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
      }

      const float mass_sm_scale =
          original_sm_scale * q_scale *
          K_scale[k_scale_idx + mass_block * k_scale_advance_offset];
      float warp_mass = 0.0f;
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t row_half = 0; row_half < 2; row_half++) {
          int32_t row_scores[num_tiles_k][4];
          int32_t local_max = INT_MIN;
#pragma unroll
          for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            constexpr int32_t invalid_score = -1000000000;
            const uint32_t elements[4] = {row_half * 2,
                                          row_half * 2 + 1,
                                          row_half * 2 + 4,
                                          row_half * 2 + 5};
#pragma unroll
            for (uint32_t value = 0; value < 4; value++) {
              const uint32_t element = elements[value];
              const uint32_t kv_idx =
                  mass_block * CTA_K + K_idx_lane_offset + fk * 16 +
                  8 * (element / 4) + element % 2;
              const int32_t score =
                  kv_idx < kv_len ? RS[fq][fk][element] : invalid_score;
              row_scores[fk][value] = score;
              local_max = max(local_max, score);
            }
          }
          float tile_m =
              fmaf(__int2float_rz(local_max), mass_sm_scale, -S_U8_OFFSET);
          tile_m = max(tile_m,
                       __shfl_xor_sync(0xffffffff, tile_m, 0x1));
          tile_m = max(tile_m,
                       __shfl_xor_sync(0xffffffff, tile_m, 0x2));

          float numerator = 0.0f;
#pragma unroll
          for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            if (mass_block + 1 == num_mass_blocks) {
              numerator += math::ptx_exp2(fmaf(
                  __int2float_rz(row_scores[fk][0]), mass_sm_scale, -tile_m));
              numerator += math::ptx_exp2(fmaf(
                  __int2float_rz(row_scores[fk][1]), mass_sm_scale, -tile_m));
              numerator += math::ptx_exp2(fmaf(
                  __int2float_rz(row_scores[fk][2]), mass_sm_scale, -tile_m));
              numerator += math::ptx_exp2(fmaf(
                  __int2float_rz(row_scores[fk][3]), mass_sm_scale, -tile_m));
            } else {
              numerator += pack_scaled_exp2_u8x4(
                               row_scores[fk][0], row_scores[fk][1],
                               row_scores[fk][2], row_scores[fk][3],
                               mass_sm_scale, -tile_m)
                               .denominator;
            }
          }
          numerator *= math::ptx_exp2(tile_m - m[fq][row_half]);
          numerator +=
              __shfl_xor_sync(0xffffffff, numerator, 0x1);
          numerator +=
              __shfl_xor_sync(0xffffffff, numerator, 0x2);

          float denominator = d[fq][row_half];
          denominator +=
              __shfl_xor_sync(0xffffffff, denominator, 0x1);
          denominator +=
              __shfl_xor_sync(0xffffffff, denominator, 0x2);
          const uint32_t query_idx =
              bx * CTA_Q +
              get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
              fq * 16 + lane_id / 4 + row_half * 8;
          if ((lane_id & 3) == 0 && query_idx < qo_len) {
            warp_mass += numerator / denominator;
          }
        }
      }
#pragma unroll
      for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        warp_mass +=
            __shfl_down_sync(0xffffffff, warp_mass, offset);
      }
      if (lane_id == 0) {
        mass_smem[warp_id] = warp_mass;
      }
      __syncthreads();
      if (warp_id == 0 && lane_id == 0) {
        float cta_mass = 0.0f;
#pragma unroll
        for (uint32_t warp = 0; warp < num_warps; warp++) {
          cta_mass += mass_smem[warp];
        }
        block_mass[mass_row * num_mass_blocks + mass_block] = cta_mass;
        cta_mass_total += cta_mass;
      }
      __syncthreads();
    }
    if (warp_id == 0 && lane_id == 0) {
      const float valid_rows = min(CTA_Q, qo_len - bx * CTA_Q);
      const float normalization = valid_rows / cta_mass_total;
      for (uint32_t mass_block = 0; mass_block < num_mass_blocks;
           mass_block++) {
        block_mass[mass_row * num_mass_blocks + mass_block] *= normalization;
      }
    }
    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (custom_mask) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++) {
        int row_valid = valid[fq][k] ? 1 : 0;
        row_valid |= __shfl_xor_sync(0xffffffff, row_valid, 0x1);
        row_valid |= __shfl_xor_sync(0xffffffff, row_valid, 0x2);
        valid[fq][k] = row_valid != 0;
      }
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          if (!valid[fq][(k % 4) / 2])
            RO[fq][fv][k] = 0.0f;
        }
      }
    }
  }

  // ! here we just implement the case for fp32 acumulation
  if constexpr (fuse_v_scale) {
    float v_scale[4];
    float v_center[4];
    float *V_scale_base_ptr =
        V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
    float *V_center_base_ptr =
        V_center + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      ((float2 *)v_scale)[0] = *((float2 *)(V_scale_base_ptr + fv * 16));
      ((float2 *)v_scale)[1] = *((float2 *)(V_scale_base_ptr + fv * 16 + 8));
      ((float2 *)v_center)[0] = *((float2 *)(V_center_base_ptr + fv * 16));
      ((float2 *)v_center)[1] =
          *((float2 *)(V_center_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const float scale_value = v_scale[(k / 4) * 2 + (k % 2)];
          const float center_value = v_center[(k / 4) * 2 + (k % 2)];
          if constexpr (custom_mask) {
            RO[fq][fv][k] = valid[fq][(k % 4) / 2]
                                 ? fmaf(RO[fq][fv][k], scale_value, center_value)
                                 : 0.0f;
          } else {
            RO[fq][fv][k] =
                fmaf(RO[fq][fv][k], scale_value, center_value);
          }
        }
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      uint32_t offset_O = smem_O.get_permuted_offset(
          smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value) {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
          if constexpr (std::is_same<DTypeOut, half>::value) {
            ((half2 *)RO_f16)[k] = __float22half2_rn(((float2 *)RO[fq][fv])[k]);
          } else {
            ((nv_bfloat162 *)RO_f16)[k] =
                __float22bfloat162_rn(((float2 *)RO[fq][fv])[k]);
          }
        }

        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[1];

        offset_O = smem_O.get_permuted_offset(
            smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[3];
      } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
        // TODO: not implement
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr =
      O + batch_id * stride_bz_o + head_id * stride_h_o +
      (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() +
       lane_id / global_to_shared_line_lanes_O) *
          stride_seq_o +
      lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
          lane_id / global_to_shared_line_lanes_O,
      lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++) {
      if (O_load_idx_lane_base < qo_len) {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(
          offset_O);
    }

    offset_O =
        smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(
            offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) -
         (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse) {
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) +
                       WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr =
        Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len) {
      lse_lane_ptr[0] = math::ptx_log2(d[fq][k]) + m[fq][k];
    }
  }
}
