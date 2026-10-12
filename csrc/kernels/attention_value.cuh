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
#include "attention_softmax.cuh"

#pragma nv_diag_suppress 174

#pragma nv_diag_default 174

template <uint32_t num_tiles_q, uint32_t num_tiles_v>
__device__ __forceinline__ void normalize_attention_output(float RO[][num_tiles_v][8], float m[][2],
                                                           float d[][2]) {

  // accumulate_d performs partial accumulation with cuda core
  // aggregate d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x1); // sum 0 and 1, 2 and 3
      d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x2); // sum 0 and 2, 1 and 3
    }
  }

  // divide O by d
  float d_rcp[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      d_rcp[fq][k] = math::ptx_rcp(d[fq][k]);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
      }
    }
  }
}

template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void
accumulate_int8_pv_pipelined(const smem_t<swizzle_mode, stride>& smem_V, int32_t RS_scale[][8][8],
                             uint32_t RS_u8[][4][4], float RO[][8][8]) {

  auto multiply = [&](uint32_t frag, int32_t partial[4]) {
    uint32_t RV[4][2];
#pragma unroll
    for (uint32_t fk = 0; fk < 4; ++fk) {
      const uint32_t offset = smem_V.get_permuted_offset(get_lane_id() % 8 + frag * 8,
                                                         (get_lane_id() / 8) % 2 + fk * 2);
      smem_V.ldmatrix_m8n8x2(offset, RV[fk]);
      if (fk == 0)
        mma::mma_sync_m16n8k32_row_col_u8s8s32<mma::MMAMode::kInit>(partial, RS_u8[0][fk], RV[fk]);
      else
        mma::mma_sync_m16n8k32_row_col_u8s8s32<mma::MMAMode::kInplaceUpdate>(partial, RS_u8[0][fk],
                                                                             RV[fk]);
    }
  };
  auto accumulate = [&](uint32_t frag, const int32_t partial[4]) {
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
      float& output = RO[0][frag / 2][(frag % 2) * 4 + k];
      output = fmaf(__int2float_rn(partial[k]), __int_as_float(RS_scale[0][0][k / 2]), output);
    }
  };

  int32_t partial[2][4];
  multiply(0, partial[0]);
#pragma unroll
  for (uint32_t frag = 1; frag < 16; ++frag) {
    multiply(frag, partial[frag % 2]);
    accumulate(frag - 1, partial[(frag - 1) % 2]);
  }
  accumulate(15, partial[1]);
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, bool pipeline_pv = true,
          typename DTypeSVAccum>
__device__ __forceinline__ void
accumulate_int8_pv(const smem_t<swizzle_mode, stride>& smem_V, int32_t RS_scale[][num_tiles_k][8],
                   uint32_t RS_u8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8]) {
#if __CUDA_ARCH__ == 890
  if constexpr (pipeline_pv && num_warps_q == 4 && num_warps_k == 1 && num_tiles_q == 1 &&
                num_tiles_k == 8 && num_tiles_v == 8) {
    accumulate_int8_pv_pipelined<swizzle_mode, stride>(smem_V, RS_scale, RS_u8, RO);
    return;
  }
#endif

  static_assert(std::is_same<DTypeSVAccum, float>::value);
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
  uint32_t offsets_V[num_tiles_k / 2];
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
    offsets_V[fk] = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
  }
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
    uint32_t RV[num_tiles_k / 2][4];
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      smem_V.ldmatrix_m8n8x4(offsets_V[fk], RV[fk]);
      offsets_V[fk] = smem_V.advance_offset_by_row<16>(offsets_V[fk]);
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      int32_t partial[8];
      mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInit>(partial, RS_u8[fq][0], RV[0]);
#pragma unroll
      for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
        mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInplaceUpdate>(
            partial, RS_u8[fq][fk], RV[fk]);
      }
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[fq][fv][k] = fmaf(__int2float_rn(partial[k]),
                             __int_as_float(RS_scale[fq][0][(k % 4) / 2]), RO[fq][fv][k]);
      }
    }
  }
}
