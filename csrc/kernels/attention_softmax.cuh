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

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, bool exp_offset,
          bool fuse_scale = false>
__device__ __forceinline__ void
update_online_softmax(float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
                      float d[][2], float pv_scale[][2], const float& sm_scale,
                      const float exp_offset_value) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      // assign the smallest value possible
      const float m_prev = m[fq][k];
      float m_temp = -50000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const float m_local = max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]),
                                  max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp = max(m_temp, m_local);
      }

      if constexpr (!fuse_scale) {
        if constexpr (exp_offset) {
          m_temp = fmaf(m_temp, sm_scale, -exp_offset_value);
        } else {
          m_temp *= sm_scale;
        }
      } else if constexpr (exp_offset) {
        m_temp -= exp_offset_value;
      }

      // exchange element with the 4 threads in the row
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp,
                                           0x1)); // 0 exchange with 1, 2 with 3
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp,
                                           0x2)); // 0 exchange with 2, 1 with 3

      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale = math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      pv_scale[fq][k] = tile_m < m_prev ? smaller_scale : 1.0f;

      // update denominator
      d[fq][k] *= o_scale;

      // update RO
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2 + 0] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      // raise RS to exponent
      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        if constexpr (fuse_scale) {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(RS[fq][fk][k * 2 + 0] + negative_m);
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(RS[fq][fk][k * 2 + 1] + negative_m);
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(RS[fq][fk][k * 2 + 4] + negative_m);
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(RS[fq][fk][k * 2 + 5] + negative_m);
        } else {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
        }
      }
    }
  }
}

struct PackedU8RowSum {
  uint32_t probabilities;
  float denominator;
};

__device__ __forceinline__ PackedU8RowSum pack_scaled_exp2_u8x4(const int32_t a, const int32_t b,
                                                                const int32_t c, const int32_t d,
                                                                const float scale,
                                                                const float negative_m) {
  const float probability_a = math::ptx_exp2(fmaf(math::i32_to_f32_exact(a), scale, negative_m));
  const float probability_b = math::ptx_exp2(fmaf(math::i32_to_f32_exact(b), scale, negative_m));
  const float probability_c = math::ptx_exp2(fmaf(math::i32_to_f32_exact(c), scale, negative_m));
  const float probability_d = math::ptx_exp2(fmaf(math::i32_to_f32_exact(d), scale, negative_m));
  const uint32_t probabilities =
      mma::pack_u8x4_bounded(probability_a, probability_b, probability_c, probability_d);
  const uint32_t denominator = __dp4a(probabilities, 0x01010101u, 0u);
  return {probabilities, __uint2float_rn(denominator)};
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          bool capture_tile_max = false>
__device__ __forceinline__ void
update_online_softmax_i32(int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
                          float d[][2], const float sm_scale, const float exp_offset_value,
                          uint32_t RS_u8[][num_tiles_k / 2][4],
                          float (*tile_max_out)[2] = nullptr) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      const float m_prev = m[fq][k];
      int32_t m_temp_i32 = INT_MIN;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const int32_t m_local = max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]),
                                    max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp_i32 = max(m_temp_i32, m_local);
      }

      float m_temp = fmaf(math::i32_to_f32_exact(m_temp_i32), sm_scale, -exp_offset_value);
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      const float tile_m = m_temp;
      if constexpr (capture_tile_max) {
        // Capture the tile maximum, not the running online-softmax maximum.
        // Block-mass capture used to repeat exactly this reduction and FMA.
        tile_max_out[fq][k] = tile_m;
      }
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale = math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
      d[fq][k] *= o_scale;
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        const PackedU8RowSum probabilities_0 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1], RS[fq][fk * 2][k * 2 + 4],
            RS[fq][fk * 2][k * 2 + 5], sm_scale, negative_m);
        const PackedU8RowSum probabilities_1 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2 + 1][k * 2], RS[fq][fk * 2 + 1][k * 2 + 1], RS[fq][fk * 2 + 1][k * 2 + 4],
            RS[fq][fk * 2 + 1][k * 2 + 5], sm_scale, negative_m);
        RS_u8[fq][fk][k] = probabilities_0.probabilities;
        RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
        d[fq][k] += probabilities_0.denominator * tile_scale;
        d[fq][k] += probabilities_1.denominator * tile_scale;
      }
      // QK scores are dead after packing. Reuse their registers for the
      // probability scale rather than extending the kernel's live state.
      RS[fq][0][k] = __float_as_int(tile_scale);
    }
  }
}

__device__ __forceinline__ PackedU8RowSum pack_exp2_u8x4(const float a, const float b,
                                                         const float c, const float d,
                                                         const float negative_m) {
  // Same [0, 255] bound as pack_scaled_exp2_u8x4: the tile max carries
  // S_U8_OFFSET, so the bounded pack and the exact int->float apply.
  const float pa = math::ptx_exp2(a + negative_m);
  const float pb = math::ptx_exp2(b + negative_m);
  const float pc = math::ptx_exp2(c + negative_m);
  const float pd = math::ptx_exp2(d + negative_m);
  const uint32_t packed = mma::pack_u8x4_bounded(pa, pb, pc, pd);
  return {packed, math::i32_to_f32_exact(static_cast<int32_t>(__dp4a(packed, 0x01010101u, 0u)))};
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v>
__device__ __forceinline__ void
update_online_softmax_f32(float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
                          float d[][2], const float exp_offset_value,
                          uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      const float m_prev = m[fq][k];
      float m_temp_f32 = -50000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const float m_local = max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]),
                                  max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp_f32 = max(m_temp_f32, m_local);
      }

      float m_temp = m_temp_f32 - exp_offset_value;
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale = math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
      d[fq][k] *= o_scale;
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        const PackedU8RowSum probabilities_0 =
            pack_exp2_u8x4(RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1],
                           RS[fq][fk * 2][k * 2 + 4], RS[fq][fk * 2][k * 2 + 5], negative_m);
        const PackedU8RowSum probabilities_1 = pack_exp2_u8x4(
            RS[fq][fk * 2 + 1][k * 2], RS[fq][fk * 2 + 1][k * 2 + 1], RS[fq][fk * 2 + 1][k * 2 + 4],
            RS[fq][fk * 2 + 1][k * 2 + 5], negative_m);
        RS_u8[fq][fk][k] = probabilities_0.probabilities;
        RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
        d[fq][k] += probabilities_0.denominator * tile_scale;
        d[fq][k] += probabilities_1.denominator * tile_scale;
      }
      // QK scores are dead after packing. Reuse their registers for the
      // probability scale rather than extending the kernel's live state.
      RS[fq][0][k] = tile_scale;
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void pack_probabilities_u8(T RS[][num_tiles_k][8],
                                                      uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      RS_u8[fq][fk][0] = mma::pack_u8x4_bounded(RS[fq][fk * 2][0], RS[fq][fk * 2][1],
                                                RS[fq][fk * 2][4], RS[fq][fk * 2][5]);
      RS_u8[fq][fk][1] = mma::pack_u8x4_bounded(RS[fq][fk * 2][2], RS[fq][fk * 2][3],
                                                RS[fq][fk * 2][6], RS[fq][fk * 2][7]);
      RS_u8[fq][fk][2] = mma::pack_u8x4_bounded(RS[fq][fk * 2 + 1][0], RS[fq][fk * 2 + 1][1],
                                                RS[fq][fk * 2 + 1][4], RS[fq][fk * 2 + 1][5]);
      RS_u8[fq][fk][3] = mma::pack_u8x4_bounded(RS[fq][fk * 2 + 1][2], RS[fq][fk * 2 + 1][3],
                                                RS[fq][fk * 2 + 1][6], RS[fq][fk * 2 + 1][7]);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void accumulate_probability_mass(uint32_t P[][num_tiles_k / 2][4],
                                                            float d[][2], float pv_scale[][2]) {
  // P is exactly the packed probability operand consumed by integer P*V.
  // Reuse it for the denominator; no runtime branch, second rounding, or
  // additional probability workspace is needed. FP32 online tile rescaling
  // remains identical to the numerator's tile rescaling.
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; ++fk) {
#pragma unroll
      for (uint32_t row = 0; row < 2; ++row) {
        const uint32_t sum =
            __dp4a(P[fq][fk][row + 2], 0x01010101u, __dp4a(P[fq][fk][row], 0x01010101u, 0u));
        d[fq][row] += math::i32_to_f32_exact(static_cast<int32_t>(sum)) * pv_scale[fq][row];
      }
    }
  }
}
