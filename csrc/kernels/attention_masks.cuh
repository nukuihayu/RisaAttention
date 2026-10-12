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

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_causal_mask(const uint32_t& Q_idx_lane_base, const uint32_t& K_idx_lane_base,
                  float RS[][num_tiles_k][8], const float mask_value = -50000.0f) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t q_idx = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx > q_idx);

        if (out_of_boundary) {
          RS[fq][fk][k] = mask_value;
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_custom_mask(const uint32_t& Q_idx_lane_base, const uint32_t& K_idx_lane_base,
                  float RS[][num_tiles_k][8], bool valid[][2], const void* mask,
                  const int64_t mask_stride_b, const int64_t mask_stride_h,
                  const int64_t mask_stride_q, const int64_t mask_stride_k, const uint32_t batch_id,
                  const uint32_t head_id, const uint32_t qo_len, const uint32_t kv_len,
                  const int mask_dtype_code, const float score_scale) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t q_idx = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const uint32_t q_group = (k % 4) / 2;
        bool keep = q_idx < qo_len && kv_idx < kv_len;
        float bias = 0.0f;

        if (keep) {
          const int64_t offset = static_cast<int64_t>(batch_id) * mask_stride_b +
                                 static_cast<int64_t>(head_id) * mask_stride_h +
                                 static_cast<int64_t>(q_idx) * mask_stride_q +
                                 static_cast<int64_t>(kv_idx) * mask_stride_k;
          if (mask_dtype_code == 3) {
            keep = static_cast<const uint8_t*>(mask)[offset] != 0;
          } else {
            if (mask_dtype_code == 0) {
              bias = static_cast<const float*>(mask)[offset];
            } else if (mask_dtype_code == 1) {
              bias = static_cast<float>(static_cast<const half*>(mask)[offset]);
            } else {
              bias = static_cast<float>(static_cast<const nv_bfloat16*>(mask)[offset]);
            }
            keep = isfinite(bias);
          }
        }

        valid[fq][q_group] = valid[fq][q_group] || keep;
        RS[fq][fk][k] = keep ? fmaf(RS[fq][fk][k], score_scale, bias * math::log2e) : -INFINITY;
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_custom_key_mask(const uint32_t& K_idx_lane_base, float RS[][num_tiles_k][8], bool valid[][2],
                      const void* mask, const int64_t mask_stride_b, const int64_t mask_stride_h,
                      const int64_t mask_stride_k, const uint32_t batch_id, const uint32_t head_id,
                      const uint32_t kv_len, const int mask_dtype_code, const float score_scale) {
  const int64_t head_offset = static_cast<int64_t>(batch_id) * mask_stride_b +
                              static_cast<int64_t>(head_id) * mask_stride_h;

#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
    bool keep[4];
    float bias[4];
#pragma unroll
    for (uint32_t value = 0; value < 4; value++) {
      const uint32_t kv_idx = K_idx_lane_base + fk * 16 + (value >> 1) * 8 + (value & 1);
      keep[value] = kv_idx < kv_len;
      bias[value] = 0.0f;
      if (keep[value]) {
        const int64_t offset = head_offset + static_cast<int64_t>(kv_idx) * mask_stride_k;
        if (mask_dtype_code == 3) {
          keep[value] = static_cast<const uint8_t*>(mask)[offset] != 0;
        } else {
          if (mask_dtype_code == 0) {
            bias[value] = static_cast<const float*>(mask)[offset];
          } else if (mask_dtype_code == 1) {
            bias[value] = static_cast<float>(static_cast<const half*>(mask)[offset]);
          } else {
            bias[value] = static_cast<float>(static_cast<const nv_bfloat16*>(mask)[offset]);
          }
          keep[value] = isfinite(bias[value]);
        }
      }
    }

    const bool tile_valid = keep[0] || keep[1] || keep[2] || keep[3];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      valid[fq][0] = valid[fq][0] || tile_valid;
      valid[fq][1] = valid[fq][1] || tile_valid;
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t value = (k >> 2) * 2 + (k & 1);
        RS[fq][fk][k] =
            keep[value] ? fmaf(RS[fq][fk][k], score_scale, bias[value] * math::log2e) : -INFINITY;
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_out_of_bound_mask(const uint32_t& K_idx_lane_base, float RS[][num_tiles_k][8],
                        const uint32_t& kv_len, const float mask_value = -50000.0f) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx >= kv_len);

        if (out_of_boundary) {
          RS[fq][fk][k] = mask_value;
        }
      }
    }
  }
}

// Summarize a broadcast mask tile cooperatively. Every warp sees the same
// mask and takes the same branch; no CTA synchronization or workspace is needed.
template <uint32_t tile_size>
__device__ __forceinline__ bool
key_mask_tile_is_empty(const void* mask, const int64_t row_offset, const int64_t key_stride,
                       const uint32_t tile, const uint32_t key_length, const int dtype_code) {
  bool visible = false;
#pragma unroll
  for (uint32_t offset = get_lane_id(); offset < tile_size; offset += 32) {
    const uint32_t key = tile * tile_size + offset;
    if (key < key_length) {
      const int64_t index = row_offset + static_cast<int64_t>(key) * key_stride;
      if (dtype_code == 3) {
        visible |= static_cast<const uint8_t*>(mask)[index] != 0;
      } else {
        const float bias = dtype_code == 0 ? static_cast<const float*>(mask)[index]
                           : dtype_code == 1
                               ? static_cast<float>(static_cast<const half*>(mask)[index])
                               : static_cast<float>(static_cast<const nv_bfloat16*>(mask)[index]);
        visible |= isfinite(bias);
      }
    }
  }
  return __ballot_sync(0xffffffffu, visible) == 0;
}
