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
#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "math.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"

#define WARP_SIZE 32

// Shift the online-softmax maximum so exp2(score - max) spans the complete
// unsigned INT8 range. P uses U8 because softmax probabilities are nonnegative;
// Q, K, and V remain signed INT8.
#define S_U8_OFFSET 7.9943534f

#define div_ceil(M, N) (((M) + (N) - 1) / (N))

enum class MaskMode {
  kNone = 0,
  kCausal = 1,
  kCustom = 2,
  kCustomKey = 3,
};

enum class DataType {
  kInt8,
};

enum class QuantGranularity {
  kPerTensor = 0,
  kPerBlock = 1,
  kPerWarp = 2,
  kPerThread = 3,
};

__device__ __forceinline__ uint32_t get_warp_id() { return threadIdx.y; }

__device__ __forceinline__ uint32_t get_lane_id() { return threadIdx.x; }

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_q() {
  return get_warp_id() / num_warps_k;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_k() {
  return get_warp_id() % num_warps_k;
}
