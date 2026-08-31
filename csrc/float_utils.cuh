/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Extracted from comfy-kitchen and reduced to the helpers used by Sage INT8
 * attention.
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace risa {

__device__ __forceinline__ float warp_reduce_fmax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float value) {
  int32_t quantized;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(quantized) : "f"(value));
  return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ int8_t quant_int8_rcp(float value, float inverse_scale) {
  return float_to_int8_rn(value * inverse_scale);
}

__device__ __forceinline__ void store4_i8(
    int8_t *pointer, int8_t a, int8_t b, int8_t c, int8_t d) {
  *reinterpret_cast<int32_t *>(pointer) =
      static_cast<uint32_t>(static_cast<uint8_t>(a)) |
      (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
      (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

template <typename T>
#pragma nv_diag_suppress 1056
__device__ __forceinline__ const T *load_f16x8(const T *value) {
  float4 loaded = *reinterpret_cast<const float4 *>(value);
  return reinterpret_cast<const T *>(&loaded);
}
#pragma nv_diag_default 1056

}  // namespace risa
