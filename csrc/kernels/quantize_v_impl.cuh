#pragma once
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Fused V -> signed INT8 quantization with per-channel scaling.
// Output layout [batch_size,num_heads,head_dim,padded_length] with INT8 MMA 16-element permutation
// and per-row scale.
//
// One thread block per (b, h, d_tile) with D_TILE=8 d-channels. Short K uses
// 128 threads so it does not launch idle warps; longer K normally uses 512.
// SM120 BF16 token-major D128 uses 256 threads for measured short sequences.
// Threads cooperate along sequence_length with vectorized 128-bit loads, warp-shuffle range
// reduction, and reverse Pass 2 iteration for L2 cache reuse. Pass 1 uses 4x
// manual unroll for memory-level parallelism at low occupancy.
//
// Pass 2 uses linear source indices for byte stores. Long absmax sequences
// assemble four destination bytes per word to reduce store instructions; the
// forward permutation maps each byte back to its source token.

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include "quantization_input.cuh"

#include "device_properties.cuh"
#include <cfloat>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int kChannelTile = 8;

// Host-side selection uses the active device, not the cubins in the binary.
// Cache only successful queries and recheck after device switches.
bool use_sm120_v_schedule() {
#ifdef RISA_V_QUANT_SM120
  return true;
#else
  const auto& properties = attention_device_properties();
  return properties.major == 12 && properties.minor == 0;
#endif
}

// INT8 MMA 16-element inverse permutation.
// Forward: fwd(j) maps bit pattern {b3,b2,b1,b0} -> {b1,b3,b2,b0}.
// Inverse: inv(s) maps {b3,b2,b1,b0} -> {b2,b0,b3,b1}.
__device__ __forceinline__ int inv_perm16(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) | (((w >> 2) & 1) << 3);
}

template <typename T> __device__ __forceinline__ T warp_reduce_sum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename T> __device__ __forceinline__ void load_tile(const T* ptr, float* out_vals) {
  if constexpr (std::is_same_v<T, float>) {
    float4 v0 = *reinterpret_cast<const float4*>(ptr);
    float4 v1 = *reinterpret_cast<const float4*>(ptr + 4);
    out_vals[0] = v0.x;
    out_vals[1] = v0.y;
    out_vals[2] = v0.z;
    out_vals[3] = v0.w;
    out_vals[4] = v1.x;
    out_vals[5] = v1.y;
    out_vals[6] = v1.z;
    out_vals[7] = v1.w;
  } else {
    const T* loaded = risa::load_f16x8(ptr);
#pragma unroll
    for (int i = 0; i < kChannelTile; ++i) {
      out_vals[i] = static_cast<float>(loaded[i]);
    }
  }
}

template <typename T, int THREADS, bool USE_MIDPOINT_AFFINE, bool PACK_OUTPUT = false>
__global__ void quantize_values_kernel(const T* __restrict__ v, int8_t* __restrict__ out,
                                       float* __restrict__ scale_out,
                                       float* __restrict__ center_out, int sequence_length,
                                       int padded_length, int num_heads, int head_dim,
                                       int64_t batch_stride, int64_t head_stride,
                                       int64_t token_stride, risa::QuantInput transform = {}) {
  static_assert(!PACK_OUTPUT || !USE_MIDPOINT_AFFINE);
  const int d_tiles = head_dim / kChannelTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % num_heads;
  const int b = bh / num_heads;
  transform = transform.head<risa::kGatherQuantization>(b, h, num_heads, sequence_length, head_dim);
  const int d0 = d_tile * kChannelTile;

  const T* base = v + b * batch_stride + h * head_stride + d0;
  constexpr int WARPS = THREADS / 32;

  // Pass 1: per-channel range. Midpoint-affine minimizes the worst-case
  // symmetric INT8 step; absmax retains the legacy zero-centered mapping.
  float mn[kChannelTile];
  float mx[kChannelTile];
#pragma unroll
  for (int i = 0; i < kChannelTile; ++i) {
    mn[i] = FLT_MAX;
    mx[i] = -FLT_MAX;
  }

  // 4x unrolled: issue 4 independent 128-bit loads per iteration
  // so the memory controller can overlap them (critical at <=50% occupancy).
  int n = threadIdx.x;
  const int N_body = sequence_length - 3 * THREADS;
  for (; n < N_body; n += 4 * THREADS) {
    float t0[kChannelTile], t1[kChannelTile], t2[kChannelTile], t3[kChannelTile];
    load_tile(base + (int64_t)n * token_stride, t0);
    load_tile(base + (int64_t)(n + THREADS) * token_stride, t1);
    load_tile(base + (int64_t)(n + 2 * THREADS) * token_stride, t2);
    load_tile(base + (int64_t)(n + 3 * THREADS) * token_stride, t3);
#pragma unroll
    for (int di = 0; di < kChannelTile; ++di) {
      if constexpr (USE_MIDPOINT_AFFINE) {
        mn[di] = fminf(mn[di], fminf(fminf(t0[di], t1[di]), fminf(t2[di], t3[di])));
        mx[di] = fmaxf(mx[di], fmaxf(fmaxf(t0[di], t1[di]), fmaxf(t2[di], t3[di])));
      } else {
        mx[di] = fmaxf(mx[di], fmaxf(fmaxf(fabsf(t0[di]), fabsf(t1[di])),
                                     fmaxf(fabsf(t2[di]), fabsf(t3[di]))));
      }
    }
  }
  for (; n < sequence_length; n += THREADS) {
    float tmp[kChannelTile];
    load_tile(base + (int64_t)n * token_stride, tmp);
#pragma unroll
    for (int di = 0; di < kChannelTile; ++di) {
      if constexpr (USE_MIDPOINT_AFFINE) {
        mn[di] = fminf(mn[di], tmp[di]);
        mx[di] = fmaxf(mx[di], tmp[di]);
      } else {
        mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
      }
    }
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int di = 0; di < kChannelTile; ++di) {
    if constexpr (USE_MIDPOINT_AFFINE)
      mn[di] = -risa::warp_reduce_fmax(-mn[di]);
    mx[di] = risa::warp_reduce_fmax(mx[di]);
  }

  // Cross-warp reduction
  __shared__ float warp_mn[kChannelTile][WARPS];
  __shared__ float warp_mx[kChannelTile][WARPS];
  __shared__ float inv_sc_sh[kChannelTile];
  __shared__ float center_sh[kChannelTile];
  __shared__ float warp_residual_sum[kChannelTile][WARPS];

  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kChannelTile; ++di) {
      if constexpr (USE_MIDPOINT_AFFINE)
        warp_mn[di][warp] = mn[di];
      warp_mx[di][warp] = mx[di];
    }
  }
  __syncthreads();

  if (threadIdx.x < kChannelTile) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      if constexpr (USE_MIDPOINT_AFFINE)
        min_val = fminf(min_val, warp_mn[threadIdx.x][w]);
      max_val = fmaxf(max_val, warp_mx[threadIdx.x][w]);
    }

    const float center = USE_MIDPOINT_AFFINE ? 0.5f * (min_val + max_val) : 0.0f;
    const float radius = USE_MIDPOINT_AFFINE ? 0.5f * (max_val - min_val) : max_val;
    const float sc = fmaxf(radius * (1.f / 127.f), 1e-12f);
    const int parameter_index = (b * num_heads + h) * head_dim + d0 + threadIdx.x;
    scale_out[parameter_index] = sc;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
    center_sh[threadIdx.x] = center;
  }
  __syncthreads();

  float inv_sc[kChannelTile];
  float center[kChannelTile];
#pragma unroll
  for (int di = 0; di < kChannelTile; ++di) {
    inv_sc[di] = inv_sc_sh[di];
    center[di] = center_sh[di];
  }

  // Pass 2: quantize and permute in reverse order for L2 reuse.
  // Word stores map destination positions back to the source; byte stores
  // retain the linear-read schedule and the affine residual accumulation order.
  const int64_t out_row = static_cast<int64_t>((b * num_heads + h) * head_dim + d0);
  float residual_sum[kChannelTile] = {};

  if constexpr (PACK_OUTPUT) {
    for (int word = (padded_length / 4) - 1 - threadIdx.x; word >= 0; word -= THREADS) {
      uint32_t packed[kChannelTile] = {};
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const int dst = word * 4 + byte;
        // Forward MMA permutation: {b3,b2,b1,b0} -> {b1,b3,b2,b0}.
        const int j = dst & 15;
        const int src = (dst & ~15) | (j & 1) | ((j & 2) << 2) | ((j & 12) >> 1);
        float tmp[kChannelTile];
        if (src < sequence_length) {
          load_tile(base + (int64_t)transform.row<risa::kGatherQuantization>(src) * token_stride,
                    tmp);
#pragma unroll
          for (int di = 0; di < kChannelTile; ++di) {
            const int8_t code = risa::float_to_int8_rn((tmp[di] - center[di]) * inv_sc[di]);
            packed[di] |= uint32_t(uint8_t(code)) << (byte * 8);
          }
        }
      }
#pragma unroll
      for (int di = 0; di < kChannelTile; ++di)
        reinterpret_cast<uint32_t*>(out + (out_row + di) * padded_length)[word] = packed[di];
    }
  } else {
    for (int src = sequence_length - 1 - threadIdx.x; src >= 0; src -= THREADS) {
      const int w = src & 15;
      const int dst = (src & ~15) | inv_perm16(w);

      float tmp[kChannelTile];
      load_tile(base + (int64_t)transform.row<risa::kGatherQuantization>(src) * token_stride, tmp);
#pragma unroll
      for (int di = 0; di < kChannelTile; ++di) {
        const float normalized = (tmp[di] - center[di]) * inv_sc[di];
        const int8_t code = risa::float_to_int8_rn(normalized);
        out[(out_row + di) * padded_length + dst] = code;
        if constexpr (USE_MIDPOINT_AFFINE)
          residual_sum[di] += normalized - static_cast<float>(code);
      }
    }
  }

  if constexpr (USE_MIDPOINT_AFFINE) {
#pragma unroll
    for (int di = 0; di < kChannelTile; ++di) {
      residual_sum[di] = warp_reduce_sum(residual_sum[di]);
    }
    if (lane == 0) {
#pragma unroll
      for (int di = 0; di < kChannelTile; ++di) {
        warp_residual_sum[di][warp] = residual_sum[di];
      }
    }
    __syncthreads();

    // For the fixed midpoint codes and scale, add the mean normalized rounding
    // residual. This is the least-squares-optimal reconstruction center.
    if (threadIdx.x < kChannelTile) {
      float total_residual = 0.0f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        total_residual += warp_residual_sum[threadIdx.x][w];
      }
      const float scale = 1.0f / inv_sc_sh[threadIdx.x];
      const int parameter_index = (b * num_heads + h) * head_dim + d0 + threadIdx.x;
      center_out[parameter_index] =
          center_sh[threadIdx.x] + scale * (total_residual / static_cast<float>(sequence_length));
    }
  } else if (threadIdx.x < kChannelTile) {
    const int parameter_index = (b * num_heads + h) * head_dim + d0 + threadIdx.x;
    center_out[parameter_index] = 0.0f;
  }

  // Zero-fill padding region [sequence_length, padded_length) with the same inverse permutation
  // so that the layout matches the eager reference for partial 16-element groups.
  if constexpr (!PACK_OUTPUT) {
    for (int src = sequence_length + threadIdx.x; src < padded_length; src += THREADS) {
      const int w = src & 15;
      const int dst = (src & ~15) | inv_perm16(w);
#pragma unroll
      for (int di = 0; di < kChannelTile; ++di)
        out[(out_row + di) * padded_length + dst] = 0;
    }
  }
}

template <typename T, int THREADS>
void launch_absmax_values(const T* v, int8_t* out, float* scale, float* center, int sequence_length,
                          int padded_length, int num_heads, int head_dim, int64_t batch_stride,
                          int64_t head_stride, int64_t token_stride, risa::QuantInput transform,
                          int blocks, cudaStream_t stream) {
  // Four-byte stores use ordinary integer instructions on every supported SM.
  // Short sequences keep all token lanes active instead of grouping four tokens.
  if (sequence_length >= 4 * THREADS) {
    quantize_values_kernel<T, THREADS, false, true><<<blocks, THREADS, 0, stream>>>(
        v, out, scale, center, sequence_length, padded_length, num_heads, head_dim, batch_stride,
        head_stride, token_stride, transform);
  } else {
    quantize_values_kernel<T, THREADS, false><<<blocks, THREADS, 0, stream>>>(
        v, out, scale, center, sequence_length, padded_length, num_heads, head_dim, batch_stride,
        head_stride, token_stride, transform);
  }
}

} // namespace

extern "C" void launch_quantize_v(const void* v, void* out, void* scale, void* center,
                                  int batch_size, int num_heads, int sequence_length, int head_dim,
                                  int padded_length, int64_t batch_stride, int64_t head_stride,
                                  int64_t token_stride, int input_dtype_code,
                                  bool use_midpoint_affine, cudaStream_t stream,
                                  risa::QuantInput transform) {
  // An extent of one forms no offset, and PyTorch rewrites such a stride to 1 on
  // the way through DLPack (ATen/DLConvertor.cpp, gh-83069) on 2.9 and older, so
  // policing it would reject a tensor the caller has no way to hand over
  // differently.
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const auto stride_ok = [element_size](int64_t stride, int extent) {
    return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
  };
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(batch_stride, batch_size) ||
      !stride_ok(head_stride, num_heads) || !stride_ok(token_stride, sequence_length)) {
    throw std::runtime_error("quantize_v: V base pointer and batch_size/num_heads/sequence_length "
                             "strides must be 16-byte aligned");
  }
  if (head_dim <= 0 || head_dim % kChannelTile != 0) {
    throw std::runtime_error("quantize_v: head_dim must be a positive multiple of 8");
  }

  const int blocks = batch_size * num_heads * (head_dim / kChannelTile);

  const bool contiguous =
      token_stride == head_dim &&
      (num_heads == 1 || head_stride == static_cast<int64_t>(sequence_length) * head_dim) &&
      (batch_size == 1 ||
       batch_stride == static_cast<int64_t>(num_heads) * sequence_length * head_dim);
  const bool token_major =
      head_stride == head_dim &&
      (token_stride == static_cast<int64_t>(num_heads) * head_dim ||
       token_stride == static_cast<int64_t>(3) * num_heads * head_dim) &&
      (batch_size == 1 || batch_stride == static_cast<int64_t>(sequence_length) * token_stride);
  const bool use_wide_block =
      !use_midpoint_affine && input_dtype_code != 0 && head_dim == 128 &&
      sequence_length >= 12288 && batch_size * num_heads >= 32 && (contiguous || token_major) &&
      static_cast<int64_t>(batch_size) * num_heads * sequence_length * head_dim * element_size >=
          128 * 1024 * 1024 &&
      use_sm120_v_schedule();

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (sequence_length <= 256) {
      if (use_midpoint_affine) {
        quantize_values_kernel<T, 128, true><<<blocks, 128, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform);
      } else {
        launch_absmax_values<T, 128>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform, blocks, stream);
      }
    } else if (use_wide_block) {
      launch_absmax_values<T, 1024>(
          static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
          static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
          batch_stride, head_stride, token_stride, transform, blocks, stream);
    } else if (input_dtype_code == 2 && head_dim == 128 && token_stride > head_dim &&
               sequence_length >= 1024 && sequence_length <= 4096 && use_sm120_v_schedule()) {
      // Measured SM120 token-major D128 path: keep identical INT8 codes,
      // range, and residual-center correction with fewer reduction warps.
      if (use_midpoint_affine) {
        quantize_values_kernel<T, 256, true><<<blocks, 256, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform);
      } else {
        launch_absmax_values<T, 256>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform, blocks, stream);
      }
    } else {
      if (use_midpoint_affine) {
        quantize_values_kernel<T, 512, true><<<blocks, 512, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform);
      } else {
        launch_absmax_values<T, 512>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale),
            static_cast<float*>(center), sequence_length, padded_length, num_heads, head_dim,
            batch_stride, head_stride, token_stride, transform, blocks, stream);
      }
    }
  });

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("quantize_v kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}
