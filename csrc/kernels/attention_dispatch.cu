// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. DLPack-compatible launcher for the
// Pure integer attention kernel: signed INT8 Q/K/V, unsigned INT8 softmax
// probabilities, INT32 tensor-core P*V accumulation, and FP32 online-softmax
// state. V scaling is fused; optional LSE uses natural-log units.

#include "attention_int8.cuh"
#include "device_properties.cuh"
#ifdef RISA_HAVE_SM120_TMA
#include "dense_tma_sm120.h"
#endif
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace {

template <int HEAD_DIM, int CTA_K, MaskMode mask_mode, typename DTypeOut,
          bool use_sparse_support = false, bool capture_block_mass = false, bool is_causal = false,
          int CTA_Q = 128>
void launch_impl(int8_t* q, int8_t* k, int8_t* v, DTypeOut* o, float* q_scale, float* k_scale,
                 float* v_scale, float* v_center, const void* mask, int64_t mask_stride_b,
                 int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k,
                 int mask_dtype_code, int qo_len, int kv_len, int num_qo_heads, int num_kv_groups,
                 int64_t stride_bz_q, int64_t stride_seq_q, int64_t stride_h_q, int64_t stride_bz_k,
                 int64_t stride_seq_k, int64_t stride_h_k, int64_t stride_bz_v, int64_t stride_h_v,
                 int64_t stride_d_v, int64_t stride_bz_o, int64_t stride_seq_o, int64_t stride_h_o,
                 float sm_scale, int batch_size, const int32_t* sparse_row_offsets,
                 const int32_t* sparse_block_indices, float* block_mass, float* lse,
                 cudaStream_t stream, const int32_t* key_mask_tiles = nullptr) {
  // Tiling constants must match attention.py and bindings.cpp.
  // D>=128 otherwise needs too many live FP32 output accumulators per thread.
  // A 16-row warp tile halves that accumulator set.
  constexpr int WARP_Q = HEAD_DIM >= 128 ? 16 : 32;
  constexpr int WARP_K = CTA_K;

  size_t smem_max = std::max(static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                                 CTA_K * HEAD_DIM * sizeof(int8_t) +
                                                 CTA_K * HEAD_DIM * sizeof(int8_t)),
                             static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

  auto kernel =
      int8_attention_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
                            QuantGranularity::kPerThread, QuantGranularity::kPerThread, DTypeOut,
                            mask_mode, use_sparse_support, capture_block_mass, is_causal>;
  if constexpr (mask_mode == MaskMode::kNone && !is_causal) {
    if (sm_scale <= 0.0f)
      kernel = int8_attention_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
                                     QuantGranularity::kPerThread, QuantGranularity::kPerThread,
                                     DTypeOut, mask_mode, use_sparse_support, capture_block_mass,
                                     is_causal, false>;
  }
  if constexpr (mask_mode == MaskMode::kCustomKey && !use_sparse_support && !capture_block_mass &&
                !is_causal) {
    if (key_mask_tiles != nullptr)
      kernel = int8_attention_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
                                     QuantGranularity::kPerThread, QuantGranularity::kPerThread,
                                     DTypeOut, mask_mode, false, false, false, true, true>;
  }

  // This limit is invariant for a kernel specialization on one device. Avoid
  // resetting driver metadata at every attention call. Separate thread-local
  // caches avoid synchronization and reconfigure on device switches.
  static thread_local int configured_devices[3] = {-1, -1, -1};
  const int kernel_index =
      key_mask_tiles != nullptr
          ? 2
          : ((mask_mode == MaskMode::kNone && !is_causal && sm_scale <= 0.0f) ? 1 : 0);
  int& configured_device = configured_devices[kernel_index];
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  if (device != configured_device) {
    error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem_max));
    if (error != cudaSuccess) {
      throw std::runtime_error("risa_attention failed to request " + std::to_string(smem_max) +
                               " bytes of dynamic shared memory: " + cudaGetErrorString(error));
    }
    configured_device = device;
  }

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, v_center, mask, mask_stride_b, mask_stride_h,
      mask_stride_q, mask_stride_k, mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q,
      stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
      stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, sparse_row_offsets,
      sparse_block_indices, block_mass, key_mask_tiles);

  if constexpr (capture_block_mass) {
    const uint32_t num_mass_blocks = div_ceil(kv_len, CTA_K);
    dim3 mass_grid(grid.x * num_mass_blocks, num_qo_heads, batch_size);
    captured_block_mass_kernel<CTA_Q><<<mass_grid, CTA_Q, 0, stream>>>(
        block_mass, qo_len, num_mass_blocks, grid.x, num_qo_heads);
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("risa_attention kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

} // anonymous namespace

extern "C" void
launch_attention(const void* q, const void* k, const void* v, void* o, const void* q_scale,
                 const void* k_scale, const void* v_scale, const void* v_center, const void* mask,
                 int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
                 int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch_size, int qo_len,
                 int kv_len, int num_qo_heads, int num_kv_heads, int head_dim, int64_t stride_bz_q,
                 int64_t stride_seq_q, int64_t stride_h_q, int64_t stride_bz_k,
                 int64_t stride_seq_k, int64_t stride_h_k, int64_t stride_bz_v, int64_t stride_h_v,
                 int64_t stride_d_v, int64_t stride_bz_o, int64_t stride_seq_o, int64_t stride_h_o,
                 float sm_scale, int output_dtype_code, const void* sparse_row_offsets,
                 const void* sparse_block_indices, void* block_mass, bool is_causal, void* lse,
                 cudaStream_t stream, const int32_t* key_mask_tiles) {
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("risa_attention: cta_k must be 64 or 128");
  }
  if (cta_k == 128 && head_dim == 64) {
    throw std::runtime_error("risa_attention: cta_k 128 requires head_dim 128 or 256");
  }
  const bool has_sparse_support = sparse_row_offsets != nullptr || sparse_block_indices != nullptr;
  const bool has_block_mass = block_mass != nullptr;
  if (has_sparse_support && (sparse_row_offsets == nullptr || sparse_block_indices == nullptr)) {
    throw std::runtime_error(
        "risa_attention: sparse row offsets and block indices must be provided together");
  }

  if (has_block_mass && has_sparse_support) {
    throw std::runtime_error("risa_attention: block mass requires dense attention");
  }
  int num_kv_groups = num_qo_heads / num_kv_heads;

  // Upstream kernel uses non-const pointers; cast away const from the
  // extern "C" boundary (kernel does not modify inputs).
  auto query_codes = const_cast<int8_t*>(static_cast<const int8_t*>(q));
  auto key_codes = const_cast<int8_t*>(static_cast<const int8_t*>(k));
  auto value_codes = const_cast<int8_t*>(static_cast<const int8_t*>(v));
  auto query_scales = const_cast<float*>(static_cast<const float*>(q_scale));
  auto key_scales = const_cast<float*>(static_cast<const float*>(k_scale));
  auto value_scales = const_cast<float*>(static_cast<const float*>(v_scale));
  auto value_centers = const_cast<float*>(static_cast<const float*>(v_center));

#ifdef RISA_HAVE_SM120_TMA
  // Tensor maps use uint32 offsets; all other layouts and optional features
  // retain the general kernel's 64-bit addressing and mask/LSE semantics.
  const int64_t max_stride =
      std::max({stride_bz_q, stride_h_q, stride_seq_q, stride_bz_k, stride_h_k, stride_seq_k,
                stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, stride_h_o, stride_seq_o});
  const bool bounded_offsets =
      max_stride <= UINT32_MAX &&
      static_cast<uint64_t>(batch_size - 1) * max_stride <= UINT32_MAX &&
      static_cast<uint64_t>(num_qo_heads - 1) * std::max(stride_h_q, stride_h_o) <= UINT32_MAX &&
      static_cast<uint64_t>(num_kv_heads - 1) * std::max(stride_h_k, stride_h_v) <= UINT32_MAX;
  if (!has_sparse_support && !has_block_mass && !is_causal && !mask && !lse && head_dim == 128 &&
      cta_k == 128 && output_dtype_code == 2 && qo_len >= 8192 && kv_len >= 8192 &&
      bounded_offsets) {
    DenseTmaArgs args{query_codes,
                      key_codes,
                      value_codes,
                      o,
                      query_scales,
                      key_scales,
                      value_scales,
                      value_centers,
                      uint32_t(batch_size),
                      uint32_t(qo_len),
                      uint32_t(kv_len),
                      uint32_t(num_qo_heads),
                      uint32_t(num_kv_heads),
                      uint32_t(stride_bz_q),
                      uint32_t(stride_h_q),
                      uint32_t(stride_seq_q),
                      uint32_t(stride_bz_k),
                      uint32_t(stride_h_k),
                      uint32_t(stride_seq_k),
                      uint32_t(stride_bz_v),
                      uint32_t(stride_h_v),
                      uint32_t(stride_d_v),
                      uint32_t(stride_bz_o),
                      uint32_t(stride_h_o),
                      uint32_t(stride_seq_o),
                      sm_scale};
    if (launch_dense_tma_sm120(args, stream))
      return;
  }
#endif

  // Upstream's SM120 image-size D128 schedule. Dense tiles may be smaller
  // than the 128-row Q quantization groups, which the kernel indexes explicitly.
  if (!has_sparse_support && !has_block_mass && !is_causal && !mask && !lse && head_dim == 128 &&
      cta_k == 128 && qo_len >= 4096 && qo_len <= 4608 && kv_len >= 4096 && kv_len <= 4608 &&
      num_qo_heads <= 32) {
    const auto& properties = attention_device_properties();
    const int multiprocessors = properties.multiprocessors;
#ifdef RISA_SM120_ONLY
    bool smaller_tile = true;
#else
    bool smaller_tile = properties.major == 12;
#endif
    if (smaller_tile && num_qo_heads >= 30) {
      const int64_t batch_heads = static_cast<int64_t>(batch_size) * num_qo_heads;
      const int64_t rounds_64 = div_ceil(batch_heads * div_ceil(qo_len, 64), multiprocessors);
      const int64_t rounds_128 = div_ceil(batch_heads * div_ceil(qo_len, 128), multiprocessors);
      smaller_tile = rounds_64 < 2 * rounds_128;
    }
    if (smaller_tile) {
      auto run = [&]<typename Output>() {
        launch_impl<128, 128, MaskMode::kNone, Output, false, false, false, 64>(
            query_codes, key_codes, value_codes, static_cast<Output*>(o), query_scales, key_scales,
            value_scales, value_centers, mask, mask_stride_b, mask_stride_h, mask_stride_q,
            mask_stride_k, mask_dtype_code, qo_len, kv_len, num_qo_heads, num_kv_groups,
            stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k,
            stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale,
            batch_size, nullptr, nullptr, nullptr, nullptr, stream);
      };
      if (output_dtype_code == 1)
        run.template operator()<half>();
      else
        run.template operator()<nv_bfloat16>();
      return;
    }
  }

  // One dispatch tree keeps dense, sparse and mass capture on the same
  // ConvRot-quantized Q/K and masked score implementations.
  auto dispatch = [&]<int HD, int CK, MaskMode MM, typename DT>() {
    auto run = [&]<bool SPARSE, bool MASS, bool CAUSAL>() {
      launch_impl<HD, CK, MM, DT, SPARSE, MASS, CAUSAL>(
          query_codes, key_codes, value_codes, static_cast<DT*>(o), query_scales, key_scales,
          value_scales, value_centers, mask, mask_stride_b, mask_stride_h, mask_stride_q,
          mask_stride_k, mask_dtype_code, qo_len, kv_len, num_qo_heads, num_kv_groups, stride_bz_q,
          stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, batch_size,
          static_cast<const int32_t*>(sparse_row_offsets),
          static_cast<const int32_t*>(sparse_block_indices), static_cast<float*>(block_mass),
          static_cast<float*>(lse), stream, key_mask_tiles);
    };
    if (has_block_mass) {
      if (is_causal)
        run.template operator()<false, true, true>();
      else
        run.template operator()<false, true, false>();
    } else if (has_sparse_support) {
      if (is_causal)
        run.template operator()<true, false, true>();
      else
        run.template operator()<true, false, false>();
    } else {
      if (is_causal)
        run.template operator()<false, false, true>();
      else
        run.template operator()<false, false, false>();
    }
  };
  auto dispatch_mask = [&]<int HD, int CK, typename DT>() {
    if (mask == nullptr)
      dispatch.template operator()<HD, CK, MaskMode::kNone, DT>();
    else if (mask_stride_q == 0)
      dispatch.template operator()<HD, CK, MaskMode::kCustomKey, DT>();
    else
      dispatch.template operator()<HD, CK, MaskMode::kCustom, DT>();
  };
  auto dispatch_head = [&]<int HD>() {
    if constexpr (HD != 64) {
      if (cta_k == 128) {
        if (output_dtype_code == 1)
          dispatch_mask.template operator()<HD, 128, half>();
        else
          dispatch_mask.template operator()<HD, 128, nv_bfloat16>();
        return;
      }
    }
    if (output_dtype_code == 1)
      dispatch_mask.template operator()<HD, 64, half>();
    else
      dispatch_mask.template operator()<HD, 64, nv_bfloat16>();
  };
  if (head_dim == 64)
    dispatch_head.template operator()<64>();
  else if (head_dim == 128)
    dispatch_head.template operator()<128>();
  else if (head_dim == 256)
    dispatch_head.template operator()<256>();
  else
    throw std::runtime_error("risa_attention: unsupported head_dim");
}
