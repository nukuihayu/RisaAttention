// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace risa {
#ifdef RISA_FUSED_GATHER
static constexpr bool kGatherQuantization = true;
#else
static constexpr bool kGatherQuantization = false;
#endif

// Shared host ABI; explicit template arguments specialize only device loads.
struct QuantInput {
  const int32_t* permutation = nullptr;
#ifdef __CUDACC__
  template <bool Gather>
  __device__ __forceinline__ QuantInput head(int batch, int head_index, int heads, int length,
                                             int) const {
    QuantInput result = *this;
    if constexpr (Gather) {
      if (permutation)
        result.permutation += (int64_t(batch) * heads + head_index) * length;
    }
    return result;
  }
  template <bool Gather> __device__ __forceinline__ int row(int token) const {
    if constexpr (Gather) {
      if (permutation)
        return __ldg(permutation + token);
    }
    return token;
  }
  template <bool Gather, typename T>
  __device__ __forceinline__ float scalar(const T* base, int token, int channel, int64_t stride,
                                          int) const {
    return float(__ldg(base + int64_t(row<Gather>(token)) * stride + channel));
  }
#endif
};
} // namespace risa
