// SPDX-License-Identifier: Apache-2.0
// SPDX-FileContributor: Adapted from comfy-kitchen masked-tile skipping.
#include "attention_masks.cuh"
#include "launchers.h"

#include <stdexcept>

namespace {

template <uint32_t tile_size>
__global__ void prepare_key_mask_tiles_kernel(const void* mask, int32_t* output, int heads,
                                              int length, int64_t stride_b, int64_t stride_h,
                                              int64_t stride_k, int dtype_code) {
  constexpr uint32_t warps = 4;
  const uint32_t tiles = div_ceil(length, tile_size);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * heads + blockIdx.y;
  const int64_t mask_row =
      static_cast<int64_t>(blockIdx.x) * stride_b + static_cast<int64_t>(blockIdx.y) * stride_h;
  int32_t* flags = output + row * (tiles + 1);
  for (uint32_t tile = threadIdx.y; tile < tiles; tile += warps) {
    const bool excluded =
        key_mask_tile_is_empty<tile_size>(mask, mask_row, stride_k, tile, length, dtype_code);
    if (threadIdx.x == 0)
      flags[tile] = excluded;
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    uint32_t active_count = 0;
    for (uint32_t base = 0; base < tiles; base += 32) {
      const uint32_t tile = base + threadIdx.x;
      const bool active = tile < tiles && flags[tile] == 0;
      const uint32_t votes = __ballot_sync(0xffffffffu, active);
      const uint32_t rank = __popc(votes & ((1u << threadIdx.x) - 1));
      __syncwarp();
      if (active)
        flags[active_count + rank] = tile;
      __syncwarp();
      active_count += __popc(votes);
    }
    if (threadIdx.x == 0)
      flags[tiles] = active_count;
  }
}

} // namespace

extern "C" void launch_prepare_key_mask_tiles(const void* mask, int32_t* tiles, int batch_size,
                                              int query_heads, int key_length, int tile_size,
                                              int64_t stride_b, int64_t stride_h, int64_t stride_k,
                                              int dtype_code, cudaStream_t stream) {
  const dim3 grid(batch_size, query_heads), block(32, 4);
  if (tile_size == 64)
    prepare_key_mask_tiles_kernel<64><<<grid, block, 0, stream>>>(
        mask, tiles, query_heads, key_length, stride_b, stride_h, stride_k, dtype_code);
  else
    prepare_key_mask_tiles_kernel<128><<<grid, block, 0, stream>>>(
        mask, tiles, query_heads, key_length, stride_b, stride_h, stride_k, dtype_code);
  const auto error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
