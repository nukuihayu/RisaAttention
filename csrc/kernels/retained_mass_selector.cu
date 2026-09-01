// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cstdint>

namespace {

__global__ void select_mass_rows_kernel(
    const float *__restrict__ sorted_mass,
    const int64_t *__restrict__ descending_indices,
    int32_t *__restrict__ counts, uint8_t *__restrict__ support,
    double *__restrict__ row_totals, const uint32_t row_count,
    const uint32_t key_blocks, const double theta) {
  const uint32_t row = blockIdx.x;
  if (row >= row_count) {
    return;
  }

  const uint32_t lane = threadIdx.x;
  const uint32_t base = row * key_blocks;
  double lane_total = 0.0;
  for (uint32_t rank = lane; rank < key_blocks; rank += 32) {
    lane_total += static_cast<double>(sorted_mass[base + rank]);
    support[base + rank] = 0;
  }
  double total = lane_total;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down_sync(0xffffffffU, total, offset);
  }
  total = __shfl_sync(0xffffffffU, total, 0);

  const double target = theta * total;
  double cumulative = 0.0;
  double selected = total;
  uint32_t count = key_blocks;
  for (uint32_t chunk = 0; chunk < key_blocks; chunk += 32) {
    const uint32_t rank = chunk + lane;
    double prefix =
        rank < key_blocks ? static_cast<double>(sorted_mass[base + rank]) : 0.0;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
      const double previous =
          __shfl_up_sync(0xffffffffU, prefix, offset);
      if (lane >= offset) {
        prefix += previous;
      }
    }
    const bool reaches_target = rank < key_blocks &&
                                cumulative + prefix >= target;
    const uint32_t candidates =
        __ballot_sync(0xffffffffU, reaches_target);
    if (candidates != 0) {
      const uint32_t first_lane = __ffs(candidates) - 1;
      count = chunk + first_lane + 1;
      selected = cumulative +
                 __shfl_sync(0xffffffffU, prefix, first_lane);
      break;
    }
    cumulative += __shfl_sync(0xffffffffU, prefix, 31);
  }
  __syncwarp();
  for (uint32_t rank = lane; rank < count; rank += 32) {
    support[base + static_cast<uint32_t>(descending_indices[base + rank])] = 1;
  }

  if (lane == 0) {
    counts[row] = static_cast<int32_t>(count);
    row_totals[row * 2] = selected;
    row_totals[row * 2 + 1] = total;
  }
}

__global__ void finalize_offsets_and_summary_kernel(
    const int32_t *__restrict__ counts, const uint8_t *__restrict__ support,
    const double *__restrict__ row_totals, int32_t *__restrict__ row_offsets,
    int32_t *__restrict__ block_indices, double *__restrict__ summary,
    const uint32_t row_count, const uint32_t key_blocks) {
  __shared__ double selected_sums[256];
  __shared__ double total_sums[256];

  double selected = 0.0;
  double total = 0.0;
  for (uint32_t row = threadIdx.x; row < row_count; row += blockDim.x) {
    selected += row_totals[row * 2];
    total += row_totals[row * 2 + 1];
  }
  selected_sums[threadIdx.x] = selected;
  total_sums[threadIdx.x] = total;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      selected_sums[threadIdx.x] += selected_sums[threadIdx.x + stride];
      total_sums[threadIdx.x] += total_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int32_t selected_blocks =
        row_offsets[row_count - 1] + counts[row_count - 1];
    row_offsets[row_count] = selected_blocks;
    summary[0] = selected_sums[0];
    summary[1] = total_sums[0];
    summary[2] = static_cast<double>(selected_blocks);
  }
}

__global__ void compact_mass_csr_kernel(
    const uint8_t *__restrict__ support,
    const int32_t *__restrict__ row_offsets,
    int32_t *__restrict__ block_indices, const uint32_t row_count,
    const uint32_t key_blocks) {
  constexpr uint32_t warps_per_block = 8;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * warps_per_block + warp;
  if (row < row_count) {
    uint32_t selected_before = 0;
    const uint32_t support_base = row * key_blocks;
    const uint32_t output_base = row_offsets[row];
    for (uint32_t chunk = 0; chunk < key_blocks; chunk += 32) {
      const uint32_t key = chunk + lane;
      const bool keep =
          key < key_blocks && support[support_base + key] != 0;
      const uint32_t mask = __ballot_sync(0xffffffffU, keep);
      if (keep) {
        const uint32_t lower_mask = lane == 0 ? 0 : ((1U << lane) - 1U);
        block_indices[output_base + selected_before + __popc(mask & lower_mask)] =
            static_cast<int32_t>(key);
      }
      selected_before += __popc(mask);
    }
  }
}

} // namespace

extern "C" size_t retained_mass_selector_workspace_size(int row_count) {
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes,
                                static_cast<const int32_t *>(nullptr),
                                static_cast<int32_t *>(nullptr), row_count);
  return bytes;
}

extern "C" void launch_retained_mass_selector(
    const float *sorted_mass, const int64_t *descending_indices,
    int32_t *counts, uint8_t *support, double *row_totals,
    int32_t *row_offsets, int32_t *block_indices, double *summary,
    void *scan_workspace, size_t scan_workspace_bytes, int row_count,
    int key_blocks, double theta, cudaStream_t stream) {
  select_mass_rows_kernel<<<row_count, 32, 0, stream>>>(
      sorted_mass, descending_indices, counts, support, row_totals,
      static_cast<uint32_t>(row_count), static_cast<uint32_t>(key_blocks),
      theta);
  cub::DeviceScan::ExclusiveSum(scan_workspace, scan_workspace_bytes, counts,
                                row_offsets, row_count, stream);
  finalize_offsets_and_summary_kernel<<<1, 256, 0, stream>>>(
      counts, support, row_totals, row_offsets, block_indices, summary,
      static_cast<uint32_t>(row_count), static_cast<uint32_t>(key_blocks));
  constexpr uint32_t warps_per_block = 8;
  const uint32_t blocks =
      (static_cast<uint32_t>(row_count) + warps_per_block - 1) /
      warps_per_block;
  compact_mass_csr_kernel<<<blocks, 256, 0, stream>>>(
      support, row_offsets, block_indices, static_cast<uint32_t>(row_count),
      static_cast<uint32_t>(key_blocks));
}
