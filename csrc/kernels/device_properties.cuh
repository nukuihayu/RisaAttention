// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

struct AttentionDeviceProperties {
  int device = -1;
  int major = 0;
  int minor = 0;
  int multiprocessors = 0;
};

inline const AttentionDeviceProperties& attention_device_properties() {
  int device;
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  static thread_local AttentionDeviceProperties cached;
  if (cached.device != device) {
    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(error));
    cached = {device, properties.major, properties.minor, properties.multiProcessorCount};
  }
  return cached;
}
