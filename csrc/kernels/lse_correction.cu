// SPDX-License-Identifier: Apache-2.0
#include "dtype_dispatch.cuh"
namespace {
template <typename T>
__global__ void lse_shift_kernel(const T* q, const T* k, const int* anchors, float* out, int Hq,
                                 int Hk, int Nq, int head_dim, int64_t qb, int64_t qh, int64_t qn,
                                 int64_t kb, int64_t kh, int64_t kn, float scale) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  const int head = blockIdx.y, batch = blockIdx.z;
  if (row >= Nq)
    return;
  const int kv_head = head / (Hq / Hk);
  const int anchor = anchors[batch * Hk + kv_head];
  float sum = 0.0f;
  if (anchor >= 0) {
    const T* qr = q + batch * qb + head * qh + row * qn;
    const T* kr = k + batch * kb + kv_head * kh + anchor * kn;
    for (int d = lane; d < head_dim; d += 32) {
      // Match the reference FP32 multiply then reduce (no contracted FMA).
      sum = __fadd_rn(sum, __fmul_rn(float(qr[d]), float(kr[d])));
    }
    for (int step = 16; step; step >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, step);
  }
  if (lane == 0)
    out[(int64_t(batch) * Hq + head) * Nq + row] = sum * scale;
}
} // namespace

extern "C" void launch_correct_lse(const void* q, const void* k, const int* anchors, float* out,
                                   int batch_size, int Hq, int Hk, int Nq, int head_dim, int64_t qb,
                                   int64_t qh, int64_t qn, int64_t kb, int64_t kh, int64_t kn,
                                   float scale, int dtype, cudaStream_t stream) {
  const dim3 grid((Nq + 3) / 4, Hq, batch_size);
  DISPATCH_FP_DTYPE(dtype, T, [&] {
    lse_shift_kernel<T><<<grid, 128, 0, stream>>>(static_cast<const T*>(q),
                                                  static_cast<const T*>(k), anchors, out, Hq, Hk,
                                                  Nq, head_dim, qb, qh, qn, kb, kh, kn, scale);
  });
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("LSE shift launch failed: ") + cudaGetErrorString(error));
}
