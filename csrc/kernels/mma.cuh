// SPDX-License-Identifier: Apache-2.0
// Derived from SageAttention (https://github.com/thu-ml/SageAttention), commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. INT8 kernels require SM80 or newer.

/*
 * Adapted from Flashinfer,
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/mma.cuh
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Modifications copyright (c) 2024 by SageAttention team.
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "RISA Attention requires SM80 or newer"
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace mma {

__device__ __forceinline__ uint32_t pack_u8x4(float a, float b, float c, float d) {
  uint32_t qa, qb, qc, qd;
  // Keeping the conversions adjacent lets ptxas combine each pair into one
  // F2IP instruction on Ampere and newer.
  asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qa) : "f"(a));
  asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qb) : "f"(b));
  asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qc) : "f"(c));
  asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qd) : "f"(d));
  uint32_t qdc, packed;
  asm volatile("cvt.pack.sat.u8.s32.b32 %0, %1, %2, 0;" : "=r"(qdc) : "r"(qd), "r"(qc));
  asm volatile("cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;"
               : "=r"(packed)
               : "r"(qb), "r"(qa), "r"(qdc));
  return packed;
}

__device__ __forceinline__ uint32_t pack_u8x4_bounded(float a, float b, float c, float d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  const uint32_t qa = __float_as_uint(a + 8388608.0f);
  const uint32_t qb = __float_as_uint(b + 8388608.0f);
  const uint32_t qc = __float_as_uint(c + 8388608.0f);
  const uint32_t qd = __float_as_uint(d + 8388608.0f);
  const uint32_t ab = __byte_perm(qa, qb, 0x0040); // {a0, b0, .., ..}
  const uint32_t cd = __byte_perm(qc, qd, 0x0040); // {c0, d0, .., ..}
  return __byte_perm(ab, cd, 0x5410);              // {a0, b0, c0, d0}
#else
  return pack_u8x4(a, b, c, d);
#endif
}

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x2 instruction, loads data from shared
 * memory to fragment \tparam T data type of the fragment \param R pointer to
 * the fragment \param smem_ptr pointer to the shared memory
 */
template <typename T> __device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t* R, T* smem_ptr) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared
 * memory to fragment \tparam T data type of the fragment \param R pointer to
 * the fragment \param smem_ptr pointer to the shared memory
 */
template <typename T> __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data
 * from shared memory to fragment and transposes the fragment \tparam T data
 * type of the fragment \param R pointer to the fragment \param smem_ptr pointer
 * to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major
 * f16 matrix multiplication, accumulated in f32. \tparam mma_mode The mode of
 * mma instruction, either kInit or kInplaceUpdate \param C pointer to the
 * accumulator \param A pointer to the fragment of matrix A \param B pointer to
 * the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column
 * major f16 matrix multiplication, accumulated in f32. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major
 * f16 matrix multiplication, accumulated in f16. \tparam mma_mode The mode of
 * mma instruction, either kInit or kInplaceUpdate \param C pointer to the
 * accumulator \param A pointer to the fragment of matrix A \param B pointer to
 * the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column
 * major f16 matrix multiplication, accumulated in f16. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major
 * int8 matrix multiplication, accumulated in int32. \tparam mma_mode The mode
 * of mma instruction, either kInit or kInplaceUpdate \param C pointer to the
 * accumulator \param A pointer to the fragment of matrix A \param B pointer to
 * the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                  uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]),
                   "r"(C[1]), "r"(C[2]), "r"(C[3]));
  } else if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
  }
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column
 * major int8 matrix multiplication, accumulated in int32. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]),
                   "r"(C[1]), "r"(C[2]), "r"(C[3]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]),
                   "r"(C[5]), "r"(C[6]), "r"(C[7]));
  } else if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
  }
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_u8s8s32(int32_t* C, uint32_t* A,
                                                                  uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]),
                   "r"(C[1]), "r"(C[2]), "r"(C[3]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
  }
}

/*!
 * \brief Wrapper of two m16n8k32 instructions for an unsigned INT8 row-major
 * matrix and signed INT8 column-major matrix, accumulated in INT32. Softmax
 * probabilities naturally use the full unsigned 0..255 range while V remains
 * symmetrically quantized to signed INT8.
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_u8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]),
                   "r"(C[1]), "r"(C[2]), "r"(C[3]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]),
                   "r"(C[5]), "r"(C[6]), "r"(C[7]));
  } else if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
                   "r"(0), "r"(0));
  }
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major
 * int4 matrix multiplication, accumulated in int32. \tparam mma_mode The mode
 * of mma instruction, either kInit or kInplaceUpdate \param C pointer to the
 * accumulator \param A pointer to the fragment of matrix A \param B pointer to
 * the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n16k64 instruction for row major and column
 * major int4 matrix multiplication, accumulated in int32. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major
 * fp8 e4m3 matrix multiplication, accumulated in fp32. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column
 * major fp8 matrix multiplication, accumulated in fp16. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column
 * major fp8 matrix multiplication, accumulated in fp32. \tparam mma_mode The
 * mode of mma instruction, either kInit or kInplaceUpdate \param C pointer to
 * the accumulator \param A pointer to the fragment of matrix A \param B pointer
 * to the fragment of matrix B
 */

/*!
 * \brief Use mma instructions to compute rowsum.
 */

/*!
 * \brief Use mma instructions to compute rowsum.
 */

} // namespace mma
