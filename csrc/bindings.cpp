/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>


namespace nb = nanobind;

// Helper: Map nanobind dtype to internal dtype code
// Returns: 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Float) {
        if (dtype.bits == 32) return 0;  // float32
        if (dtype.bits == 16) return 1;  // float16
        if (dtype.bits == 8) return 5;   // float8_e4m3fn (default)
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Bfloat && dtype.bits == 16) {
        return 2;  // bfloat16
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::UInt && dtype.bits == 8) {
        return 3;  // uint8
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Int && dtype.bits == 8) {
        return 4;  // int8
    }
    return -1;  // unsupported
}


extern "C" {
void launch_quant_qk_per_thread_int8(
    const void* q, void* q_int8, void* q_scale,
    const void* k, void* k_int8, void* k_scale,
    int B, int H_q, int Lq, int H_kv, int Lk, int C,
    int BLKQ, int WARPQ, int BLKK, int WARPK,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int input_dtype_code, void* anchor_indices, cudaStream_t stream);

void launch_quant_v_int8_kernel(
    const void* v, void* out, void* scale, void* center,
    int B, int H, int N, int D, int padded_N,
    int64_t sb, int64_t sh, int64_t sn,
    int input_dtype_code, cudaStream_t stream);

void launch_risa_attention_kernel(
    const void* q, const void* k, const void* v, void* o,
    const void* q_scale, const void* k_scale, const void* v_scale,
    const void* v_center,
    const void* mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int cta_k, int B, int Lq, int Lk, int H_q, int H_kv, int D,
    int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h,
    float sm_scale, int output_dtype_code, const void* sparse_row_offsets,
    const void* sparse_block_indices, void* block_mass, cudaStream_t stream);
}

// Nanobind wrapper: signed INT8 V quantization

void quant_v_int8(
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> out,
    nb::ndarray<nb::device::cuda> scale,
    nb::ndarray<nb::device::cuda> center,
    int padded_n,
    int input_dtype_code,
    uintptr_t stream_ptr)
{
    if (v.ndim() != 4) {
        throw std::runtime_error("quant_v_int8: v must be 4D [B,H,N,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quant_v_int8_kernel(
        v.data(), out.data(), scale.data(), center.data(),
        static_cast<int>(v.shape(0)),
        static_cast<int>(v.shape(1)),
        static_cast<int>(v.shape(2)),
        static_cast<int>(v.shape(3)),
        padded_n,
        v.stride(0), v.stride(1), v.stride(2),
        input_dtype_code, stream);
}

// Nanobind wrapper: stabilized INT8 Q/K per-thread quant (contiguous HND layout)
void quant_qk_per_thread_int8(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    int BLKQ, int WARPQ, int BLKK, int WARPK,
    int input_dtype_code,
    uintptr_t stream_ptr,
    uintptr_t anchor_indices_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("quant_qk_per_thread_int8: q and k must be 4D [B,H,L,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quant_qk_per_thread_int8(
        q.data(), q_int8.data(), q_scale.data(),
        k.data(), k_int8.data(), k_scale.data(),
        static_cast<int>(q.shape(0)),
        static_cast<int>(q.shape(1)),
        static_cast<int>(q.shape(2)),
        static_cast<int>(k.shape(1)),
        static_cast<int>(k.shape(2)),
        static_cast<int>(q.shape(3)),
        BLKQ, WARPQ, BLKK, WARPK,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        input_dtype_code,
        reinterpret_cast<void *>(anchor_indices_ptr), stream);
}

// Quantization half of the split INT8 SDPA API.  This deliberately launches
// the same Q/K and V kernels with the same tiling as risa_sdpa below, so moving
// the attention launch after the caller releases its input tensors does not
// change any numerical results.
void risa_sdpa_quantize(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> v_scale,
    nb::ndarray<nb::device::cuda> v_center,
    int cta_k,
    int input_dtype_code,
    uintptr_t stream_ptr,
    uintptr_t anchor_indices_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
        throw std::runtime_error(
            "risa_sdpa_quantize: q, k, and v must be 4D [B,H,L,D]");
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("risa_sdpa_quantize: cta_k must be 64 or 128");
    }
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error(
            "risa_sdpa_quantize: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    }
    if (!anchor_indices_ptr) {
        throw std::runtime_error(
            "risa_sdpa_quantize: anchor_indices scratch is required");
    }

    const int B = static_cast<int>(q.shape(0));
    const int H_q = static_cast<int>(q.shape(1));
    const int Lq = static_cast<int>(q.shape(2));
    const int D = static_cast<int>(q.shape(3));
    const int H_kv = static_cast<int>(k.shape(1));
    const int Lk = static_cast<int>(k.shape(2));
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    if (cta_k == 128 && D == 64) {
        throw std::runtime_error(
            "risa_sdpa_quantize: cta_k 128 is unsupported for head_dim 64");
    }

    if (k.shape(0) != B || v.shape(0) != B || v.shape(1) != H_kv ||
        v.shape(2) != Lk || k.shape(3) != D || v.shape(3) != D) {
        throw std::runtime_error("risa_sdpa_quantize: incompatible q, k, and v shapes");
    }
    if (q_int8.ndim() != 4 || k_int8.ndim() != 4 || v_int8.ndim() != 2 ||
        q_int8.shape(0) != B || q_int8.shape(1) != H_q ||
        q_int8.shape(2) != Lq || q_int8.shape(3) != D ||
        k_int8.shape(0) != B || k_int8.shape(1) != H_kv ||
        k_int8.shape(2) != Lk || k_int8.shape(3) != D ||
        v_int8.shape(0) != static_cast<size_t>(B) * H_kv * D ||
        v_int8.shape(1) != padded_Lk) {
        throw std::runtime_error("risa_sdpa_quantize: incompatible INT8 output shapes");
    }

    constexpr int BLKQ = 128;
    const int WARPQ = D == 256 ? 16 : 32;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_quant_qk_per_thread_int8(
        q.data(), q_int8.data(), q_scale.data(),
        k.data(), k_int8.data(), k_scale.data(),
        B, H_q, Lq, H_kv, Lk, D,
        BLKQ, WARPQ, cta_k, cta_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        input_dtype_code,
        reinterpret_cast<void *>(anchor_indices_ptr), stream);

    launch_quant_v_int8_kernel(
        v.data(), v_int8.data(), v_scale.data(), v_center.data(),
        B, H_kv, Lk, D, padded_Lk,
        v.stride(0), v.stride(1), v.stride(2),
        input_dtype_code, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("risa_sdpa_quantize kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

// Attention half of the split INT8 SDPA API.  The input tensors use the exact
// packed layouts produced by risa_sdpa_quantize; no floating-point Q/K/V
// tensor is retained or reconstructed.
void risa_sdpa_prequantized(
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale,
    nb::ndarray<nb::device::cuda> v_center,
    int cta_k,
    float sm_scale,
    int output_dtype_code,
    uintptr_t stream_ptr,
    std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> sparse_row_offsets = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> sparse_block_indices = std::nullopt,
    std::optional<nb::ndarray<nb::device::cuda>> block_mass = std::nullopt)
{
    if (q_int8.ndim() != 4 || k_int8.ndim() != 4 ||
        v_int8.ndim() != 2 || o.ndim() != 4) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: q/k/o must be 4D and packed v must be 2D");
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("risa_sdpa_prequantized: cta_k must be 64 or 128");
    }
    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }

    const int B = static_cast<int>(q_int8.shape(0));
    const int H_q = static_cast<int>(q_int8.shape(1));
    const int Lq = static_cast<int>(q_int8.shape(2));
    const int D = static_cast<int>(q_int8.shape(3));
    const int H_kv = static_cast<int>(k_int8.shape(1));
    const int Lk = static_cast<int>(k_int8.shape(2));
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    if (cta_k == 128 && (D == 64 || attn_mask.has_value())) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: cta_k 128 requires unmasked head_dim 128 or 256");
    }
    if (sparse_row_offsets.has_value() != sparse_block_indices.has_value()) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: sparse row offsets and block indices must be provided together");
    }
    if (attn_mask.has_value() && sparse_row_offsets.has_value()) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: sparse support cannot be combined with an attention mask");
    }
    if (block_mass.has_value() &&
        (attn_mask.has_value() || sparse_row_offsets.has_value())) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: block mass requires dense unmasked attention");
    }
    if (k_int8.shape(0) != B || k_int8.shape(3) != D ||
        o.shape(0) != B || o.shape(1) != H_q || o.shape(2) != Lq ||
        o.shape(3) != D ||
        v_int8.shape(0) != static_cast<size_t>(B) * H_kv * D ||
        v_int8.shape(1) != padded_Lk) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: incompatible quantized tensor shapes");
    }
    const bool output_hnd =
        (Lq <= 1 || o.stride(2) == D) &&
        (H_q <= 1 || o.stride(1) == static_cast<int64_t>(Lq) * D);
    const bool output_nhd =
        (H_q <= 1 || o.stride(1) == D) &&
        (Lq <= 1 || o.stride(2) == static_cast<int64_t>(H_q) * D);
    if (q_int8.stride(3) != 1 || q_int8.stride(2) != D ||
        q_int8.stride(1) != static_cast<int64_t>(Lq) * D ||
        k_int8.stride(3) != 1 || k_int8.stride(2) != D ||
        k_int8.stride(1) != static_cast<int64_t>(Lk) * D ||
        v_int8.stride(1) != 1 || v_int8.stride(0) != padded_Lk ||
        o.stride(3) != 1 || (!output_hnd && !output_nhd)) {
        throw std::runtime_error(
            "risa_sdpa_prequantized: quantized tensors must be contiguous and "
            "output must use HND- or NHD-contiguous storage");
    }

    const void *mask_ptr = nullptr;
    int64_t mask_stride_b = 0;
    int64_t mask_stride_h = 0;
    int64_t mask_stride_q = 0;
    int64_t mask_stride_k = 0;
    int mask_dtype_code = -1;
    if (attn_mask.has_value()) {
        const auto &mask = attn_mask.value();
        if (mask.ndim() != 4 || mask.shape(0) != B || mask.shape(1) != H_q ||
            mask.shape(2) != Lq || mask.shape(3) != Lk) {
            throw std::runtime_error(
                "risa_sdpa_prequantized: attention mask must be expanded to [B,H_q,Lq,Lk]");
        }
        if (mask.dtype().code == (uint8_t)nb::dlpack::dtype_code::Bool) {
            mask_dtype_code = 3;
        } else {
            mask_dtype_code = map_dtype_to_code(mask.dtype());
        }
        if (mask_dtype_code < 0 || mask_dtype_code > 3) {
            throw std::runtime_error(
                "risa_sdpa_prequantized: attention mask must be bool, float16, bfloat16, or float32");
        }
        mask_ptr = mask.data();
        mask_stride_b = mask.stride(0);
        mask_stride_h = mask.stride(1);
        mask_stride_q = mask.stride(2);
        mask_stride_k = mask.stride(3);
    }

    const void *sparse_row_offsets_ptr = nullptr;
    const void *sparse_block_indices_ptr = nullptr;
    if (sparse_row_offsets.has_value()) {
        const auto &row_offsets = sparse_row_offsets.value();
        const auto &block_indices = sparse_block_indices.value();
        const int num_q_blocks = (Lq + 127) / 128;
        const int num_k_blocks = (Lk + cta_k - 1) / cta_k;
        const int64_t expected_rows = static_cast<int64_t>(B) * H_q * num_q_blocks;
        const auto int_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
        if (row_offsets.ndim() != 1 ||
            row_offsets.shape(0) != static_cast<size_t>(expected_rows + 1) ||
            block_indices.ndim() != 1 ||
            row_offsets.dtype().code != int_code || row_offsets.dtype().bits != 32 ||
            block_indices.dtype().code != int_code || block_indices.dtype().bits != 32) {
            throw std::runtime_error(
                "risa_sdpa_prequantized: sparse support must be contiguous int32 CSR with B*Hq*ceil(Lq/128)+1 row offsets");
        }
        if (row_offsets.stride(0) != 1 || block_indices.stride(0) != 1 ||
            block_indices.shape(0) == 0 || num_k_blocks <= 0) {
            throw std::runtime_error(
                "risa_sdpa_prequantized: sparse CSR tensors must be non-empty and contiguous");
        }
        sparse_row_offsets_ptr = row_offsets.data();
        sparse_block_indices_ptr = block_indices.data();
    }

    void *block_mass_ptr = nullptr;
    if (block_mass.has_value()) {
        auto &mass = block_mass.value();
        const int num_q_blocks = (Lq + 127) / 128;
        const int num_k_blocks = (Lk + cta_k - 1) / cta_k;
        const auto float_code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
        if (mass.ndim() != 2 ||
            mass.shape(0) != static_cast<size_t>(B) * H_q * num_q_blocks ||
            mass.shape(1) != static_cast<size_t>(num_k_blocks) ||
            mass.dtype().code != float_code || mass.dtype().bits != 32 ||
            mass.stride(1) != 1 || mass.stride(0) != num_k_blocks) {
            throw std::runtime_error(
                "risa_sdpa_prequantized: block_mass must be contiguous float32 "
                "[B*Hq*ceil(Lq/128), ceil(Lk/cta_k)]");
        }
        block_mass_ptr = mass.data();
    }

    const int64_t qi_st_bz64 = static_cast<int64_t>(H_q) * Lq * D;
    const int64_t ki_st_bz64 = static_cast<int64_t>(H_kv) * Lk * D;
    const int64_t v_st_bz64 = static_cast<int64_t>(H_kv) * D * padded_Lk;
    const int64_t o_st_bz64 =
        B <= 1 ? static_cast<int64_t>(H_q) * Lq * D : o.stride(0);
    const int64_t o_st_n64 = Lq <= 1 ? D : o.stride(2);
    const int64_t o_st_h64 = H_q <= 1 ? static_cast<int64_t>(Lq) * D : o.stride(1);
    if (qi_st_bz64 > INT_MAX || ki_st_bz64 > INT_MAX || v_st_bz64 > INT_MAX ||
        o_st_bz64 > INT_MAX || o_st_n64 > INT_MAX || o_st_h64 > INT_MAX) {
        throw std::overflow_error(
            "risa_sdpa_prequantized: tensor strides exceed int32 range; reduce batch/seq/head dimensions");
    }

    const int qi_st_h = Lq * D;
    const int qi_st_n = D;
    const int qi_st_bz = static_cast<int>(qi_st_bz64);
    const int ki_st_h = Lk * D;
    const int ki_st_n = D;
    const int ki_st_bz = static_cast<int>(ki_st_bz64);
    const int v_st_d = padded_Lk;
    const int v_st_h = D * padded_Lk;
    const int v_st_bz = static_cast<int>(v_st_bz64);
    const int o_st_h = static_cast<int>(o_st_h64);
    const int o_st_n = static_cast<int>(o_st_n64);
    const int o_st_bz = static_cast<int>(o_st_bz64);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_risa_attention_kernel(
        q_int8.data(), k_int8.data(), v_int8.data(), o.data(),
        q_scale.data(), k_scale.data(), v_scale.data(), v_center.data(),
        mask_ptr, mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
        mask_dtype_code, cta_k,
        B, Lq, Lk, H_q, H_kv, D,
        qi_st_bz, qi_st_n, qi_st_h,
        ki_st_bz, ki_st_n, ki_st_h,
        v_st_bz, v_st_h, v_st_d,
        o_st_bz, o_st_n, o_st_h,
        sm_scale, output_dtype_code, sparse_row_offsets_ptr,
        sparse_block_indices_ptr, block_mass_ptr, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("risa_sdpa_prequantized kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

// Nanobind wrapper: pure INT8 QK / U8-softmax / INT8-V attention kernel
void sage_attn(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale,
    nb::ndarray<nb::device::cuda> v_center,
    float sm_scale,
    int output_dtype_code,
    uintptr_t stream_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
        throw std::runtime_error("sage_attn: q, k, v, o must be 4D");
    }

    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error("sage_attn: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }

    constexpr int CTA_K = 64;
    const int64_t padded_k_length =
        ((static_cast<int64_t>(k.shape(2)) + CTA_K - 1) / CTA_K) * CTA_K;
    if (v.shape(3) < padded_k_length || v.shape(3) % CTA_K != 0) {
        throw std::runtime_error(
            "sage_attn: packed V sequence extent must cover K and be a multiple of 64");
    }

    const int64_t st_q_bz = static_cast<int64_t>(q.stride(0));
    const int64_t st_k_bz = static_cast<int64_t>(k.stride(0));
    const int64_t st_v_bz = static_cast<int64_t>(v.stride(0));
    const int64_t st_o_bz = static_cast<int64_t>(o.stride(0));
    if (st_q_bz > INT_MAX || st_k_bz > INT_MAX ||
        st_v_bz > INT_MAX || st_o_bz > INT_MAX) {
        throw std::overflow_error(
            "sage_attn: tensor strides exceed int32 range");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_risa_attention_kernel(
        q.data(), k.data(), v.data(), o.data(),
        q_scale.data(), k_scale.data(), v_scale.data(), v_center.data(),
        nullptr, 0, 0, 0, 0, -1,
        CTA_K,
        static_cast<int>(q.shape(0)),
        static_cast<int>(q.shape(2)),
        static_cast<int>(k.shape(2)),
        static_cast<int>(q.shape(1)),
        static_cast<int>(k.shape(1)),
        static_cast<int>(q.shape(3)),
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(2), o.stride(1),
        sm_scale, output_dtype_code, nullptr, nullptr, nullptr, stream);
}

// Fused RISA SDPA: Q/K quantization, V quantization, and attention in one call.
// All scratch buffers are pre-allocated by the caller (Python frontend).
void risa_sdpa(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> v_scale,
    nb::ndarray<nb::device::cuda> v_center,
    float sm_scale,
    int input_dtype_code,
    int output_dtype_code,
    uintptr_t stream_ptr,
    uintptr_t anchor_indices_ptr,
    std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt,
    int cta_k = 0)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
        throw std::runtime_error("risa_sdpa: q, k, v, o must be 4D [B,H,L,D]");
    }

    const int B = static_cast<int>(q.shape(0));
    const int H_q = static_cast<int>(q.shape(1));
    const int Lq = static_cast<int>(q.shape(2));
    const int D = static_cast<int>(q.shape(3));
    const int H_kv = static_cast<int>(k.shape(1));
    const int Lk = static_cast<int>(k.shape(2));

    const void *mask_ptr = nullptr;
    int64_t mask_stride_b = 0;
    int64_t mask_stride_h = 0;
    int64_t mask_stride_q = 0;
    int64_t mask_stride_k = 0;
    int mask_dtype_code = -1;
    if (attn_mask.has_value()) {
        const auto &mask = attn_mask.value();
        if (mask.ndim() != 4 || mask.shape(0) != B || mask.shape(1) != H_q ||
            mask.shape(2) != Lq || mask.shape(3) != Lk) {
            throw std::runtime_error(
                "risa_sdpa: attention mask must be expanded to [B,H_q,Lq,Lk]");
        }
        if (mask.dtype().code == (uint8_t)nb::dlpack::dtype_code::Bool) {
            mask_dtype_code = 3;
        } else {
            mask_dtype_code = map_dtype_to_code(mask.dtype());
        }
        if (mask_dtype_code < 0 || mask_dtype_code > 3) {
            throw std::runtime_error(
                "risa_sdpa: attention mask must be bool, float16, bfloat16, or float32");
        }
        mask_ptr = mask.data();
        mask_stride_b = mask.stride(0);
        mask_stride_h = mask.stride(1);
        mask_stride_q = mask.stride(2);
        mask_stride_k = mask.stride(3);
    }

    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("risa_sdpa: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    }
    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error(
            "risa_sdpa: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }
    if (cta_k == 0) {
        cta_k = !attn_mask.has_value() && D >= 128 && Lk > 1024
            ? 128
            : 64;
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("risa_sdpa: cta_k must be 64 or 128");
    }
    if (cta_k == 128 && (D == 64 || attn_mask.has_value())) {
        throw std::runtime_error(
            "risa_sdpa: cta_k 128 requires unmasked head_dim 128 or 256");
    }
    if (!anchor_indices_ptr) {
        throw std::runtime_error(
            "risa_sdpa: anchor_indices scratch is required");
    }
    constexpr int BLKQ = 128;
    const int WARPQ = D == 256 ? 16 : 32;
    const int BLKK = cta_k;
    const int WARPK = cta_k;
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_quant_qk_per_thread_int8(
        q.data(), q_int8.data(), q_scale.data(),
        k.data(), k_int8.data(), k_scale.data(),
        B, H_q, Lq, H_kv, Lk, D,
        BLKQ, WARPQ, BLKK, WARPK,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        input_dtype_code,
        reinterpret_cast<void *>(anchor_indices_ptr), stream);

    launch_quant_v_int8_kernel(
        v.data(), v_int8.data(), v_scale.data(), v_center.data(),
        B, H_kv, Lk, D, padded_Lk,
        v.stride(0), v.stride(1), v.stride(2),
        input_dtype_code, stream);

    // int64_t arithmetic to detect overflow before narrowing to int.
    const int64_t qi_st_bz64 = static_cast<int64_t>(H_q)  * Lq * D;
    const int64_t ki_st_bz64 = static_cast<int64_t>(H_kv) * Lk * D;
    const int64_t v_st_bz64  = static_cast<int64_t>(H_kv) * D * padded_Lk;

    if (qi_st_bz64 > INT_MAX || ki_st_bz64 > INT_MAX || v_st_bz64 > INT_MAX) {
        throw std::overflow_error(
            "risa_sdpa: tensor strides exceed int32 range; reduce batch/seq/head dimensions");
    }

    const int qi_st_h = Lq * D, qi_st_n = D, qi_st_bz = static_cast<int>(qi_st_bz64);
    const int ki_st_h = Lk * D, ki_st_n = D, ki_st_bz = static_cast<int>(ki_st_bz64);
    const int o_st_h  = Lq * D, o_st_n  = D, o_st_bz  = static_cast<int>(qi_st_bz64);
    // v_int8 is [B*H_kv*D, padded_Lk] (2D from quant kernel).
    // Attention expects V as [B, H, D, padded_N].
    const int v_st_d  = padded_Lk;
    const int v_st_h  = D * padded_Lk;
    const int v_st_bz = static_cast<int>(v_st_bz64);

    launch_risa_attention_kernel(
        q_int8.data(), k_int8.data(), v_int8.data(), o.data(),
        q_scale.data(), k_scale.data(), v_scale.data(), v_center.data(),
        mask_ptr, mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
        mask_dtype_code, cta_k,
        B, Lq, Lk, H_q, H_kv, D,
        qi_st_bz, qi_st_n, qi_st_h,
        ki_st_bz, ki_st_n, ki_st_h,
        v_st_bz, v_st_h, v_st_d,
        o_st_bz, o_st_n, o_st_h,
        sm_scale, output_dtype_code, nullptr, nullptr, nullptr, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("risa_sdpa kernel launch failed: ") + cudaGetErrorString(err));
    }
}

NB_MODULE(_C, m) {
    m.doc() = "RISA Attention CUDA kernels";
    m.def("risa_sdpa_quantize", &risa_sdpa_quantize,
          nb::arg("q"), nb::arg("k"), nb::arg("v"),
          nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"),
          nb::arg("k_scale"), nb::arg("v_int8"), nb::arg("v_scale"),
          nb::arg("v_center"),
          nb::arg("cta_k"), nb::arg("input_dtype_code"),
          nb::arg("stream_ptr"), nb::arg("anchor_indices_ptr"));
    m.def("risa_sdpa_prequantized", &risa_sdpa_prequantized,
          nb::arg("q_int8"), nb::arg("k_int8"), nb::arg("v_int8"),
          nb::arg("o"), nb::arg("q_scale"), nb::arg("k_scale"),
          nb::arg("v_scale"), nb::arg("v_center"), nb::arg("cta_k"),
          nb::arg("sm_scale"),
          nb::arg("output_dtype_code"), nb::arg("stream_ptr"),
          nb::arg("attn_mask") = nb::none(),
          nb::arg("sparse_row_offsets") = nb::none(),
          nb::arg("sparse_block_indices") = nb::none(),
          nb::arg("block_mass") = nb::none());
    m.def("risa_sdpa", &risa_sdpa,
          nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("o"),
          nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"),
          nb::arg("k_scale"), nb::arg("v_int8"), nb::arg("v_scale"),
          nb::arg("v_center"),
          nb::arg("sm_scale"), nb::arg("input_dtype_code"),
          nb::arg("output_dtype_code"), nb::arg("stream_ptr"),
          nb::arg("anchor_indices_ptr"), nb::arg("attn_mask") = nb::none(),
          nb::arg("cta_k") = 0);
}
