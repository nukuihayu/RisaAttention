// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernels/quantization_input.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {
void launch_correct_lse(const void* q, const void* k, const int* anchors, float* out,
                        int batch_size, int Hq, int Hk, int Nq, int head_dim, int64_t qb,
                        int64_t qh, int64_t qn, int64_t kb, int64_t kh, int64_t kn, float scale,
                        int dtype, cudaStream_t stream);
void launch_quantize_qk(const void* q, void* q_int8, void* q_scale, const void* k, void* k_int8,
                        void* k_scale, int batch_size, int query_heads, int query_length,
                        int key_value_heads, int key_length, int head_dim, int BLKQ, int WARPQ,
                        int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
                        int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
                        int64_t k_stride_n, int input_dtype_code, void* anchor_indices,
                        cudaStream_t stream, risa::QuantInput qt = {}, risa::QuantInput kt = {});

void launch_quantize_qk_gather(const void* q, void* q_int8, void* q_scale, const void* k,
                               void* k_int8, void* k_scale, int batch_size, int query_heads,
                               int query_length, int key_value_heads, int key_length, int head_dim,
                               int BLKQ, int WARPQ, int BLKK, int WARPK, int64_t q_stride_b,
                               int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b,
                               int64_t k_stride_h, int64_t k_stride_n, int input_dtype_code,
                               void* anchor_indices, cudaStream_t stream, risa::QuantInput qt,
                               risa::QuantInput kt);

void launch_quantize_v(const void* v, void* out, void* scale, void* center, int batch_size,
                       int num_heads, int sequence_length, int head_dim, int padded_length,
                       int64_t batch_stride, int64_t head_stride, int64_t token_stride,
                       int input_dtype_code, bool use_midpoint_affine, cudaStream_t stream,
                       risa::QuantInput transform = {});

void launch_quantize_v_gather(const void* v, void* out, void* scale, void* center, int batch_size,
                              int num_heads, int sequence_length, int head_dim, int padded_length,
                              int64_t batch_stride, int64_t head_stride, int64_t token_stride,
                              int input_dtype_code, bool use_midpoint_affine, cudaStream_t stream,
                              risa::QuantInput transform);

void launch_prepare_key_mask_tiles(const void* mask, int32_t* tiles, int batch_size,
                                   int query_heads, int key_length, int tile_size, int64_t stride_b,
                                   int64_t stride_h, int64_t stride_k, int dtype_code,
                                   cudaStream_t stream);
void launch_attention(const void* q, const void* k, const void* v, void* o, const void* q_scale,
                      const void* k_scale, const void* v_scale, const void* v_center,
                      const void* mask, int64_t mask_stride_b, int64_t mask_stride_h,
                      int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int cta_k,
                      int batch_size, int query_length, int key_length, int query_heads,
                      int key_value_heads, int head_dim, int64_t q_st_bz, int64_t q_st_n,
                      int64_t q_st_h, int64_t k_st_bz, int64_t k_st_n, int64_t k_st_h,
                      int64_t v_st_bz, int64_t v_st_h, int64_t v_st_d, int64_t o_st_bz,
                      int64_t o_st_n, int64_t o_st_h, float sm_scale, int output_dtype_code,
                      const void* sparse_row_offsets, const void* sparse_block_indices,
                      void* block_mass, bool is_causal, void* lse, cudaStream_t stream,
                      const int32_t* key_mask_tiles = nullptr);
void launch_select_retained_mass(const float* sorted_mass, const int64_t* descending_indices,
                                 int32_t* counts, uint8_t* support, double* row_totals,
                                 int32_t* row_offsets, int32_t* block_indices, double* summary,
                                 void* scan_workspace, size_t scan_workspace_bytes, int row_count,
                                 int key_blocks, double theta, cudaStream_t stream);
size_t retained_mass_selector_workspace_size(int row_count);
}
