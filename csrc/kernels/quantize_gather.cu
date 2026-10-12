// SPDX-License-Identifier: Apache-2.0
// Compile the same native quantizers with virtual gather loads. The
// ordinary translation units compile those loads to identity operations.
#define RISA_FUSED_GATHER 1
#define launch_quantize_qk launch_quantize_qk_gather
#define launch_quantize_v launch_quantize_v_gather
#include "quantize_qk_impl.cuh"
#include "quantize_v_impl.cuh"
