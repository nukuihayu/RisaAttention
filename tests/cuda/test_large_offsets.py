# SPDX-License-Identifier: Apache-2.0
"""Opt-in real >4 GiB addressing checks (about 17 GiB peak device memory)."""

import math
import os

import pytest
import torch

from risa_attention import _backend

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        os.environ.get("RISA_LARGE_GPU_TEST") != "1" or not _backend.is_available(),
        reason="set RISA_LARGE_GPU_TEST=1 on a GPU with 18 GiB free memory",
    ),
]


@pytest.fixture(scope="module")
def wide_buffers():
    if torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        pytest.skip("large-offset checks require 18 GiB free CUDA memory")
    length = (1 << 30) // 64 + 128
    heads, dim = 5, 64
    q = torch.zeros((1, heads, length, dim), dtype=torch.int8, device="cuda")
    k = torch.zeros((1, 1, 1, dim), dtype=torch.int8, device="cuda")
    v = torch.zeros((dim, 64), dtype=torch.int8, device="cuda")
    v[:, 0] = 100
    qs = torch.ones((1, heads, math.ceil(length / 128) * 32), device="cuda")
    ks = torch.ones((1, 1, 4), device="cuda")
    vs = torch.full((dim,), 0.2, device="cuda")
    assert (heads - 1) * q.stride(1) > (1 << 32) - 1
    center = torch.zeros_like(vs)
    return q, k, v, qs, ks, vs, center


@pytest.mark.parametrize("route", ["dense", "sparse", "capture"])
def test_real_wide_head_offsets(wide_buffers, route):
    q, k, v, qs, ks, vs, center = wide_buffers
    output = torch.empty(q.shape, dtype=torch.float16, device="cuda")
    rows = q.shape[1] * math.ceil(q.shape[2] / 128)
    kwargs = {}
    if route == "sparse":
        offsets = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
        indices = torch.zeros(rows, device="cuda", dtype=torch.int32)
        kwargs.update(
            sparse_row_offsets=_backend.wrap_for_dlpack(offsets),
            sparse_block_indices=_backend.wrap_for_dlpack(indices),
        )
    elif route == "capture":
        workspace = torch.empty(rows + q.shape[1] * q.shape[2] * 4, device="cuda")
        kwargs["block_mass"] = _backend.wrap_for_dlpack(workspace)
    _backend._C.risa_sdpa_prequantized(
        *(_backend.wrap_for_dlpack(t) for t in (q, k, v, output, qs, ks, vs, center)),
        64,
        0.125,
        1,
        torch.cuda.current_stream().cuda_stream,
        **kwargs,
    )
    # Single-key attention is exactly V for every query, including the head
    # whose base address is beyond uint32. Reductions check all output writes.
    assert output.amin().item() == 20.0
    assert output.amax().item() == 20.0
    if route == "capture":
        # Packed-U8 normalization uses a reciprocal whose rounding can leave
        # the reconstructed FP32 mass one ULP below the exact query count.
        torch.testing.assert_close(
            workspace[:rows], torch.full_like(workspace[:rows], 128.0), rtol=0.0, atol=2e-5
        )
