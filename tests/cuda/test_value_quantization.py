"""Exact V quantization ABI across vector stores, layouts and padded tails."""

import pytest
import torch

from risa_attention import (
    int8_attention_is_available,
    prepare_kv_permutation,
    prequantize_int8_attention,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not int8_attention_is_available(), reason="requires CUDA extension"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["hnd", "nhd", "interleaved"])
@pytest.mark.parametrize("length", [1025, 2048, 2049, 4097])
def test_absmax_values_preserve_packing_and_tail(dtype, layout, length):
    # Exact multiples of powers of two make the expected INT8 codes and scales
    # independent of floating-point rounding in the reference calculation.
    batch, heads, channels = 2, 3, 128
    tokens = torch.arange(length, device="cuda")[None, None, :, None]
    head_offsets = torch.arange(batch * heads, device="cuda").reshape(batch, heads, 1, 1)
    codes = ((tokens + head_offsets) % 255 - 127).expand(batch, heads, length, channels)
    scales = 2.0 ** ((torch.arange(channels, device="cuda") % 4).float() - 2)
    v = (codes * scales).to(dtype)
    if layout == "nhd":
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "interleaved":
        storage = torch.empty(batch, length, 3, heads, channels, device="cuda", dtype=dtype)
        v = storage[:, :, 1].transpose(1, 2).copy_(v)
    q = torch.zeros(batch, heads * 2, 3, channels, device="cuda", dtype=dtype)
    k = torch.zeros_like(v)
    packed = prequantize_int8_attention(q, k, v, v_quantization="absmax")
    padded_length = packed.v.shape[-1]
    expected = torch.zeros(batch, heads, channels, padded_length, device="cuda", dtype=torch.int8)
    expected[..., :length] = codes.transpose(2, 3).to(torch.int8)
    # Public packed-V ABI, expressed as a fixed table rather than the kernel's
    # bit-index calculation. Padding participates in the same permutation.
    order = torch.tensor([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15], device="cuda")
    expected = expected.reshape(batch, heads, channels, -1, 16)[..., order]
    torch.testing.assert_close(packed.v.reshape_as(expected), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        packed.v_scale.reshape(batch, heads, channels),
        scales.expand(batch, heads, channels),
        atol=0,
        rtol=0,
    )
    assert torch.count_nonzero(packed.v_center) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["hnd", "nhd", "interleaved"])
def test_packed_values_match_materialized_permutation(dtype, layout):
    q = torch.randn(1, 4, 17, 128, device="cuda", dtype=dtype)
    k, v = [torch.randn(1, 2, 2049, 128, device="cuda", dtype=dtype) for _ in range(2)]
    if layout == "nhd":
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "interleaved":
        storage = torch.empty(1, 2049, 3, 2, 128, device="cuda", dtype=dtype)
        v = storage[:, :, 1].transpose(1, 2).copy_(v)
    indices = torch.stack([torch.randperm(2049, device="cuda") for _ in range(2)])[None]
    permutation = prepare_kv_permutation(indices)
    fused = prequantize_int8_attention(q, k, v, kv_permutation=permutation, fuse_gather=True)
    eager = prequantize_int8_attention(q, k, v, kv_permutation=permutation, fuse_gather=False)
    for attribute in ("v", "v_scale", "v_center"):
        torch.testing.assert_close(
            getattr(fused, attribute), getattr(eager, attribute), atol=0, rtol=0
        )


def test_packed_value_quantization_on_nondefault_stream_and_graph():
    q = torch.randn(1, 4, 17, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 2, 4097, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = prequantize_int8_attention(q, k, v)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = prequantize_int8_attention(q, k, v)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    for attribute in ("v", "v_scale", "v_center"):
        torch.testing.assert_close(
            getattr(actual, attribute), getattr(expected, attribute), atol=0, rtol=0
        )
