"""Normalization, signed scales, and architecture-specific dispatch regressions."""

import pytest
import torch

from risa_attention import (
    int8_attention,
    int8_attention_from_prequantized,
    int8_attention_is_available,
    prequantize_int8_attention,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not int8_attention_is_available(), reason="requires the rebuilt CUDA extension"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("key_length", [512, 1025])
@pytest.mark.parametrize("scale", [None, 0.0, -0.125])
@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_constant_values_preserve_probability_mass(dtype, head_dim, key_length, scale, mode):
    torch.manual_seed(41)
    q = torch.randn(1, 4, 129, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, key_length, head_dim, device="cuda", dtype=dtype)
    # Each channel is constant across keys, so attention must preserve it.
    values = torch.linspace(-3, 5, head_dim, device="cuda", dtype=dtype)
    v = values.expand_as(k).contiguous()
    output = int8_attention(q, k, v, scale=scale, v_quantization=mode)
    torch.testing.assert_close(output, values.expand_as(output), atol=0, rtol=0)


@pytest.mark.parametrize("key_length", [512, 1025])
@pytest.mark.parametrize("scale", [0.0, -0.125])
@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_signed_scale_tail_masks_and_lse(key_length, scale, mode):
    torch.manual_seed(43)
    q = torch.randn(1, 4, 129, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, key_length, 128, device="cuda", dtype=q.dtype)
    v = torch.full_like(k, 3)
    mask = torch.ones(129, key_length, device="cuda", dtype=torch.bool)
    mask[0] = False
    mask[1:, 7::9] = False
    output, lse = int8_attention(
        q,
        k,
        v,
        scale=scale,
        attn_mask=mask,
        is_causal=True,
        v_quantization=mode,
        return_lse=True,
    )
    torch.testing.assert_close(
        output[:, :, 1:], torch.full_like(output[:, :, 1:], 3), atol=0, rtol=0
    )
    assert torch.count_nonzero(output[:, :, 0]) == 0
    assert torch.isneginf(lse[:, :, 0]).all()
    assert torch.isfinite(lse[:, :, 1:]).all()
    if scale == 0:
        visible = mask & (
            torch.arange(key_length, device="cuda")[None, :]
            <= torch.arange(129, device="cuda")[:, None]
        )
        expected = visible.sum(-1).float().log().expand_as(lse)
        torch.testing.assert_close(lse, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("scale", [None, 0.0, -0.125])
@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_tma_matches_general_output_layout(scale, mode):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 TMA specialization")
    torch.manual_seed(47)
    q = torch.randn(1, 4, 8193, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 8193, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k) + 3
    packed = prequantize_int8_attention(q, k, v, scale=scale, v_quantization=mode)
    tma_output = int8_attention_from_prequantized(packed)
    general_output = int8_attention_from_prequantized(packed, output_layout="nhd")
    if scale is None or scale >= 0:
        torch.testing.assert_close(tma_output, general_output, atol=0, rtol=0)
    else:
        # Two valid FP32 online rescaling orders can straddle a BF16 midpoint.
        upward = torch.full_like(general_output, torch.inf)
        step = (torch.nextafter(general_output, upward) - general_output).abs().float()
        assert ((tma_output.float() - general_output.float()).abs() <= step).all()


@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_nondefault_stream_and_cuda_graph(mode):
    q = torch.randn(1, 4, 8193, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 2, 8193, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        packed = prequantize_int8_attention(q, k, v, v_quantization=mode)
        expected = int8_attention_from_prequantized(packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = int8_attention_from_prequantized(packed)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["hnd", "token_major", "interleaved"])
def test_large_value_schedule_preserves_codes_and_scales(dtype, layout):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 value quantization schedule")
    heads, length, channels = 56, 14850, 128
    q = torch.randn(1, heads, 17, channels, device="cuda", dtype=dtype)
    if layout == "hnd":
        k, v = [
            torch.randn(1, heads, length, channels, device="cuda", dtype=dtype) for _ in range(2)
        ]
    elif layout == "token_major":
        k, v = [
            torch.randn(1, length, heads, channels, device="cuda", dtype=dtype).transpose(1, 2)
            for _ in range(2)
        ]
    else:
        combined = torch.randn(1, length, 3, heads, channels, device="cuda", dtype=dtype)
        k, v = (combined[:, :, index].transpose(1, 2) for index in (1, 2))
    wide = prequantize_int8_attention(q, k, v, v_quantization="absmax")
    # One head keeps the original 512-thread schedule while quantizing identical V.
    narrow = prequantize_int8_attention(q[:, :1], k[:, :1], v[:, :1], v_quantization="absmax")
    padded_length = wide.v.shape[-1]
    torch.testing.assert_close(
        wide.v.reshape(1, heads, channels, padded_length)[:, :1],
        narrow.v.reshape(1, 1, channels, padded_length),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        wide.v_scale.reshape(1, heads, channels)[:, :1],
        narrow.v_scale.reshape(1, 1, channels),
        atol=0,
        rtol=0,
    )


def test_residual_center_preserves_uniform_attention_mean():
    q = torch.zeros(1, 2, 129, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 512, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k) + 3
    v[..., ::53, :] += 12
    output = int8_attention(q, k, v, v_quantization="midpoint_affine")
    expected = v.float().mean(-2, keepdim=True).to(v.dtype).expand_as(output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("length,heads", [(4096, 16), (4422, 30), (4608, 32)])
@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_image_query_tile_preserves_quantization_groups(length, heads, mode):
    torch.manual_seed(59)
    q = torch.randn(1, heads, length, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 2, length, 128, device="cuda", dtype=q.dtype) for _ in range(2)]
    packed = prequantize_int8_attention(q, k, v, v_quantization=mode, return_lse=True)
    image_output = int8_attention_from_prequantized(packed)
    # LSE explicitly selects the generic 128-query tile and uses the same Q/K/V codes.
    general_output, lse = int8_attention_from_prequantized(packed, return_lse=True)
    torch.testing.assert_close(image_output, general_output, atol=0, rtol=0)
    assert torch.isfinite(lse).all()
