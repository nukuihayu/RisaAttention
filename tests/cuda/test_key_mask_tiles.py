# SPDX-License-Identifier: Apache-2.0
"""Compare broadcast-mask skipping with the same packed generic-mask path."""

from dataclasses import replace

import pytest
import torch

import risa_attention as risa

pytestmark = pytest.mark.skipif(
    not risa.int8_attention_is_available(), reason="requires the rebuilt CUDA extension"
)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_key_mask_tile_skipping_matches_generic_mask(head_dim, mask_dtype, mode):
    torch.manual_seed(74)
    q = torch.randn(2, 4, 257, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 1153, head_dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k) + 3
    storage = torch.zeros(2, 4, 1, 2306, device="cuda", dtype=mask_dtype)
    mask = storage[..., ::2]
    visible = torch.ones(2, 4, 1, 1153, device="cuda", dtype=torch.bool)
    visible[..., :128] = False
    visible[..., 384:768] = False
    visible[0, 1] = False
    visible[1, 2, :, -1] = False
    visible[..., 897:913] = False
    if mask_dtype == torch.bool:
        mask.copy_(visible)
    else:
        mask.copy_(torch.linspace(-1, 2, 1153, device="cuda"))
        mask.masked_fill_(~visible, -torch.inf)
    packed = risa.prequantize_int8_attention(
        q, k, v, attn_mask=mask, v_quantization=mode, return_lse=True
    )
    generic = replace(packed, attn_mask=packed.attn_mask.clone())
    actual, lse = risa.int8_attention_from_prequantized(packed, return_lse=True)
    expected, expected_lse = risa.int8_attention_from_prequantized(generic, return_lse=True)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    # Broadcast and query-dependent masks already differ slightly in FP32
    # score rounding before U8 packing, even without tile skipping.
    torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-6)
    assert torch.count_nonzero(actual[0, 1]) == 0
    assert torch.isneginf(lse[0, 1]).all()
    torch.testing.assert_close(
        risa.int8_attention(q, k, v, attn_mask=mask, v_quantization=mode),
        actual,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_key_mask_skipping_stream_and_graph(mode):
    q = torch.randn(1, 4, 513, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 2049, 128, device="cuda", dtype=q.dtype)
    v = torch.full_like(k, 2.75)
    mask = torch.ones(1, 1, 1, 2049, device="cuda", dtype=torch.bool)
    mask[..., 128:1536] = False
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = risa.int8_attention(q, k, v, attn_mask=mask, v_quantization=mode)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = risa.int8_attention(q, k, v, attn_mask=mask, v_quantization=mode)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(output, v[..., :513, :].repeat_interleave(2, dim=1), atol=0, rtol=0)
    torch.testing.assert_close(captured, output, atol=0, rtol=0)


@pytest.mark.parametrize("inference", [False, True])
def test_key_mask_metadata_and_mask_mutation(inference):
    q = torch.randn(1, 4, 257, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 1153, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    mask = torch.ones(1, 1, 1, 1153, device="cuda", dtype=torch.bool)
    if inference:
        with torch.inference_mode():
            mask = mask.clone()
    with torch.inference_mode():
        mask[..., :512] = False
    packed = risa.prequantize_int8_attention(q, k, v, attn_mask=mask)
    from risa_attention._key_mask import prepare_key_mask_tiles

    tiles = prepare_key_mask_tiles(
        packed.attn_mask, packed.cta_k, 257, packed.attention_scale, False
    )
    expected = torch.tensor([4, 5, 6, 7, 8, 9], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(tiles[..., :6], expected.expand(1, 4, 6))
    assert (tiles[..., -1] == 6).all()
    with torch.inference_mode():
        mask.fill_(True)
    actual = risa.int8_attention_from_prequantized(packed)
    generic = risa.int8_attention_from_prequantized(
        replace(packed, attn_mask=packed.attn_mask.clone())
    )
    torch.testing.assert_close(actual, generic, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["absmax", "midpoint_affine"])
def test_prequantized_mask_mutation_during_graph_replay(mode):
    with torch.inference_mode():
        q = torch.randn(1, 4, 257, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 1153, 128, device="cuda", dtype=q.dtype)
        v = torch.full_like(k, 2.75)
        mask = torch.ones(1, 1, 1, 1153, device="cuda", dtype=torch.bool)
        mask[..., :768] = False
        packed = risa.prequantize_int8_attention(q, k, v, attn_mask=mask, v_quantization=mode)
        for _ in range(3):
            risa.int8_attention_from_prequantized(packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = risa.int8_attention_from_prequantized(packed)
        mask.fill_(False)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(captured) == 0
