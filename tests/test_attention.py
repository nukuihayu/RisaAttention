# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch

import risa_attention as risa
import risa_attention.attention as risa_attention_module

_CUDA_READY = torch.cuda.is_available() and risa.int8_attention_is_available()
requires_int8_attention = pytest.mark.skipif(
    not _CUDA_READY,
    reason="requires the CUDA extension on an INT8-attention-capable GPU",
)


def _qkv(batch, q_heads, kv_heads, q_length, kv_length, head_dim, dtype=torch.bfloat16):
    q = torch.randn(
        batch, q_length, q_heads, head_dim, device="cuda", dtype=dtype
    ).transpose(1, 2)
    k = torch.randn(
        batch, kv_length, kv_heads, head_dim, device="cuda", dtype=dtype
    ).transpose(1, 2)
    v = torch.randn(
        batch, kv_length, kv_heads, head_dim, device="cuda", dtype=dtype
    ).transpose(1, 2)
    return q, k, v


def _nrmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt()
    return (error / magnitude).item()


def test_int8_attention_availability_is_bool():
    assert isinstance(risa.int8_attention_is_available(), bool)


def test_int8_attention_cta_k_selection():
    select = risa_attention_module._select_cta_k
    assert select(128, 1025, has_mask=False) == 128
    assert select(64, 1025, has_mask=False) == 64
    assert select(128, 1025, has_mask=True) == 64


def test_prequantized_attention_rejects_cpu_tensors():
    packed = risa_attention_module.PrequantizedInt8Attention(
        q=torch.empty(1, 1, 1, 64, dtype=torch.int8),
        k=torch.empty(1, 1, 1, 64, dtype=torch.int8),
        v=torch.empty(64, 64, dtype=torch.int8),
        q_scale=torch.empty(1, dtype=torch.float32),
        k_scale=torch.empty(1, dtype=torch.float32),
        v_scale=torch.empty(64, dtype=torch.float32),
        v_center=torch.empty(64, dtype=torch.float32),
        original_head_dim=64,
        input_dtype=torch.float16,
        attention_scale=0.125,
        cta_k=64,
        attn_mask=None,
    )

    with pytest.raises(ValueError, match="CUDA device"):
        risa.int8_attention_from_prequantized(packed)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 5), True),
        ((7, 0), False),
        ((8, 0), True),
        ((8, 7), True),
        ((11, 0), True),
    ],
)
def test_int8_attention_capability_dispatch(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(risa_attention_module._backend, "_C", object())
    assert risa_attention_module.is_available() is expected


@requires_int8_attention
def test_int8_attention_allocates_only_integer_8bit_scratch(monkeypatch):
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    allocated_dtypes = []
    original_empty = torch.empty

    def recording_empty(*args, **kwargs):
        allocated_dtypes.append(kwargs.get("dtype"))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", recording_empty)
    risa.int8_attention(q, k, v)

    assert allocated_dtypes.count(torch.int8) == 3
    assert allocated_dtypes.count(torch.int32) == 1
    assert torch.float8_e4m3fn not in allocated_dtypes


@requires_int8_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [1, 64, 96, 128, 192, 256])
def test_int8_attention_matches_sdpa(dtype, head_dim):
    q, k, v = _qkv(1, 8, 8, 257, 257, head_dim, dtype)
    assert not q.is_contiguous()

    actual = risa.int8_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@pytest.mark.parametrize(
    "option",
    [
        "softmax_dtype",
        "post_pv_dtype",
        "convrot",
        "stabilize_k",
        "smooth_k",
        "is_causal",
    ],
)
def test_int8_attention_rejects_removed_options(option):
    with pytest.raises(TypeError):
        risa.int8_attention(None, None, None, **{option: "input"})


@requires_int8_attention
def test_int8_attention_gqa_and_unequal_lengths():
    q, k, v = _qkv(1, 16, 4, 191, 257, 128)
    actual = risa.int8_attention(q, k, v, scale=0.07)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        scale=0.07,
    )

    assert actual.shape == (1, 16, 191, 128)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("masked", [False, True])
def test_int8_attention_batch_two_direct_and_prequantized(masked):
    q, k, v = _qkv(2, 8, 2, 193, 257, 128)
    mask = None
    if masked:
        mask = torch.zeros(2, 1, 1, 257, device="cuda", dtype=torch.bfloat16)
        mask[0, ..., 240:] = -torch.inf
        mask[1, ..., :17] = -torch.inf

    actual = risa.int8_attention(q, k, v, attn_mask=mask)
    quantized = risa.prequantize_int8_attention(q, k, v, attn_mask=mask)
    prequantized = risa.int8_attention_from_prequantized(quantized)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=mask,
    )

    assert torch.equal(prequantized, actual)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16])
def test_int8_attention_mask_gqa_broadcast_and_fully_masked_row(head_dim, mask_dtype):
    q, k, v = _qkv(1, 8, 2, 193, 257, head_dim)
    if mask_dtype == torch.bool:
        mask = torch.rand(1, 1, 193, 257, device="cuda") > 0.15
        mask[..., 7, :] = False
    else:
        mask = torch.zeros(1, 1, 193, 257, device="cuda", dtype=mask_dtype)
        mask[..., 220:] = -torch.inf
        mask[..., 7, :] = -torch.inf

    actual = risa.int8_attention(q, k, v, attn_mask=mask)
    baseline_mask = mask
    if mask.dtype != torch.bool and mask.dtype != q.dtype:
        baseline_mask = mask.to(q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=baseline_mask,
    )

    assert torch.count_nonzero(actual[..., 7, :]) == 0
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize(
    "mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32]
)
def test_int8_attention_key_broadcast_mask(mask_dtype):
    q, k, v = _qkv(1, 8, 2, 193, 257, 64)
    if mask_dtype == torch.bool:
        mask = torch.rand(1, 1, 1, 257, device="cuda") > 0.15
    else:
        mask = torch.linspace(-1, 1, 257, device="cuda", dtype=mask_dtype).reshape(
            1, 1, 1, 257
        )
        mask[..., 240:] = -torch.inf

    actual = risa.int8_attention(q, k, v, attn_mask=mask)
    baseline_mask = mask
    if mask.dtype != torch.bool and mask.dtype != q.dtype:
        baseline_mask = mask.to(q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=baseline_mask,
    )

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.bfloat16])
def test_int8_attention_fully_masked_key_broadcast_is_zero(mask_dtype):
    q, k, v = _qkv(1, 4, 4, 129, 97, 64)
    if mask_dtype == torch.bool:
        mask = torch.zeros(1, 1, 1, 97, dtype=torch.bool, device="cuda")
    else:
        mask = torch.full((1, 1, 1, 97), -torch.inf, dtype=mask_dtype, device="cuda")

    actual = risa.int8_attention(q, k, v, attn_mask=mask)

    assert torch.count_nonzero(actual) == 0


@requires_int8_attention
def test_int8_attention_stabilizes_large_common_key_component():
    torch.manual_seed(7)
    q, k, v = _qkv(1, 16, 16, 513, 513, 128)
    common_key = torch.randn(1, 16, 1, 128, device="cuda", dtype=torch.float32)
    common_key.mul_(40.0 / common_key.square().mean(-1, keepdim=True).sqrt())
    k.add_(common_key.to(k.dtype))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = risa.int8_attention(q, k, v)

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_midpoint_centers_biased_values():
    torch.manual_seed(19)
    q, k, v = _qkv(1, 8, 2, 257, 257, 128)
    v.add_(3.0)

    packed = risa.prequantize_int8_attention(q, k, v)
    actual = risa.int8_attention_from_prequantized(packed)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
    )

    assert packed.v_center.abs().mean() > 2.0
    assert _nrmse(actual, expected) < 0.01


@requires_int8_attention
def test_int8_attention_v_center_removes_channel_mean_residual():
    torch.manual_seed(29)
    _, _, v = _qkv(1, 8, 2, 257, 257, 128)
    v.add_(torch.linspace(-3.0, 3.0, 128, device="cuda").to(v.dtype))

    packed = risa.prequantize_int8_attention(v, v, v)
    source_mean = v.float().mean(dim=2).reshape(-1)
    code_mean = packed.v.float().sum(dim=1) / v.shape[2]
    reconstructed_mean = packed.v_center + packed.v_scale * code_mean

    torch.testing.assert_close(
        reconstructed_mean, source_mean, rtol=0.0, atol=2e-5
    )


@requires_int8_attention
def test_int8_attention_stabilization_is_deterministic():
    torch.manual_seed(11)
    q, k, v = _qkv(1, 8, 8, 257, 257, 128)

    first = risa.int8_attention(q, k, v)
    second = risa.int8_attention(q, k, v)

    assert torch.equal(first, second)


@requires_int8_attention
@pytest.mark.parametrize(
    "configuration",
    [
        {
            "q_length": 257,
            "kv_length": 257,
            "head_dim": 64,
            "dtype": torch.float16,
        },
        {"q_length": 193, "kv_length": 1281, "head_dim": 128},
        {
            "q_length": 193,
            "kv_length": 257,
            "head_dim": 96,
            "dtype": torch.float32,
        },
        {
            "q_length": 193,
            "kv_length": 1281,
            "head_dim": 256,
        },
        {"q_length": 257, "kv_length": 257, "head_dim": 256},
    ],
)
def test_prequantized_attention_is_bitwise_identical_to_fused(configuration):
    torch.manual_seed(123)
    q, k, v = _qkv(
        1,
        8,
        2,
        configuration["q_length"],
        configuration["kv_length"],
        configuration["head_dim"],
        configuration.get("dtype", torch.bfloat16),
    )

    expected = risa.int8_attention(q, k, v)
    quantized = risa.prequantize_int8_attention(q, k, v)
    actual = risa.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_sequence_major_output_is_bitwise_identical_and_flattens_as_view():
    q, k, v = _qkv(2, 8, 2, 257, 1281, 128)
    quantized = risa.prequantize_int8_attention(q, k, v)

    hnd = risa.int8_attention_from_prequantized(quantized)
    nhd = risa.int8_attention_from_prequantized(quantized, output_layout="nhd")
    flattened = nhd.transpose(1, 2).reshape(2, 257, -1)

    assert torch.equal(nhd, hnd)
    assert hnd.is_contiguous()
    assert not nhd.is_contiguous()
    assert flattened.data_ptr() == nhd.data_ptr()


@requires_int8_attention
def test_prequantized_attention_rejects_unknown_output_layout():
    q, k, v = _qkv(1, 2, 2, 64, 64, 64)
    quantized = risa.prequantize_int8_attention(q, k, v)

    with pytest.raises(ValueError, match="output_layout"):
        risa.int8_attention_from_prequantized(quantized, output_layout="blocked")


@requires_int8_attention
@pytest.mark.parametrize("head_dim", [96, 192])
def test_prequantized_sequence_major_output_handles_padded_head_dim(head_dim):
    q, k, v = _qkv(1, 4, 2, 193, 257, head_dim)
    quantized = risa.prequantize_int8_attention(q, k, v)

    hnd = risa.int8_attention_from_prequantized(quantized)
    nhd = risa.int8_attention_from_prequantized(quantized, output_layout="nhd")

    assert nhd.shape == hnd.shape == (1, 4, 193, head_dim)
    assert torch.equal(nhd, hnd)


@requires_int8_attention
def test_prequantized_masked_attention_is_bitwise_identical_to_fused():
    q, k, v = _qkv(1, 8, 2, 193, 257, 128)
    mask = torch.linspace(-1, 1, 257, device="cuda", dtype=torch.float32).reshape(
        1, 1, 1, 257
    )
    mask[..., 240:] = -torch.inf

    expected = risa.int8_attention(q, k, v, attn_mask=mask)
    quantized = risa.prequantize_int8_attention(
        q,
        k,
        v,
        attn_mask=mask,
    )
    actual = risa.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_attention_releases_float_inputs_before_execution():
    q, k, v = _qkv(1, 8, 2, 513, 769, 128)
    expected = risa.int8_attention(q, k, v)
    input_references = tuple(weakref.ref(tensor) for tensor in (q, k, v))

    quantized = risa.prequantize_int8_attention(q, k, v)
    del q, k, v
    gc.collect()
    assert all(reference() is None for reference in input_references)

    # Force allocator reuse on the same stream before consuming the packed
    # tensors. This catches a split implementation that only appears correct
    # while its asynchronous quantization inputs remain allocated.
    allocator_churn = torch.empty(
        64 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    allocator_churn.fill_(0xA5)
    actual = risa.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_int8_attention_torch_compile_fullgraph():
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    compiled = torch.compile(
        lambda q_, k_, v_: risa.int8_attention(q_, k_, v_),
        backend="eager",
        fullgraph=True,
    )
    actual = compiled(q, k, v)
    expected = risa.int8_attention(q, k, v)
    torch.testing.assert_close(actual, expected)


@requires_int8_attention
def test_int8_attention_cuda_graph():
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        risa.int8_attention(q, k, v)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = risa.int8_attention(q, k, v)
    graph.replay()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_rotation_handles_outliers():
    torch.manual_seed(1)
    q, k, v = _qkv(1, 8, 8, 513, 513, 128)
    q[..., 0].mul_(12)
    k[..., 0].mul_(12)
    q.mul_(q.float().square().mean(-1, keepdim=True).rsqrt().to(q.dtype))
    k.mul_(k.float().square().mean(-1, keepdim=True).rsqrt().to(k.dtype))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = risa.int8_attention(q, k, v)

    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_accepts_dlpack_normalized_batch_stride():
    """A size-one extent carries no address, so its stride must not be policed.

    PyTorch rewrites the stride of any size-one dimension to 1 on the way out
    through DLPack (ATen/DLConvertor.cpp, gh-83069), so a batch-one attention
    input arrives with stride 1 no matter what the caller built.
    """
    torch.manual_seed(0)
    length, heads, head_dim = 372, 8, 128
    packed = [
        torch.randn(length, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    reported = [t.transpose(0, 1).unsqueeze(0) for t in packed]
    normalized = [
        torch.as_strided(
            t, (1, heads, length, head_dim), (1, head_dim, heads * head_dim, 1)
        )
        for t in packed
    ]
    assert normalized[0].stride(0) == 1
    assert torch.equal(reported[0], normalized[0])

    expected = risa.int8_attention_from_prequantized(
        risa.prequantize_int8_attention(*reported)
    )
    actual = risa.int8_attention_from_prequantized(
        risa.prequantize_int8_attention(*normalized)
    )

    assert torch.equal(actual, expected)


def _structured_sparse_qkv(length=512):
    torch.manual_seed(41)
    q, k, v = _qkv(1, 4, 2, length, length, 128)
    q.mul_(0.3)
    k.mul_(0.3)
    prototypes = torch.randn(2, 4, 128, device="cuda", dtype=torch.float32)
    prototypes.mul_(8.0 / prototypes.norm(dim=-1, keepdim=True))
    for head in range(4):
        for block in range(length // 128):
            q[:, head, block * 128 : (block + 1) * 128].add_(
                prototypes[head // 2, block % 4].to(q.dtype)
            )
    for head in range(2):
        for block in range(length // 128):
            k[:, head, block * 128 : (block + 1) * 128].add_(
                prototypes[head, block % 4].to(k.dtype)
            )
    return q, k, v


@requires_int8_attention
def test_sparse_full_support_is_bitwise_dense_int8():
    q, k, v = _qkv(1, 4, 2, 512, 512, 128)
    pattern = risa.build_retained_mass_pattern(q, k, theta=1.0)
    packed = risa.prequantize_int8_attention(q, k, v)

    dense = risa.int8_attention_from_prequantized(packed)
    sparse = risa.int8_attention_from_prequantized(packed, sparse_pattern=pattern)

    assert pattern.coverage == 1.0
    assert torch.equal(sparse, dense)


@requires_int8_attention
def test_sparse_sparse_support_matches_exact_sparse_sdpa():
    q, k, v = _structured_sparse_qkv()
    pattern = risa.build_retained_mass_pattern(q, k, theta=0.99)
    actual = risa.sparse_int8_attention(q, k, v, pattern)
    actual_nhd = risa.sparse_int8_attention(q, k, v, pattern, output_layout="nhd")

    mask = torch.zeros(1, 4, 512, 512, dtype=torch.bool, device="cuda")
    for head in range(4):
        for query_block in range(4):
            row = head * 4 + query_block
            start = int(pattern.row_offsets[row])
            end = int(pattern.row_offsets[row + 1])
            for key_block in pattern.block_indices[start:end].tolist():
                mask[
                    0,
                    head,
                    query_block * 128 : (query_block + 1) * 128,
                    key_block * pattern.key_block_size : (key_block + 1)
                    * pattern.key_block_size,
                ] = True
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=mask,
    )

    assert pattern.coverage < 0.75
    assert pattern.measured_retained_mass >= 0.99
    assert torch.equal(actual_nhd, actual)
    assert _nrmse(actual, expected) < 0.03
    assert risa.measure_pattern_recall(q, k, pattern) == pytest.approx(
        pattern.measured_retained_mass, abs=2e-6
    )


@requires_int8_attention
def test_sparse_rejects_pattern_for_different_shape():
    q, k, v = _structured_sparse_qkv()
    pattern = risa.build_retained_mass_pattern(q, k)
    q_short = q[:, :, :-1]
    k_short = k[:, :, :-1]
    v_short = v[:, :, :-1]

    with pytest.raises(ValueError, match="metadata does not match"):
        risa.sparse_int8_attention(q_short, k_short, v_short, pattern)


@requires_int8_attention
def test_sparse_sparse_support_handles_partial_tail_tile():
    q, k, v = _qkv(1, 4, 2, 1281, 1281, 128)
    tail_prototype = torch.randn(128, device="cuda", dtype=torch.float32)
    tail_prototype.mul_(12.0 / tail_prototype.norm())
    q[:, :, -1].copy_(tail_prototype.to(q.dtype))
    k[:, :, -1].copy_(tail_prototype.to(k.dtype))
    pattern = risa.build_retained_mass_pattern(q, k, theta=0.5)
    assert pattern.key_block_size == 128
    assert torch.any(pattern.block_indices == 10)

    actual = risa.sparse_int8_attention(q, k, v, pattern)
    mask = torch.zeros(1, 4, 1281, 1281, dtype=torch.bool, device="cuda")
    for head in range(4):
        for query_block in range(11):
            row = head * 11 + query_block
            start = int(pattern.row_offsets[row])
            end = int(pattern.row_offsets[row + 1])
            for key_block in pattern.block_indices[start:end].tolist():
                mask[
                    0,
                    head,
                    query_block * 128 : min((query_block + 1) * 128, 1281),
                    key_block * 128 : min((key_block + 1) * 128, 1281),
                ] = True
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=mask,
    )
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("length,head_dim", [(257, 64), (512, 128), (1025, 256)])
def test_fused_frozen_support_construction_preserves_dense_output(length, head_dim):
    q, k, v = _qkv(1, 4, 2, length, length, head_dim, dtype=torch.float16)

    expected = risa.int8_attention(q, k, v)
    actual, pattern = risa.construct_sparse_int8_attention(q, k, v, theta=0.9)

    assert torch.equal(actual, expected)
    assert pattern.row_offsets.is_cuda
    assert pattern.block_indices.is_cuda
    assert pattern.measured_retained_mass >= 0.9
    assert risa.measure_pattern_recall(q, k, pattern) >= 0.89


@requires_int8_attention
def test_prequantized_frozen_support_construction_supports_sequence_major_output():
    q, k, v = _qkv(1, 4, 2, 257, 257, 128, dtype=torch.bfloat16)
    quantized = risa.prequantize_int8_attention(q, k, v)

    output, pattern = risa.construct_sparse_int8_attention_from_prequantized(
        quantized,
        theta=0.99,
        output_layout="nhd",
    )
    flattened = output.transpose(1, 2).reshape(1, 257, -1)

    assert output.shape == q.shape
    assert flattened.data_ptr() == output.data_ptr()
    assert pattern.retained_mass_target == 0.99
    assert pattern.q_length == pattern.kv_length == 257


@requires_int8_attention
def test_fused_frozen_support_construction_structured_recall():
    q, k, v = _structured_sparse_qkv(512)

    _, pattern = risa.construct_sparse_int8_attention(q, k, v, theta=0.99)
    exact_recall = risa.measure_pattern_recall(q, k, pattern)

    assert pattern.coverage < 0.75
    assert pattern.measured_retained_mass >= 0.99
    assert exact_recall >= 0.989


@requires_int8_attention
def test_sparse_explicit_theta_matches_default():
    q, k, v = _structured_sparse_qkv(512)

    exact_theta = risa.build_retained_mass_pattern(q, k, theta=0.99)
    exact_default = risa.build_retained_mass_pattern(q, k)
    output_theta, fused_theta = risa.construct_sparse_int8_attention(
        q, k, v, theta=0.99
    )
    output_default, fused_default = risa.construct_sparse_int8_attention(q, k, v)

    assert exact_theta.retained_mass_target == 0.99
    assert torch.equal(exact_theta.row_offsets, exact_default.row_offsets)
    assert torch.equal(exact_theta.block_indices, exact_default.block_indices)
    assert torch.equal(output_theta, output_default)
    assert fused_theta.retained_mass_target == 0.99
    assert torch.equal(fused_theta.row_offsets, fused_default.row_offsets)
    assert torch.equal(fused_theta.block_indices, fused_default.block_indices)


@pytest.mark.parametrize("theta", [0.0, 1.01, float("nan")])
def test_sparse_rejects_invalid_theta(theta):
    with pytest.raises(ValueError, match="must be finite and in"):
        risa.build_retained_mass_pattern(None, None, theta=theta)


def test_sparse_rejects_removed_retained_mass_keyword():
    with pytest.raises(TypeError, match="retained_mass"):
        risa.build_retained_mass_pattern(None, None, retained_mass=0.9)
    with pytest.raises(TypeError, match="retained_mass"):
        risa.construct_sparse_int8_attention(None, None, None, retained_mass=0.9)
