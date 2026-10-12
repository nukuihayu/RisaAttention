"""RISA attention runtime tests."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import pytest
import torch

from risa_attention._validation import apply_score_mask
from tests.support.risa_backend import risa

pytestmark = pytest.mark.cuda


@unittest.skipUnless(risa.is_available(), "requires rebuilt RISA CUDA extension and SM80+ GPU")
class RisaCudaTests(unittest.TestCase):
    def test_capture_reuses_tile_max_without_changing_output_bits(self):
        # Known packed scores test tile maxima independently of the Q/K
        # quantizer. Descending tiles catch accidental reuse of the running
        # maximum; short/partial sequences cover all pipeline stages.
        for mode in ("midpoint_affine", "absmax"):
            for dim in (64, 128, 256):
                for length in (63, 65, 193, 1153):
                    with self.subTest(mode=mode, dim=dim, length=length):
                        q = torch.zeros(1, 2, 137, dim, device="cuda", dtype=torch.float16)
                        k = torch.zeros(1, 1, length, dim, device="cuda", dtype=torch.float16)
                        packed = risa.prequantize_int8_attention(
                            q, k, torch.randn_like(k) + 2, scale=1 / 128, v_quantization=mode
                        )
                        packed.q.fill_(1)
                        packed.q_scale.fill_(1)
                        packed.k_scale.fill_(1)
                        blocks = (length + packed.cta_k - 1) // packed.cta_k
                        levels = torch.tensor(
                            [3 if i % 2 == 0 else -2 for i in range(blocks)],
                            dtype=torch.int8,
                            device="cuda",
                        )
                        for i in range(blocks):
                            packed.k[:, :, i * packed.cta_k : (i + 1) * packed.cta_k].fill_(
                                3 if i % 2 == 0 else -2
                            )
                        batch, heads, queries, _ = packed.q.shape
                        mass_elements = batch * heads * ((queries + 127) // 128) * blocks
                        rows = batch * heads * queries
                        workspace = torch.empty(
                            mass_elements + rows * (blocks * 2 + 2),
                            device="cuda",
                            dtype=torch.float32,
                        )
                        plain = risa.int8_attention_from_prequantized(packed)
                        captured = risa.int8_attention_from_prequantized(
                            packed, _block_mass=workspace
                        )
                        self.assertTrue(
                            torch.equal(plain.view(torch.int16), captured.view(torch.int16))
                        )
                        if blocks > 1:
                            full_stats = workspace[
                                mass_elements : mass_elements + rows * (blocks - 1) * 2
                            ]
                            maxima = full_stats.reshape(rows, blocks - 1, 2)[..., 1]
                            scale = (
                                torch.tensor(
                                    1.44269504088896340736, device="cuda", dtype=torch.float32
                                )
                                / 128
                            )
                            offset = torch.tensor(7.9943534, device="cuda", dtype=torch.float32)
                            # Double arithmetic followed by one float rounding
                            # reproduces the original float FMA for these small
                            # integer scores exactly.
                            expected = (
                                levels[:-1].double() * dim * scale.double() - offset.double()
                            ).float()
                            self.assertTrue(torch.equal(maxima, expected.expand_as(maxima)))

                        # Invalid-row cleanup is folded into V dequantization.
                        # A nonzero V center must not turn an empty row nonzero.
                        hidden = torch.zeros(
                            batch, heads, queries, length, device="cuda", dtype=torch.bool
                        )
                        masked = replace(packed, attn_mask=hidden)
                        empty = risa.int8_attention_from_prequantized(masked)
                        self.assertTrue(
                            torch.equal(
                                empty.view(torch.int16), torch.zeros_like(empty).view(torch.int16)
                            )
                        )

    def test_dense_sparse_causal_masks_and_v_modes(self):
        torch.manual_seed(17)
        for mode in ("midpoint_affine", "absmax"):
            for dim in (64, 128, 256):
                # Partial query/key tiles and GQA, with a nonzero V center.
                q = torch.randn(1, 2, 137, dim, device="cuda", dtype=torch.float16)
                k = torch.randn(1, 1, 263, dim, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k) + 2
                key_bias = torch.linspace(-2, 0, 263, device="cuda")[None, None, None, :]
                row_mask = torch.ones(137, 263, device="cuda", dtype=torch.bool)
                row_mask[0] = False
                for causal in (False, True):
                    for mask in (None, key_bias, row_mask):
                        with self.subTest(mode=mode, dim=dim, causal=causal, mask=mask is not None):
                            options = dict(v_quantization=mode, is_causal=causal, attn_mask=mask)
                            dense = risa.int8_attention(q, k, v, **options)
                            built, pattern = risa.construct_sparse_int8_attention(
                                q, k, v, theta=1.0, **options
                            )
                            sparse = risa.sparse_int8_attention(q, k, v, pattern, **options)
                            packed = risa.prequantize_int8_attention(q, k, v, **options)
                            split = risa.int8_attention_from_prequantized(
                                packed, sparse_pattern=pattern, output_layout="nhd"
                            )
                            scores = (
                                q.float()
                                @ k.float().repeat_interleave(2, 1).transpose(-1, -2)
                                / dim**0.5
                            )
                            expanded = (
                                None if mask is None else torch.broadcast_to(mask, scores.shape)
                            )
                            scores = apply_score_mask(scores, expanded, causal, 0)
                            reference = scores.softmax(-1).nan_to_num(
                                nan=0
                            ) @ v.float().repeat_interleave(2, 1)
                            torch.testing.assert_close(
                                dense.float(), reference, atol=0.10, rtol=0.03
                            )
                            torch.testing.assert_close(built, dense, atol=0.03, rtol=0.01)
                            torch.testing.assert_close(sparse, dense, atol=0.04, rtol=0.01)
                            torch.testing.assert_close(split, sparse, atol=0.005, rtol=0.005)
                            if mask is row_mask:
                                self.assertEqual(sparse[:, :, 0].count_nonzero().item(), 0)
                            if mode == "absmax":
                                self.assertEqual(packed.v_center.count_nonzero().item(), 0)
                            with self.assertRaisesRegex(ValueError, "is_causal"):
                                risa.sparse_int8_attention(
                                    q,
                                    k,
                                    v,
                                    pattern,
                                    v_quantization=mode,
                                    attn_mask=mask,
                                    is_causal=not causal,
                                )

    def test_causal_future_values_are_invisible(self):
        q = torch.zeros(1, 1, 193, 64, device="cuda", dtype=torch.float16)
        v = torch.zeros_like(q)
        v[:, :, 128:] = 100
        for mode in ("midpoint_affine", "absmax"):
            output, pattern = risa.construct_sparse_int8_attention(
                q, q, v, is_causal=True, theta=1.0, v_quantization=mode
            )
            sparse = risa.sparse_int8_attention(
                q, q, v, pattern, is_causal=True, v_quantization=mode
            )
            torch.testing.assert_close(
                output[:, :, :128], torch.zeros_like(output[:, :, :128]), atol=0.01, rtol=0
            )
            torch.testing.assert_close(sparse, output, atol=0.05, rtol=0.01)

    def test_key_bias_large_tile_and_lse(self):
        q = torch.zeros(1, 2, 17, 128, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, 1, 1153, 128, device="cuda", dtype=torch.float16)
        v = torch.ones_like(k) * 3
        mask = torch.linspace(-3, 0, 1153, device="cuda")[None, None, None, :]
        for mode in ("midpoint_affine", "absmax"):
            packed = risa.prequantize_int8_attention(
                q, k, v, attn_mask=mask, return_lse=True, v_quantization=mode
            )
            self.assertEqual(packed.cta_k, 128)
            output, lse = risa.int8_attention_from_prequantized(packed, return_lse=True)
            torch.testing.assert_close(output, torch.full_like(output, 3), atol=0.03, rtol=0)
            torch.testing.assert_close(lse, mask.logsumexp(-1).expand(1, 2, 17), atol=0.01, rtol=0)

    def test_lse_restores_shifted_keys_and_empty_rows(self):
        torch.manual_seed(29)
        q = torch.randn(1, 2, 19, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 1, 257, 128, device="cuda", dtype=torch.float16) * 0.1 + 4
        v = torch.randn_like(k)
        mask = torch.ones(19, 257, dtype=torch.bool, device="cuda")
        mask[0] = False
        output, lse = risa.int8_attention(q, k, v, attn_mask=mask, is_causal=True, return_lse=True)
        scores = q.float() @ k.float().transpose(-1, -2) / 128**0.5
        expected = apply_score_mask(scores, mask, True, 0).logsumexp(-1)
        self.assertTrue(torch.isneginf(lse[:, :, 0]).all())
        self.assertEqual(output[:, :, 0].count_nonzero().item(), 0)
        torch.testing.assert_close(lse[:, :, 1:], expected[:, :, 1:], atol=0.15, rtol=0.02)

    @unittest.skipUnless(os.getenv("RISA_LARGE_GPU_TEST") == "1", "large VRAM test is opt-in")
    def test_padded_v_stride_exceeds_int32(self):
        if torch.cuda.mem_get_info()[0] < 15 * 1024**3:
            self.skipTest("large-V regression requires 15 GiB free CUDA memory")
        q = torch.zeros(1, 56, 1, 128, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, 56, 299521, 128, device="cuda", dtype=torch.float16)
        v = torch.ones_like(k)
        for mode in ("midpoint_affine", "absmax"):
            output = risa.int8_attention(q, k, v, v_quantization=mode)
            torch.testing.assert_close(output, torch.ones_like(output), atol=0.01, rtol=0)
        torch.cuda.synchronize()
