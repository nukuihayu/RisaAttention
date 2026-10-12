"""RISA attention runtime tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from risa_attention import _validation as validation
from risa_attention import dense as execution
from tests.support.risa_backend import risa


class RisaContractTests(unittest.TestCase):
    def test_minimum_cuda_architecture(self):
        from risa_attention import _backend

        with (
            patch.object(_backend, "_C", object()),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", None),
        ):
            for capability, expected in [((7, 5), False), ((8, 0), True), ((12, 0), True)]:
                with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                    self.assertEqual(_backend.is_available(), expected)
        self.assertEqual(_backend.MINIMUM_COMPUTE_CAPABILITY, (8, 0))

    def test_custom_ops_propagate_causal_and_quantization(self):
        q = torch.randn(1, 2, 3, 64)
        for mode in ("midpoint_affine", "absmax"):
            for causal in (False, True):
                for mask in (None, torch.ones(3, 3, dtype=torch.bool)):
                    with patch.object(
                        execution, "execute_dense_attention", return_value=q.clone()
                    ) as launch:
                        risa.int8_attention(
                            q, q, q, attn_mask=mask, is_causal=causal, v_quantization=mode
                        )
                    self.assertEqual(launch.call_args.kwargs["is_causal"], causal)
                    self.assertEqual(
                        launch.call_args.kwargs["use_midpoint_affine"], mode == "midpoint_affine"
                    )

    def test_fake_and_compile_schema(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            q = torch.empty(1, 2, 17, 64, device="cuda")
            for mode in ("midpoint_affine", "absmax"):
                result = risa.int8_attention(q, q, q, is_causal=True, v_quantization=mode)
                self.assertEqual(result.shape, q.shape)

    def test_invalid_v_mode(self):
        q = torch.empty(1)
        with self.assertRaisesRegex(ValueError, "v_quantization"):
            risa.int8_attention(q, q, q, v_quantization="invalid")

    def test_mask_version_and_causal_reference(self):
        mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
        previous = validation.mask_signature(mask)
        mask[..., 0] = False
        self.assertNotEqual(previous, validation.mask_signature(mask))
        scores = validation.apply_score_mask(torch.zeros(1, 1, 3, 5), mask, True, 0)
        self.assertTrue(torch.isneginf(scores[..., 0, :]).all())
        self.assertTrue(torch.isneginf(scores[..., 1, 2:]).all())
        self.assertEqual(scores[0, 0, 1, 1], 0)
