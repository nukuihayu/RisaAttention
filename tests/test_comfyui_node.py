from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

_PACKAGE_NAME = "risa_attention_comfyui_node"
_PACKAGE_DIR = Path(__file__).parents[1] / "comfyui-risa-attention"
_PACKAGE_SPEC = importlib.util.spec_from_file_location(
    _PACKAGE_NAME,
    _PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(_PACKAGE_DIR)],
)
if _PACKAGE_SPEC is None or _PACKAGE_SPEC.loader is None:
    raise RuntimeError("failed to load the ComfyUI custom-node package")
_PACKAGE = importlib.util.module_from_spec(_PACKAGE_SPEC)
sys.modules[_PACKAGE_NAME] = _PACKAGE
_PACKAGE_SPEC.loader.exec_module(_PACKAGE)
node = importlib.import_module(f"{_PACKAGE_NAME}.node")


class FakeBackend:
    def __init__(self):
        self.last_shapes = None
        self.dense_calls = 0
        self.construct_calls = 0
        self.sparse_calls = 0

    @staticmethod
    def int8_attention_is_available():
        return True

    def int8_attention(self, q, k, v, *, scale, attn_mask):
        self.dense_calls += 1
        self.last_shapes = (q.shape, k.shape, v.shape, scale, attn_mask)
        return q

    def sparse_int8_attention(self, q, k, v, pattern, *, scale, output_layout):
        self.sparse_calls += 1
        return q

    def construct_sparse_int8_attention(self, q, k, v, *, theta, scale, output_layout):
        self.construct_calls += 1
        return q, SimpleNamespace(sparsity=0.5, theta=theta)

    def prequantize_int8_attention(self, q, k, v, *, scale, attn_mask):
        self.last_shapes = (q.shape, k.shape, v.shape, scale, attn_mask)
        return q

    def int8_attention_from_prequantized(
        self, packed, *, sparse_pattern=None, output_layout
    ):
        if sparse_pattern is not None:
            self.sparse_calls += 1
        return packed

    def construct_sparse_int8_attention_from_prequantized(
        self, packed, *, theta, output_layout
    ):
        self.construct_calls += 1
        return packed, SimpleNamespace(sparsity=0.5, theta=theta)


def test_node_contract():
    inputs = node.RISAAttentionNode.INPUT_TYPES()["required"]

    assert list(inputs) == ["model", "attention", "theta"]
    assert inputs["attention"][0] == node._ATTENTION_MODES
    assert inputs["theta"][1]["default"] == 0.99
    assert node.RISAAttentionNode.RETURN_TYPES == ("MODEL",)
    assert node.RISAAttentionNode.FUNCTION == "patch_model"


def test_dense_adapter_reshapes_comfy_tensors_and_preserves_gqa():
    backend = FakeBackend()
    attention = node.make_int8_attention_function(backend)
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 6, 16)
    v = torch.randn(1, 6, 16)

    output = attention(q, k, v, heads=4, enable_gqa=True, scale=0.25)

    assert backend.last_shapes[:4] == (
        torch.Size([1, 4, 4, 8]),
        torch.Size([1, 2, 6, 8]),
        torch.Size([1, 2, 6, 8]),
        0.25,
    )
    assert output.shape == q.shape


def test_sparse_constructs_once_per_attention_ordinal_and_reuses():
    backend = FakeBackend()
    state = node._SparsePatternState(theta=0.99)
    attention = node.make_sparse_attention_function(backend, state)
    q = torch.randn(1, 4, 2048, 8)

    state.begin_session()
    for _ in range(2):
        state.begin_forward({"uuids": ["positive"]})
        attention(q, q, q, heads=4, skip_reshape=True)
        attention(q, q, q, heads=4, skip_reshape=True)
        state.end_forward()

    assert backend.construct_calls == 2
    assert backend.sparse_calls == 2
    assert len(state.patterns) == 2
    state.end_session()
    assert state.patterns == {}


def test_sparse_keeps_conditioning_branches_separate():
    backend = FakeBackend()
    state = node._SparsePatternState(theta=0.99)
    attention = node.make_sparse_attention_function(backend, state)
    q = torch.randn(1, 4, 2048, 8)

    state.begin_session()
    for conditioning in ("positive", "negative", "positive"):
        state.begin_forward({"uuids": [conditioning]})
        attention(q, q, q, heads=4, skip_reshape=True)
        state.end_forward()
    state.end_session()

    assert backend.construct_calls == 2
    assert backend.sparse_calls == 1


def _fake_model():
    patched = SimpleNamespace(wrappers=[])
    patched.set_model_optimized_attention = lambda function: setattr(
        patched, "attention", function
    )
    patched.add_wrapper_with_key = lambda wrapper_type, key, function: (
        patched.wrappers.append((wrapper_type, key, function))
    )
    return SimpleNamespace(clone=lambda: patched), patched


def test_node_patches_dense_attention_without_sampling_wrappers():
    model, patched = _fake_model()
    backend = FakeBackend()

    with patch.object(node, "_load_risa_attention", return_value=backend):
        output = node.RISAAttentionNode().patch_model(model, "int8_attention", 0.99)

    assert output == (patched,)
    assert callable(patched.attention)
    assert callable(patched.attention.container_function)
    assert patched.wrappers == []


def test_node_patches_frozen_support_with_scoped_comfy_wrappers():
    model, patched = _fake_model()
    backend = FakeBackend()
    wrappers = SimpleNamespace(OUTER_SAMPLE="outer", DIFFUSION_MODEL="diffusion")

    original_import = node.importlib.import_module

    def import_module(name):
        if name == "comfy.patcher_extension":
            return SimpleNamespace(WrappersMP=wrappers)
        return original_import(name)

    with (
        patch.object(node, "_load_risa_attention", return_value=backend),
        patch.object(node.importlib, "import_module", side_effect=import_module),
    ):
        output = node.RISAAttentionNode().patch_model(
            model, "sparse_int8_attention", 0.99
        )

    assert output == (patched,)
    assert [wrapper[0] for wrapper in patched.wrappers] == ["outer", "diffusion"]
    assert callable(patched.attention)
