"""ComfyUI adapter for the RISA Attention CUDA package."""

from __future__ import annotations

import importlib
import threading

import torch

_RISA_ATTENTION_BACKEND = None
_MISSING = object()
_DENSE_ONLY = object()
_MIN_PATTERN_SPARSITY = 0.05
_WRAPPER_KEY = "risa_sparse_attention"
_ATTENTION_MODES = [
    "pytorch_attention",
    "int8_attention",
    "sparse_int8_attention",
]


def _load_risa_attention():
    global _RISA_ATTENTION_BACKEND
    if _RISA_ATTENTION_BACKEND is not None:
        return _RISA_ATTENTION_BACKEND
    try:
        backend = importlib.import_module("risa_attention")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The RISA Attention CUDA extension is unavailable. Build it with "
            "ComfyUI's Python environment from the custom-node directory: "
            "CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=<SM>-real' "
            "python -m pip install ."
        ) from exc
    if not backend.int8_attention_is_available():
        raise RuntimeError(
            "The installed RISA Attention extension does not support the active GPU. "
            "Rebuild it on this machine so only the current SM cubin is generated."
        )
    _RISA_ATTENTION_BACKEND = backend
    return backend


def _prepare_inputs(q, k, v, heads, mask, skip_reshape, enable_gqa):
    batch = q.shape[0]
    dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if not skip_reshape:
        q = q.unsqueeze(3).reshape(batch, -1, heads, dim_head)
        key_heads = k.shape[-1] // dim_head if enable_gqa else heads
        value_heads = v.shape[-1] // dim_head if enable_gqa else heads
        k = k.unsqueeze(3).reshape(batch, -1, key_heads, dim_head)
        v = v.unsqueeze(3).reshape(batch, -1, value_heads, dim_head)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    return q, k, v, mask, batch, dim_head


def _reshape_output(output, batch, heads, dim_head, skip_output_reshape):
    if skip_output_reshape:
        return output
    return output.transpose(1, 2).reshape(batch, -1, heads * dim_head)


def _tuple_key(value):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        value = (value,)
    result = tuple(value)
    try:
        hash(result)
    except TypeError:
        return tuple(id(item) for item in result)
    return result


def _transformer_options(args, kwargs):
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    for value in reversed(args):
        if isinstance(value, dict) and (
            "sigmas" in value
            or "sample_sigmas" in value
            or "optimized_attention_override" in value
        ):
            return value
    return {}


class _SparsePatternState:
    """Sampling-scoped patterns keyed by attention call and conditioning branch."""

    def __init__(self, theta: float):
        self.theta = float(theta)
        self.patterns = {}
        self._session_depth = 0
        self._local = threading.local()

    def begin_session(self):
        if self._session_depth == 0:
            self.patterns.clear()
        self._session_depth += 1

    def end_session(self):
        self._session_depth = max(0, self._session_depth - 1)
        if self._session_depth == 0:
            self.patterns.clear()

    def begin_forward(self, transformer_options):
        context = (
            _tuple_key(transformer_options.get("uuids")),
            _tuple_key(transformer_options.get("cond_or_uncond")),
        )
        stack = getattr(self._local, "forward_stack", None)
        if stack is None:
            stack = []
            self._local.forward_stack = stack
        stack.append([context, 0])

    def end_forward(self):
        stack = getattr(self._local, "forward_stack", None)
        if stack:
            stack.pop()

    def next_pattern_key(self, q, k, v, mask, scale):
        stack = getattr(self._local, "forward_stack", None)
        if self._session_depth == 0 or not stack:
            return None
        frame = stack[-1]
        call_index = frame[1]
        frame[1] += 1
        if (
            torch.compiler.is_compiling()
            or mask is not None
            or q.ndim != 4
            or k.ndim != 4
            or v.ndim != 4
            or q.shape[2] != k.shape[2]
        ):
            return None
        return (
            frame[0],
            call_index,
            q.device.type,
            q.device.index,
            q.dtype,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            None if scale is None else float(scale),
        )

    def get(self, key):
        return self.patterns.get(key, _MISSING)

    def put(self, key, pattern):
        self.patterns[key] = (
            pattern if pattern.sparsity >= _MIN_PATTERN_SPARSITY else _DENSE_ONLY
        )

    def outer_sample_wrapper(self, executor, *args, **kwargs):
        self.begin_session()
        try:
            return executor(*args, **kwargs)
        finally:
            self.end_session()

    def diffusion_model_wrapper(self, executor, *args, **kwargs):
        self.begin_forward(_transformer_options(args, kwargs))
        try:
            return executor(*args, **kwargs)
        finally:
            self.end_forward()


def make_int8_attention_function(backend):
    """Bind dense INT8 attention without sampling-state overhead."""
    int8_attention = backend.int8_attention
    prequantize = backend.prequantize_int8_attention
    attend_prequantized = backend.int8_attention_from_prequantized

    def risa_int8_attention(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        del attn_precision
        q, k, v, mask, batch, dim_head = _prepare_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        output = int8_attention(q, k, v, scale=kwargs.get("scale"), attn_mask=mask)
        return _reshape_output(output, batch, heads, dim_head, skip_output_reshape)

    def risa_int8_attention_containers(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        del attn_precision
        q, k, v = q.take(), k.take(), v.take()
        q, k, v, mask, batch, dim_head = _prepare_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        packed = prequantize(q, k, v, scale=kwargs.get("scale"), attn_mask=mask)
        del q, k, v
        output = attend_prequantized(
            packed,
            output_layout="hnd" if skip_output_reshape else "nhd",
        )
        return _reshape_output(output, batch, heads, dim_head, skip_output_reshape)

    risa_int8_attention.container_function = risa_int8_attention_containers
    return risa_int8_attention


def make_sparse_attention_function(backend, state=None):
    """Bind retained-mass sparse INT8 attention to ComfyUI's contract."""
    state = state or _SparsePatternState(theta=0.99)
    int8_attention = backend.int8_attention
    sparse_int8_attention = backend.sparse_int8_attention
    construct_sparse = backend.construct_sparse_int8_attention
    prequantize = backend.prequantize_int8_attention
    attend_prequantized = backend.int8_attention_from_prequantized
    construct_prequantized = backend.construct_sparse_int8_attention_from_prequantized

    def risa_sparse_attention(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        del attn_precision
        q, k, v, mask, batch, dim_head = _prepare_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        scale = kwargs.get("scale")
        key = state.next_pattern_key(q, k, v, mask, scale)
        pattern = state.get(key) if key is not None else _DENSE_ONLY
        if pattern is _MISSING:
            output_layout = "hnd" if skip_output_reshape else "nhd"
            output, pattern = construct_sparse(
                q,
                k,
                v,
                theta=state.theta,
                scale=scale,
                output_layout=output_layout,
            )
            state.put(key, pattern)
        elif pattern is _DENSE_ONLY:
            output = int8_attention(q, k, v, scale=scale, attn_mask=mask)
        else:
            output_layout = "hnd" if skip_output_reshape else "nhd"
            output = sparse_int8_attention(
                q,
                k,
                v,
                pattern,
                scale=scale,
                output_layout=output_layout,
            )
        return _reshape_output(output, batch, heads, dim_head, skip_output_reshape)

    def risa_sparse_attention_containers(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        del attn_precision
        q, k, v = q.take(), k.take(), v.take()
        q, k, v, mask, batch, dim_head = _prepare_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        scale = kwargs.get("scale")
        key = state.next_pattern_key(q, k, v, mask, scale)
        pattern = state.get(key) if key is not None else _DENSE_ONLY
        packed = prequantize(q, k, v, scale=scale, attn_mask=mask)
        del q, k, v
        output_layout = "hnd" if skip_output_reshape else "nhd"
        if pattern is _MISSING:
            output, pattern = construct_prequantized(
                packed,
                theta=state.theta,
                output_layout=output_layout,
            )
            state.put(key, pattern)
        else:
            sparse_pattern = None if pattern is _DENSE_ONLY else pattern
            output = attend_prequantized(
                packed,
                sparse_pattern=sparse_pattern,
                output_layout=output_layout,
            )
        return _reshape_output(output, batch, heads, dim_head, skip_output_reshape)

    risa_sparse_attention.container_function = risa_sparse_attention_containers
    risa_sparse_attention.sparse_state = state
    return risa_sparse_attention


class RISAAttentionNode:
    """Patch a ComfyUI model with a selected attention implementation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attention": (_ATTENTION_MODES, {"default": "sparse_int8_attention"}),
                "theta": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.9,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Attention mass retained by each frozen sparse pattern.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "RISA Attention"
    DESCRIPTION = (
        "Select PyTorch, dense RISA INT8, or sampling-scoped sparse RISA attention."
    )

    def patch_model(self, model, attention, theta):
        patched = model.clone()

        if attention == "pytorch_attention":
            attention_function = importlib.import_module(
                "comfy.ldm.modules.attention"
            ).attention_pytorch
        elif attention == "int8_attention":
            attention_function = make_int8_attention_function(_load_risa_attention())
        elif attention == "sparse_int8_attention":
            wrappers = importlib.import_module("comfy.patcher_extension").WrappersMP
            state = _SparsePatternState(theta=theta)
            attention_function = make_sparse_attention_function(
                _load_risa_attention(), state
            )
            patched.add_wrapper_with_key(
                wrappers.OUTER_SAMPLE, _WRAPPER_KEY, state.outer_sample_wrapper
            )
            patched.add_wrapper_with_key(
                wrappers.DIFFUSION_MODEL,
                _WRAPPER_KEY,
                state.diffusion_model_wrapper,
            )
        else:
            raise ValueError(f"Unsupported attention mode: {attention}")

        patched.set_model_optimized_attention(attention_function)
        return (patched,)
