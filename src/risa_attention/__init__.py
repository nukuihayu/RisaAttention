# SPDX-License-Identifier: Apache-2.0
from ._backend import is_available as int8_attention_is_available
from .attention import (
    PrequantizedInt8Attention,
    RetainedMassPattern,
    build_retained_mass_pattern,
    construct_sparse_int8_attention,
    construct_sparse_int8_attention_from_prequantized,
    int8_attention,
    int8_attention_from_prequantized,
    measure_pattern_recall,
    prequantize_int8_attention,
    sparse_int8_attention,
)
from .permutation import KVPermutation, prepare_kv_permutation

__all__ = [
    "KVPermutation",
    "prepare_kv_permutation",
    "PrequantizedInt8Attention",
    "RetainedMassPattern",
    "build_retained_mass_pattern",
    "construct_sparse_int8_attention",
    "construct_sparse_int8_attention_from_prequantized",
    "int8_attention",
    "int8_attention_from_prequantized",
    "int8_attention_is_available",
    "measure_pattern_recall",
    "prequantize_int8_attention",
    "sparse_int8_attention",
]

try:
    from ._version import __version__
except ImportError:
    # Source-only tests and direct CMake builds do not generate wheel metadata.
    from pathlib import Path as _Path

    _version_file = _Path(__file__).resolve().parents[2] / "VERSION"
    __version__ = (
        _version_file.read_text(encoding="utf-8").strip()
        if _version_file.is_file()
        else "0+unknown"
    )
