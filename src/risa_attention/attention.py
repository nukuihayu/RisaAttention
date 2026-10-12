# SPDX-License-Identifier: Apache-2.0
"""Stable attention namespace; implementations live in focused modules."""

from ._backend import is_available
from ._execution import int8_attention_from_prequantized
from ._quantization import prequantize_int8_attention
from ._types import PrequantizedInt8Attention, RetainedMassPattern
from .dense import int8_attention
from .reference import build_retained_mass_pattern, measure_pattern_recall
from .sparse import (
    construct_sparse_int8_attention,
    construct_sparse_int8_attention_from_prequantized,
    sparse_int8_attention,
)

__all__ = [
    "PrequantizedInt8Attention",
    "RetainedMassPattern",
    "is_available",
    "int8_attention",
    "int8_attention_from_prequantized",
    "prequantize_int8_attention",
    "build_retained_mass_pattern",
    "measure_pattern_recall",
    "construct_sparse_int8_attention",
    "construct_sparse_int8_attention_from_prequantized",
    "sparse_int8_attention",
]
