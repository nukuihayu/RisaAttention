"""Validated K/V gather for native RISA. No clustering or V quantizer changes."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, init=False)
class KVPermutation:
    """Owned [B,Hkv,Nkv] permutation, validated once before graph capture."""

    _indices: torch.Tensor
    _version: int

    def __init__(self, indices: torch.Tensor):
        if indices.ndim != 3 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("permutation must be [B,Hkv,Nkv] int32/int64")
        if not indices.is_cuda or indices.numel() == 0:
            raise ValueError("permutation must be a nonempty CUDA tensor")
        if indices.shape[-1] > 2**31 - 1:
            raise ValueError("permutation is too long for INT32")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare the permutation before CUDA graph capture")
        expected = torch.arange(indices.shape[-1], device=indices.device)
        if not torch.equal(indices.sort(-1).values, expected.expand_as(indices)):
            raise ValueError("each permutation row must contain each token exactly once")
        with torch.inference_mode(False):
            owned = indices.to(torch.int32).clone().contiguous()
        object.__setattr__(self, "_indices", owned)
        object.__setattr__(self, "_version", owned._version)


def prepare_kv_permutation(indices: torch.Tensor) -> KVPermutation:
    """Validate (synchronizes) and copy a permutation supplied by the caller."""
    return KVPermutation(indices)


def prepare_inputs(q, k, v, *, permutation, mask, causal, return_lse, fuse):
    if not isinstance(permutation, KVPermutation):
        raise TypeError("use prepare_kv_permutation to validate a permutation first")
    perm = permutation._indices
    if perm._version != permutation._version:
        raise ValueError("validated permutation was modified")
    if perm.device != k.device or perm.shape != k.shape[:3]:
        raise ValueError("permutation must match K batch, heads, length, and device")
    if mask is not None or causal:
        raise ValueError("K/V permutation requires dense, unmasked, noncausal attention")
    eligible = fuse and not return_lse and q.dtype == torch.bfloat16 and q.shape[-1] == 128
    if eligible:
        return q, k, v, (perm.data_ptr(),)
    index = perm.long()[..., None].expand_as(k)
    return q, k.gather(2, index), v.gather(2, index), ()
