# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

_EXTENSION_ERROR: Exception | None = None

try:
    if getattr(torch.version, "hip", None) or not getattr(torch.version, "cuda", None):
        raise ImportError("RISA Attention requires an NVIDIA CUDA build of PyTorch")
    from . import _C
except (ImportError, OSError) as error:
    _C = None
    _EXTENSION_ERROR = error


def is_available(device: torch.device | int | str | None = None) -> bool:
    """Return whether the standalone CUDA kernel can run on *device*."""
    if (
        _C is None
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None)
    ):
        return False
    try:
        resolved = None
        if device is not None:
            resolved = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if resolved.type != "cuda":
                return False
        return torch.cuda.get_device_capability(resolved) >= (7, 5)
    except (AssertionError, RuntimeError, ValueError):
        return False


def require_extension() -> None:
    if _C is None:
        detail = f": {_EXTENSION_ERROR}" if _EXTENSION_ERROR is not None else ""
        raise RuntimeError(f"RISA Attention CUDA extension is not available{detail}")


def wrap_for_dlpack(tensor: torch.Tensor):
    """Export a tensor without introducing cross-stream synchronization."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.__dlpack__(stream=-1)
