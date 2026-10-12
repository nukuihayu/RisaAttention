"""Capsule ownership, storage views, and tensor protocol compatibility."""

import pytest
import torch

from risa_attention._backend import wrap_for_dlpack


@pytest.mark.parametrize("requires_grad", [False, True])
def test_capsule_preserves_storage_and_strides(requires_grad):
    source = torch.arange(24.0).reshape(4, 6).requires_grad_(requires_grad)
    view = source[:, ::2]
    restored = torch.utils.dlpack.from_dlpack(wrap_for_dlpack(view))
    assert restored.data_ptr() == view.data_ptr()
    assert restored.stride() == view.stride()
    assert not restored.requires_grad
    restored[1, 1] = 42
    assert source[1, 2].item() == 42


def test_tensor_subclass_retains_dlpack_protocol():
    streams = []

    class ProtocolTensor(torch.Tensor):
        def __dlpack__(self, *, stream=None):
            streams.append(stream)
            return super().__dlpack__(stream=stream)

    source = torch.ones(3).as_subclass(ProtocolTensor)
    restored = torch.utils.dlpack.from_dlpack(wrap_for_dlpack(source))
    assert streams == [None]
    assert restored.data_ptr() == source.data_ptr()


def test_conjugate_view_retains_protocol_error():
    source = torch.ones(3, dtype=torch.complex64).conj()
    with pytest.raises((BufferError, RuntimeError), match="conjugate"):
        wrap_for_dlpack(source)
