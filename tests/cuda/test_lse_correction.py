"""Direct regression for fused anchor correction, independent of attention error."""

import pytest
import torch

from tests.support.risa_backend import risa


@pytest.mark.skipif(not risa.is_available(), reason="RISA CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_lse_shift_gqa_strides_and_disabled_anchor(dtype, dim):
    from risa_attention import _backend as backend

    if not hasattr(backend._C, "risa_lse_shift"):
        pytest.skip("requires rebuilt LSE kernel")
    torch.manual_seed(35)
    q = torch.randn(2, 17, 4, dim, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 9, 2, dim, device="cuda", dtype=dtype).transpose(1, 2)
    anchors = torch.tensor([[-1, 0], [3, 2]], device="cuda", dtype=torch.int32)
    out = torch.empty(2, 4, 17, device="cuda")
    backend._C.risa_lse_shift(
        *[backend.wrap_for_dlpack(x) for x in (q, k, anchors, out)],
        dim**-0.5,
        torch.cuda.current_stream().cuda_stream,
    )
    selected = k.gather(2, anchors.clamp_min(0).long()[..., None, None].expand(2, 2, 1, dim))
    selected = selected.float() * (anchors >= 0)[..., None, None]
    reference = (q.float() * selected.repeat_interleave(2, 1)).sum(-1) * dim**-0.5
    torch.testing.assert_close(out, reference, rtol=2e-5, atol=5e-6)
    assert torch.count_nonzero(out[0, :2]) == 0
