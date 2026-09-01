# ComfyUI integration

## Installation

Build the Python package with the interpreter used by ComfyUI. For the
`/workspace/ComfyUI` environment on an RTX 5090:

```bash
cd /path/to/RisaAttention
CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real" \
/opt/venv/bin/python -m pip install .

ln -s /path/to/RisaAttention/comfyui-risa-attention \
  /workspace/ComfyUI/custom_nodes/comfyui-risa-attention
```

Link only `comfyui-risa-attention`, not the repository root. The installed CUDA
package remains `risa_attention`; keeping the custom-node module name distinct
prevents import shadowing.

## Model patch

The node clones its `MODEL` input and installs one callable through
`ModelPatcher.set_model_optimized_attention()`. The source model and ComfyUI's
global attention functions are unchanged.

The three modes are:

| mode | behavior |
| --- | --- |
| `pytorch_attention` | installs ComfyUI's `attention_pytorch`; the RISA extension is not imported |
| `int8_attention` | dense rotation-stabilized INT8 Q/K/V attention |
| `sparse_int8_attention` | dense construction followed by frozen retained-mass sparse attention |

Dense INT8 handles GQA, masks and cross-attention. Sparse execution is limited
to unmasked self-attention with equal Q/K sequence lengths. Unsupported sparse
calls execute dense INT8; they do not silently switch to PyTorch.

## Sparse pattern lifetime

ComfyUI exposes two wrapper scopes used by the node:

1. `OUTER_SAMPLE` opens and closes the pattern store for one sampling run.
2. `DIFFUSION_MODEL` resets the attention-call ordinal for each transformer
   forward.

A pattern key includes conditioning identity, attention-call ordinal, device,
dtype, Q/K/V shapes and attention scale. This keeps separate layers, CFG
branches and tensor configurations from sharing a CSR pattern. The store is
cleared in `finally` blocks when sampling ends, including exception paths.

Each pattern is constructed on the first eligible call for that key and then
frozen. The node does not delay construction to LoSA's paper setting of
`t0=3`, nor does it refresh patterns. This keeps runtime state and construction
cost bounded, but it can select support from a high-noise step. Treat this as a
known quality limitation when evaluating real diffusion trajectories.

Tensor-container calls are quantized before their floating-point tensors are
released. Sequence-major output is selected when ComfyUI will immediately
flatten heads, avoiding a separate transpose allocation on supported head
dimensions.

Pattern construction is skipped during compilation, for masked attention,
cross-attention and incompatible shapes. A constructed pattern with less than
5% sparsity is marked dense for the rest of that sampling run because index
traversal is unlikely to repay its overhead.

## Theta

`theta` is the attention mass target for each query block, not a fitted model
coefficient. Lower values select fewer key tiles and increase approximation
error; higher values approach dense attention.

| value | practical meaning |
| ---: | --- |
| `0.99` | node default and the value used by published benchmarks |
| `0.999` | conservative, usually less acceleration |
| `1.0` | effectively dense and normally not useful |

The selected set is data-dependent. The guarantee is aggregate over a
128-token query block, not per token. Quality should be checked on generated
images or videos from the target model and sampler.

## Runtime overhead

Backend callables are captured when the model is patched; the hot path does not
repeat package imports. Dense mode carries no sampling state. Sparse mode adds
one Python dictionary lookup per attention call, while CSR tensors and pattern
selection remain on the GPU.

Pattern construction captures the dense kernel's online-softmax statistics and
does not replay QK. Sorting stays in PyTorch's CUDA implementation; shortest
prefix selection, count scan, measured-mass reduction and ordered CSR packing
run on the unified parallel CUDA path. On the published RTX 5090 benchmark this
reduces complete construction median by 1.7%-43.3% versus the former PyTorch
selector without changing selected support. See the
[selector benchmark](../bench/RETAINED_MASS_SELECTOR_BENCHMARK.md).

Models that bypass ComfyUI's optimized-attention hook are outside this node's
scope. Compatibility is tested against the `ModelPatcher` API in
`/workspace/ComfyUI`; future ComfyUI API changes may require an adapter update.
