# RISA Attention

CUDA-native INT8 attention for diffusion inference, with dense and
retained-mass sparse execution.

RISA (Rotation-stabilized INT8 Sparse Attention) combines a SageAttention-style
tensor-core pipeline with three numerical components:

- shared orthogonal Q/K rotation and softmax-invariant key translation;
- residual-zero midpoint-affine INT8 quantization for V;
- reusable block-sparse support selected by retained attention mass.

The package provides a PyTorch API and a standalone ComfyUI custom node.

## Highlights

- Signed INT8 Q/K/V tensor-core attention with FP32 online softmax.
- FP32, FP16 and BF16 inputs.
- Grouped-query attention, unequal Q/K lengths and broadcastable masks.
- Head dimensions from 1 to 256, padded internally to D64, D128 or D256.
- Prequantized execution for explicit memory-lifetime control.
- GPU-resident sparse pattern construction and CUDA CSR traversal.
- Sampling-scoped ComfyUI integration through
  `ModelPatcher.set_model_optimized_attention()`.
- CUDA Graph-compatible dense execution.

## Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Linux |
| Python | 3.10 or newer |
| PyTorch | CUDA build, 2.4 or newer |
| CUDA Toolkit | 13.0 or newer |
| GPU | NVIDIA SM75 or newer |
| Build system | CMake 3.26 or newer and Ninja |

RISA builds native cubins for the selected architecture. Build the package on
the target machine, or set `CMAKE_CUDA_ARCHITECTURES` explicitly when creating
a wheel.

## Installation

From a source checkout:

```bash
python -m pip install .
```

CMake uses the visible GPU as the native target by default. To build explicitly
for an RTX 5090:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real" \
python -m pip install .
```

Verify the installed extension:

```bash
python -c "import risa_attention as risa; print(risa.int8_attention_is_available())"
```

## Quick Start

Q, K and V use `[batch, heads, sequence, head_dim]` layout.

```python
import torch
import risa_attention as risa

q = torch.randn(1, 16, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

output = risa.int8_attention(q, k, v)
```

Separate quantization from attention when the caller needs explicit control
over temporary tensor lifetimes:

```python
packed = risa.prequantize_int8_attention(q, k, v)
output = risa.int8_attention_from_prequantized(packed)
```

Construct retained-mass sparse support during a dense call, then reuse it:

```python
dense_output, pattern = risa.construct_sparse_int8_attention(
    q, k, v, theta=0.99
)
later_output = risa.sparse_int8_attention(q, k, v, pattern)

print(pattern.coverage)
print(pattern.measured_retained_mass)
print(pattern.index_bytes)
```

A sparse pattern is tied to its tensor shape, head layout, device and attention
scale. Applications decide how long the pattern remains valid.

## ComfyUI

Build the package with ComfyUI's Python interpreter, then install the bundled
node directory:

```bash
cd /path/to/RisaAttention
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real" \
/path/to/comfyui/python -m pip install .

ln -s /path/to/RisaAttention/comfyui-risa-attention \
  /path/to/ComfyUI/custom_nodes/comfyui-risa-attention
```

Restart ComfyUI and add the **RISA Attention** node before the sampler.

| Mode | Behavior |
| --- | --- |
| `pytorch_attention` | ComfyUI PyTorch attention |
| `int8_attention` | Dense RISA INT8 attention |
| `sparse_int8_attention` | Dense pattern construction followed by retained-mass sparse reuse |

The node scopes sparse patterns by sampling run, conditioning branch and
attention-call ordinal. Masks, cross-attention, compilation and incompatible
sparse shapes use dense INT8. See [ComfyUI integration](docs/comfyui.md) for
lifecycle details.

## Performance

Measured on an NVIDIA GeForce RTX 5090 D v2 with PyTorch 2.13.0, CUDA 13.0,
BF16, `B=1`, `Hq=16`, `Hkv=4` and `D=128`. Dense values are fused
end-to-end latency and NRMSE against PyTorch SDPA; each case uses 30 warmups and
200 timed calls.

| Length | PyTorch SDPA | RISA dense INT8 | comfy-kitchen INT8 | RISA / SDPA |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 0.062 ms | 0.077 ms / 0.01419 | 0.070 ms / 0.01514 | 0.81x |
| 4K | 0.842 ms | 0.353 ms / 0.01528 | 0.350 ms / 0.01629 | 2.39x |
| 8K | 2.940 ms | 1.146 ms / 0.01577 | 1.102 ms / 0.01690 | 2.57x |
| 16K | 10.826 ms | 4.351 ms / 0.01563 | 4.352 ms / 0.01683 | 2.49x |

Residual-zero midpoint V removes the long-sequence regression of the original
midpoint implementation. Against comfy-kitchen's absmax V, output RMSE is
5.98%-6.96% lower on zero-centered normal inputs from 2K to 16K, with larger
improvements on shifted and nonnegative V. See the
[V quantization benchmark](bench/MIDPOINT_V_BENCHMARK.md).

Sparse measurements use structured `video_blocks` inputs, `theta=0.99` and
drift `0.05`. Construction cost is reported separately.

| Length | Construction | Dense INT8 | Sparse INT8 | Coverage | Dense / sparse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.296 ms | 0.074 ms | 0.077 ms | 57.0% | 0.97x |
| 4K | 0.792 ms | 0.350 ms | 0.213 ms | 53.0% | 1.64x |
| 8K | 2.184 ms | 1.247 ms | 0.622 ms | 50.0% | 2.01x |
| 16K | 7.495 ms | 4.338 ms | 2.500 ms | 54.7% | 1.74x |

Sparse construction must be amortized across later calls. At 1K, dense
attention is the practical choice. The 4K case repays construction after four
reuses; the measured 8K and 16K cases repay it after two.

## API

| Entry point | Purpose |
| --- | --- |
| `int8_attention` | Fused dense INT8 attention |
| `prequantize_int8_attention` | Pack Q/K/V without executing attention |
| `int8_attention_from_prequantized` | Execute dense attention from packed tensors |
| `construct_sparse_int8_attention` | Return dense output and a retained-mass pattern |
| `sparse_int8_attention` | Execute attention over a frozen sparse pattern |
| `build_retained_mass_pattern` | FP32 reference pattern builder |
| `measure_pattern_recall` | Measure exact retained mass after input drift |

Supported entry points are exported from `risa_attention`; the
`risa_attention._C` extension is private.

## Benchmarks and Tests

Run the CUDA and adapter tests:

```bash
python -m pytest -q -p no:cacheprovider
```

Benchmark commands, metrics and JSON output options are documented in
[bench/README.md](bench/README.md).

```bash
python bench/benchmark_attention.py --help
python bench/benchmark_midpoint_v.py --help
python bench/benchmark_sparse.py --help
```

## Repository Layout

```text
RisaAttention/
|-- src/risa_attention/          Python API
|-- csrc/kernels/                CUDA kernels
|-- comfyui-risa-attention/      ComfyUI node
|-- bench/                       Reproducible benchmarks
|-- tests/                       CUDA and adapter tests
|-- docs/RISA_ATTENTION_DESIGN_ZH.md  Chinese design article
|-- docs/design.md               Algorithms and implementation notes
`-- docs/comfyui.md              ComfyUI lifecycle and compatibility
```

## Documentation

- [Design and numerical methods](docs/design.md)
- [RISA Attention design article (Chinese)](docs/RISA_ATTENTION_DESIGN_ZH.md)
- [ComfyUI integration](docs/comfyui.md)
- [Benchmark guide](bench/README.md)
- [Contributing](CONTRIBUTING.md)

## License and Attribution

RISA Attention is available under the [Apache License 2.0](LICENSE).

The dense kernel builds on techniques and CUDA implementations from
SageAttention and comfy-kitchen. Sparse support selection is informed by LoSA,
and parts of the kernel infrastructure derive from FlashInfer. Derived source
files retain their upstream copyright and license notices; implementation
details and differences are documented in [docs/design.md](docs/design.md).
