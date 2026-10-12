# Benchmarks

Run from the repository root after installing this checkout. The runners verify
that the loaded Python package matches its source and record the CUDA extension
hash. Install `matplotlib` and `numpy` to generate plots; SageAttention and
comfy-kitchen are optional comparison dependencies.

## Published results

The [main README](../README.md#benchmarks) contains performance and accuracy
tables measured on October 10, 2026 with RTX 5090, PyTorch 2.13.0+cu130,
CUDA 13.0, SageAttention 2.2.0 (Sage2++) and comfy-kitchen 0.2.35.
Only PNG result images are retained in this repository:

| Suite | Accuracy and throughput | Temporary memory |
| --- | --- | --- |
| Dense | [PNG](figures/dense-rtx5090.png) | [PNG](figures/dense-memory-rtx5090.png) |
| Sparse | [PNG](figures/sparse-rtx5090.png) | [PNG](figures/sparse-memory-rtx5090.png) |

## Protocol

- Seeds `0, 1, 2`; BF16 and HND storage by default.
- 200 warmups; ten CUDA-event samples, each containing twenty calls.
- Methods rotate positions and reverse each rotation cycle on identical inputs.
  Choose a sample count divisible by the method count to balance positions.
- Accuracy uses FP32 SDPA in query chunks with at most 16M score elements.
- Full calls include Q/K/V preparation. Sparse construction uses two warmups
  and ten single-call samples, measured separately from reuse.
- Memory is peak allocated bytes above live inputs. CSR index storage is separate.
- Tables use median per-seed latency and mean accuracy. SQNR error bars show
  one standard deviation across seeds; throughput is dense-equivalent work.

`--warmup`, `--samples` and `--calls-per-sample` control timing counts.
`--v-affine off|on|both` selects `absmax`, `midpoint_affine`, or a paired run.
`--layout nhd` changes physical storage while retaining logical HND shape.
Use `--case B,Hq,Hkv,Lq,Lkv,D` and `--seed` repeatedly for explicit cases.

## Reproduce the figures

Generated JSON/CSV and profiling output belong in ignored `benchmarks/results/`.
Raw measurements are not shipped; rerun the suites to generate new plots.

Dense 512–32K GQA (`B=1, Hq=16, Hkv=4, D=128`), normal Q/K/V:

```bash
python benchmarks/dense.py --v-affine both --compare-sage --compare-kitchen \
  --output benchmarks/results/dense.json
python benchmarks/plot_scaling.py benchmarks/results/dense.json --pattern normal \
  --output benchmarks/figures/dense-rtx5090.png \
  --memory-output benchmarks/figures/dense-memory-rtx5090.png
```

Sparse 1K–32K GQA, two block clusters, prototype norm `8`, drift `0.05`,
`theta=0.99` and shared support for both V modes:

```bash
python benchmarks/sparse.py --v-affine both --compare-sage --compare-kitchen \
  --output benchmarks/results/sparse.json
python benchmarks/plot_scaling.py benchmarks/results/sparse.json \
  --output benchmarks/figures/sparse-rtx5090.png \
  --memory-output benchmarks/figures/sparse-memory-rtx5090.png
```

Export a CSV summary from a new run:

```bash
python benchmarks/report.py benchmarks/results/dense.json \
  --csv benchmarks/results/dense.csv
```

## Additional tools

- `dense.py`: `--value-distribution normal|shifted|outlier|constant` selects V
  controls; `--include-components` adds quantization and prequantized timings.
- `sparse.py`: `--pattern normal` checks the nearly dense support control.
- `masks.py`: broadcast key-mask comparisons.
- `compare_native.py`: paired timings against a baseline CUDA extension.
- `profile_operators.py`: NVTX ranges for Nsight; `--stage dense|quantize|attend`
  selects full calls, preparation or prequantized attention.
- `--guard-comfyui http://localhost:8188`: aborts if ComfyUI has queued work
  when sharing the GPU.

Use each script's `--help` for supported options. JSON schema version `2`
records environment, configuration, per-seed results, timing samples,
accuracy, memory and affine comparisons. Operator profiles have a separate
schema and cannot be passed to the scaling plotter.
