# Benchmark Guide

This directory contains the reproducible benchmarks used by the repository
reports. They measure CUDA kernels, not an end-to-end diffusion pipeline. Use
captured model Q/K/V trajectories before making quality or deployment decisions.

## Measurement Contract

- Timing uses CUDA events around the complete Python call and synchronizes each
  sample. Reported latency is the median unless a report says otherwise.
- Warmup calls are excluded. Every requested `--seed` recreates the inputs, so
  results do not depend on case order.
- `benchmark_sparse.py` evaluates numerical metrics against one FP32 PyTorch
  SDPA output outside all timing loops. `benchmark_attention.py` uses PyTorch
  SDPA at the requested input dtype.
- Peak incremental allocation is `max_memory_allocated - memory_allocated`
  after Q/K/V are live. It includes the output and call-local temporaries but
  excludes live inputs. CSR index storage is emitted separately as
  `pattern_index_mib`.
- Dense-equivalent TFLOP/s divides the dense attention operation count by wall
  time. It is an effective throughput measure that makes skipped sparse work
  visible; it is not physical FLOP/s executed by the sparse kernel.

Sparse construction is a complete dense call that also emits a pattern. Its
latency must be amortized against later compatible sparse calls; it is never
included in a steady-state sparse latency sample.

## Script Matrix

| Script | Purpose | Reference | Notes |
| --- | --- | --- | --- |
| `benchmark_attention.py` | Dense, prequantized and packing paths | PyTorch SDPA at input dtype | Optional comfy-kitchen and SageAttention2 baselines |
| `benchmark_midpoint_v.py` | RISA V quantization versus comfy-kitchen V | Shared Q/K codes and scales | Isolates the V quantizer |
| `benchmark_sparse.py` | Construction, sparse reuse, metrics and memory | FP32 PyTorch SDPA | Supports GQA and optional Sage2/Sage3 |
| `plot_scaling.py` | Accuracy, effective-throughput and memory figures | JSON from sparse benchmark | Does not rerun kernels |

All attention cases use `B,Hq,Hkv,Lq,Lkv,D`. `Hq` must be divisible by `Hkv`.
The sparse benchmark requires self-attention (`Lq=Lkv`) because a pattern is
constructed over the same sequence axis.

## Dense Attention

Compare fused, prequantized and quantization-only RISA paths with the common
GQA shape used by the project reports:

```bash
python bench/benchmark_attention.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern normal --pattern channel_outlier --pattern common_key \
  --dtype bfloat16 --seed 0 --warmup 30 --iterations 200 \
  --compare-comfy-kitchen --compare-sage-attention
```

`normal` is the baseline distribution. `channel_outlier` stresses Q/K range
stabilization, and `common_key` stresses softmax-invariant key translation.
NRMSE in this script is the output RMS error divided by the PyTorch SDPA output
RMS value.

## Midpoint-Affine V

This benchmark compares RISA's residual-zero midpoint-affine V quantizer with
comfy-kitchen's symmetric absmax V path across distributions and lengths:

```bash
python bench/benchmark_midpoint_v.py \
  --length 512 --length 1024 --length 2048 \
  --length 4096 --length 8192 --length 16384 \
  --pattern normal --pattern shifted --pattern positive --pattern channel_shift \
  --seed 0 --seed 1 --seed 2 --warmup 20 --iterations 100
```

The script checks that Q/K INT8 codes and scales are bitwise equal before it
attributes an output difference to V quantization. Current results and the
paired timing protocol are in [MIDPOINT_V_BENCHMARK.md](MIDPOINT_V_BENCHMARK.md).

## Retained-Mass Sparse Attention

The sparse benchmark constructs a support from `q0` and `k0`, then times later
calls with drifted `q` and `k`. It reports construction latency, steady-state
latency, selected coverage, construction recall, current-step recall, output
error and CUDA allocation.

```bash
python bench/benchmark_sparse.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,2048,2048,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --case 1,16,4,32768,32768,128 \
  --pattern video_blocks --theta 0.99 --drift 0.05 \
  --seed 0 --seed 1 --seed 2 \
  --warmup 30 --iterations 200 --construction-iterations 20 \
  --compare-comfy-kitchen --json bench/results/risa_gqa_1k_32k.json
```

`video_blocks` creates repeatable 128-token prototype clusters and adds
independent normal Q/K drift after construction. It exercises block-sparse
traversal but is not a replacement for a model trace. `normal` is a negative
control and generally produces near-dense support.

`coverage` is the fraction of key tiles retained. Construction recall measures
the exact mass retained on `q0/k0`; current recall measures the same frozen
pattern against the drifted `q/k`. Both should accompany any sparse speed claim.

## SageAttention 2 and 3

The official SageAttention3 Blackwell API has no GQA/MQA kernel. A fair
SageAttention2/SageAttention3 comparison must therefore use equal query and KV
heads, which is a different workload from the GQA command above:

```bash
python bench/benchmark_sparse.py \
  --case 1,16,16,1024,1024,128 \
  --case 1,16,16,2048,2048,128 \
  --case 1,16,16,4096,4096,128 \
  --case 1,16,16,8192,8192,128 \
  --case 1,16,16,16384,16384,128 \
  --case 1,16,16,32768,32768,128 \
  --pattern video_blocks --theta 0.99 --drift 0.05 \
  --seed 0 --seed 1 --seed 2 \
  --warmup 30 --iterations 200 --construction-iterations 20 \
  --compare-comfy-kitchen --compare-sageattention2 \
  --compare-sageattention3 --json bench/results/risa_sage3_mha_1k_32k.json
```

The upstream Sage2 and Sage3 APIs currently modify K during smoothing. The
benchmark clones K for each Sage call so every sample receives identical inputs
and the caller tensor is preserved; this clone is part of the reported Sage
latency and allocation. Do not compare these MHA results directly with GQA
numbers.

Generate the checked-in figures from the JSON output:

```bash
python bench/plot_scaling.py bench/results/risa_sage3_mha_1k_32k.json \
  --output bench/risa_sage2_sage3_1k_32k.png \
  --svg bench/risa_sage2_sage3_1k_32k.svg \
  --memory-output bench/risa_sage2_sage3_memory_1k_32k.png \
  --memory-svg bench/risa_sage2_sage3_memory_1k_32k.svg
```

The left panel reports mean SQNR against FP32 PyTorch SDPA with one standard
deviation over the requested seeds. The right panel reports median
dense-equivalent throughput. The second figure reports the median peak
incremental allocation described above.

![RISA, SageAttention2, and SageAttention3 accuracy and speed](risa_sage2_sage3_1k_32k.png)

![RISA, SageAttention2, and SageAttention3 temporary CUDA allocation](risa_sage2_sage3_memory_1k_32k.png)

## Additional Reports

- [MIDPOINT_V_BENCHMARK.md](MIDPOINT_V_BENCHMARK.md): V quantization accuracy and latency.
- [RETAINED_MASS_SELECTOR_BENCHMARK.md](RETAINED_MASS_SELECTOR_BENCHMARK.md): CUDA selector A/B validation.

For performance decisions, use complete-call median and tail latency rather
than a prequantized kernel alone. For sparse mode, include construction cost,
coverage and current-step recall in the comparison.
