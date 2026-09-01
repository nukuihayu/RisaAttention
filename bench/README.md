# Benchmarks

Both benchmark scripts use CUDA events, explicit synchronization, fixed input
seeds, warmup iterations and optional JSON output. They measure the complete
call as well as prequantized attention, so quantization cost stays visible.
Each case and input pattern resets to `--seed` (default `0`), making results
independent of command-line case order.

## Dense attention

`benchmark_attention.py` compares PyTorch SDPA with RISA's fused, prequantized
and quantization-only paths. comfy-kitchen and SageAttention are optional
baselines.

```bash
python bench/benchmark_attention.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern normal --pattern channel_outlier --pattern common_key \
  --dtype bfloat16 --seed 0 --warmup 30 --iterations 200 \
  --compare-comfy-kitchen --compare-sage-attention \
  --json bench/results/dense.json
```

A case is `B,Hq,Hkv,Lq,Lkv,D`. NRMSE is measured against PyTorch SDPA with the
same inputs, dtype and scale.

## V quantization comparison

residual-zero midpoint-affine V from the shared Q/K path and compares it with
comfy-kitchen's symmetric absmax V across multiple sequence lengths, V
distributions and seeds.

```bash
python bench/benchmark_midpoint_v.py \
  --length 512 --length 1024 --length 2048 \
  --length 4096 --length 8192 --length 16384 \
  --pattern normal --pattern shifted --pattern positive \
  --pattern channel_shift --seed 0 --seed 1 --seed 2 \
  --warmup 20 --iterations 100 --json bench/results/midpoint_v.json
```

The script verifies bitwise-equal Q/K codes and scales before attributing an
output error difference to V quantization. Results from the current kernel are
summarized in the repository-level `bench/MIDPOINT_V_BENCHMARK.md`.

## Retained-mass sparse attention

`benchmark_sparse.py` separates the construction call from later sparse calls.
It reports median and p10/p90 latency, HND/NHD flatten cost, CUDA memory, CSR
index size, selected coverage, exact retained mass, NRMSE, MAE, maximum error,
cosine similarity and SQNR.

```bash
python bench/benchmark_sparse.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern video_blocks --theta 0.99 --drift 0.05 \
  --seed 0 --warmup 30 --iterations 200 --construction-iterations 5 \
  --compare-comfy-kitchen \
  --json bench/results/sparse.json
```

`normal` is a negative control and is usually nearly dense. `video_blocks`
injects repeatable block structure to exercise sparse traversal and pattern
drift. Neither input generator substitutes for captured Q/K/V trajectories or
end-to-end image/video quality evaluation.

## Sol-Attn comparison

The measured comparison with comfy-kitchen's INT8 Sol-Attn implementation is
recorded in [SOL_ATTN_BENCHMARK.md](SOL_ATTN_BENCHMARK.md). It uses equal-head
self-attention because Sol-Attn does not accept GQA inputs, and reports
steady-state latency separately from RISA pattern-construction amortization.

## Reading results

Use fused end-to-end median and p90 for performance decisions. A faster
prequantized kernel is not sufficient if construction or quantization erases
the gain. Sparse comparisons also need coverage and exact current-step recall;
theta alone does not state how much mass remains after the pattern has drifted.
