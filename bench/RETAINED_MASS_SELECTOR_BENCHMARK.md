# Retained-Mass Selector Benchmark

This benchmark compares the previous PyTorch retained-mass selector with the
parallel CUDA implementation. Both implementations consume the same block
masses and implement the same rule: select the shortest descending-mass prefix
whose sum reaches `theta`, then store selected blocks in original key order.

## Environment

- GPU: NVIDIA GeForce RTX 5090 D v2, compute capability 12.0
- PyTorch: 2.13.0+cu130
- CUDA: 13.0
- Input: BF16, `B=1`, `Hq=16`, `Hkv=4`, `D=128`
- Lengths: 1K, 4K, 8K, and 16K
- Patterns: `video_blocks` and `normal`
- Retained-mass target: `theta=0.99`
- Seed: 0
- Timing: 30 warmups, 200 timed iterations, 20 construction iterations
- comfy-kitchen baseline enabled

## Complete Construction Results

Times include dense INT8 attention, exact block-mass reconstruction, sorting,
retained-mass selection, and CSR construction. Lower is better.

| Length | Pattern | PyTorch median | CUDA median | Median change | PyTorch p90 | CUDA p90 | P90 change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | video_blocks | 0.221 ms | 0.161 ms | -27.0% | 0.242 ms | 0.183 ms | -24.1% |
| 1K | normal | 0.233 ms | 0.132 ms | -43.3% | 0.261 ms | 0.148 ms | -43.3% |
| 4K | video_blocks | 0.536 ms | 0.454 ms | -15.4% | 0.569 ms | 0.467 ms | -17.8% |
| 4K | normal | 0.533 ms | 0.451 ms | -15.4% | 0.554 ms | 0.460 ms | -16.8% |
| 8K | video_blocks | 1.485 ms | 1.389 ms | -6.4% | 1.506 ms | 1.414 ms | -6.1% |
| 8K | normal | 1.458 ms | 1.392 ms | -4.5% | 1.497 ms | 1.417 ms | -5.3% |
| 16K | video_blocks | 4.969 ms | 4.877 ms | -1.8% | 4.983 ms | 4.885 ms | -2.0% |
| 16K | normal | 4.976 ms | 4.891 ms | -1.7% | 4.984 ms | 4.901 ms | -1.7% |

All eight complete construction cases improve in both median and p90 latency.
Peak incremental memory is unchanged or lower: 9.2 MiB at 1K, 44.9 MiB at
4K, 123.2 MiB at 8K, and 376.6 MiB at 16K with the CUDA selector.

## Selector Microbenchmark

This isolates sorting, shortest-prefix selection, ordered CSR packing, and the
single result synchronization. Shapes match the row and key-block counts of
the complete 16-head benchmark.

| Rows | Key blocks | PyTorch | CUDA | Change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 0.153 ms | 0.053 ms | -65.1% |
| 512 | 32 | 0.156 ms | 0.054 ms | -65.7% |
| 1024 | 64 | 0.163 ms | 0.056 ms | -65.6% |
| 2048 | 128 | 0.161 ms | 0.064 ms | -60.3% |

The implementation uses one unified path for every shape. CUB performs the
integer exclusive scan of per-row counts, FP64 row totals are reduced in
parallel, and one warp compacts each row. There is no sequence-length branch,
fixed density, Top-K approximation, or altered retained-mass threshold.

## Correctness and Quality

The A/B outputs have identical:

- selected coverage, sparsity, CSR offsets, indices, and index size;
- exact construction recall and drifted current-step recall;
- NRMSE, MAE, maximum error, cosine similarity, and SQNR;
- dense and sparse attention outputs for each benchmark case.

The reported measured construction recall differs by at most `6.9e-8` because
the CUDA implementation reduces FP64 row totals in a parallel tree instead of
PyTorch's reduction order. This does not change any selected block or output.
Selector tests cover 17 to 2048 rows, 8 to 128 key blocks, and thresholds 0.5,
0.9, 0.99, and 1.0. The complete test suite reports `104 passed`.

## Reproduction

```bash
python bench/benchmark_sparse.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern video_blocks --pattern normal \
  --dtype bfloat16 --theta 0.99 --drift 0.05 \
  --construction-iterations 20 --warmup 30 --iterations 200 --seed 0 \
  --compare-comfy-kitchen
```
