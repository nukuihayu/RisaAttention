# RISA and Sol-Attn benchmark

This benchmark compares RISA with the INT8 Sol-Attn implementation added in
Comfy-Org/comfy-kitchen commit
[`dae00a13`](https://github.com/Comfy-Org/comfy-kitchen/commit/dae00a13d458876570804523ae045a487fd92961).
It was measured on 2026-09-01. Sol-Attn requires equal Q/K/V head counts, so
these results use multi-head self-attention rather than the grouped-query
configuration in the repository's dense benchmark.

## Test conditions

- GPU: NVIDIA GeForce RTX 5090 D v2, SM120
- PyTorch: 2.13.0+cu130
- CUDA toolkit: 13.0
- Input: BF16, `B=1`, `Hq=Hkv=4`, `D=128`
- Lengths: 1K, 4K, 8K and 16K tokens
- Timing: 30 warmups and 100 CUDA-event samples
- Sol-Attn: `tau=1.0`, including routing and workspace allocation
- RISA sparse: `theta=0.99`, including current-step Q/K/V quantization
- Build: `120-real`; the installed comfy-kitchen extension contains SM120
  cubins and no PTX

The input generator is the `video_blocks` structure used by
`benchmark_sparse.py`: two 128-token prototype clusters with prototype norm
`8.0`. The retained-mass pattern is built from `q0` and `k0`; the timed Q and K
add independent normal drift with scale `0.05`. Sol-Attn routes the current Q/K
on every call. The RISA sparse call reuses the pattern built from Q0/K0.

Q/K/V values are shared across implementations. Sol-Attn receives a BTHD view
and the other implementations receive a BHND view. The layout change is a
stride-only transpose and is outside the timed call. PyTorch SDPA is the
numerical reference.

## Steady-state latency

Latency data use seed 0. Values are median with p90 in brackets.

| tokens | PyTorch SDPA | comfy-kitchen dense INT8 | Sol-Attn | RISA dense | RISA sparse | Sol / RISA sparse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.0310 [0.0319] ms | 0.0712 [0.0730] ms | 0.0905 [0.0929] ms | 0.0753 [0.0764] ms | 0.0778 [0.0813] ms | 1.16x |
| 4K | 0.2055 [0.2059] ms | 0.1314 [0.1333] ms | 0.1386 [0.1416] ms | 0.1361 [0.1382] ms | 0.1001 [0.1021] ms | 1.39x |
| 8K | 0.8356 [0.8365] ms | 0.3679 [0.3810] ms | 0.3045 [0.3073] ms | 0.3684 [0.3855] ms | 0.2299 [0.2319] ms | 1.32x |
| 16K | 3.3364 [3.3412] ms | 1.3442 [1.3892] ms | 0.7704 [0.7747] ms | 1.3458 [1.3894] ms | 0.6731 [0.6762] ms | 1.14x |

The last column divides Sol-Attn median latency by RISA sparse median latency.
It describes calls after the RISA pattern exists; construction is handled
separately below. At 1K, PyTorch SDPA remains the fastest implementation.

Peak incremental CUDA allocation for one timed call was:

| tokens | Sol-Attn | RISA sparse |
| ---: | ---: | ---: |
| 1K | 2.93 MiB | 2.51 MiB |
| 4K | 10.67 MiB | 10.02 MiB |
| 8K | 22.00 MiB | 20.04 MiB |
| 16K | 44.00 MiB | 40.07 MiB |

These values include output and temporary storage allocated by the complete
Python entry point. They do not include Q/K/V tensors already live before the
call.

## Numerical error

NRMSE is measured against BF16 PyTorch SDPA. Each cell is the mean over seeds
0, 1 and 2, followed by the observed range.

| tokens | Sol-Attn NRMSE | RISA dense NRMSE | RISA sparse NRMSE | sparse coverage | drifted exact recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.02248 [0.02141, 0.02310] | 0.01794 [0.01768, 0.01838] | 0.02308 [0.02207, 0.02389] | 52.15% | 0.99344 |
| 4K | 0.02588 [0.02508, 0.02657] | 0.01990 [0.01970, 0.02012] | 0.02270 [0.02180, 0.02401] | 50.04% | 0.99522 |
| 8K | 0.03025 [0.02901, 0.03097] | 0.02028 [0.02023, 0.02033] | 0.02483 [0.02372, 0.02546] | 51.21% | 0.99390 |
| 16K | 0.03362 [0.03311, 0.03434] | 0.02020 [0.02007, 0.02038] | 0.02325 [0.02230, 0.02469] | 50.13% | 0.99498 |

RISA sparse has slightly higher NRMSE than Sol-Attn at 1K. From 4K through
16K it has lower NRMSE and lower steady-state latency for this input structure.
RISA dense has the lowest NRMSE at every measured length, but loses the
long-sequence sparse speedup.

Sol-Attn was also measured with `tau=0.5` and `tau=1.4`. The selected route,
output and latency were effectively unchanged on these two-cluster inputs;
their proxy scores did not lie near the tested thresholds. This observation
does not imply that `tau` is inactive on model trajectories.

## Pattern construction and diffusion-step amortization

RISA construction returns the dense output for the construction step together
with the retained-mass pattern. The table reports median synchronized wall
time over five construction calls after one warm construction. The one-time
CUDA module cold start is excluded.

| tokens | RISA construction |
| ---: | ---: |
| 1K | 0.290 ms |
| 4K | 0.353 ms |
| 8K | 0.766 ms |
| 16K | 2.445 ms |

For `N` compatible attention calls, the comparison is

```math
T_{RISA}(N)=C_{build}+(N-1)C_{sparse},
```

```math
T_{Sol}(N)=N C_{Sol}.
```

Using the measured medians:

| tokens | 8 calls: Sol / RISA | 20 calls: Sol / RISA | lower 8-call total | lower 20-call total |
| ---: | ---: | ---: | --- | --- |
| 1K | 0.724 / 0.835 ms | 1.811 / 1.769 ms | Sol-Attn | RISA, by 2.3% |
| 4K | 1.109 / 1.054 ms | 2.773 / 2.255 ms | RISA, by 5.0% | RISA, by 18.7% |
| 8K | 2.436 / 2.375 ms | 6.091 / 5.134 ms | RISA, by 2.5% | RISA, by 15.7% |
| 16K | 6.163 / 7.157 ms | 15.408 / 15.234 ms | Sol-Attn | RISA, by 1.1% |

This is an attention-only upper bound. A real ComfyUI run has separate pattern
state per layer, sampling session and conditioning branch. Pattern
incompatibility or refresh adds another construction. Conversely, Sol-Attn
performs its routing on every call and does not have a persistent construction
phase.

## Scope

The benchmark establishes kernel latency, temporary CUDA allocation and
attention-output error on controlled block-correlated tensors. It does not
measure end-to-end generation speed, image SSIM or PSNR, video temporal
consistency, or pattern stability on captured model trajectories. In
particular, the 8-call and 20-call totals should not be reported as whole-model
speedups.
