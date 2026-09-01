# Residual-zero midpoint-affine INT8 V benchmark

## Change

For each V channel, RISA still chooses the midpoint center $c_0$ and half-range
scale $s$, then computes fixed signed INT8 codes

$$q_{n}=\mathrm{round}((V_{n}-c_{0})/s).$$

It now stores the least-squares-optimal center for those codes and scale:

$$
c_{\mathrm{opt}}=\frac{1}{N}\sum_{n}V_{n}
-s\frac{1}{N}\sum_{n}q_{n}
=c_{0}+s\frac{1}{N}\sum_{n}\left((V_{n}-c_{0})/s-q_{n}\right).
$$

The CUDA kernel uses the second form. It accumulates the small normalized
rounding residual during the existing quantization pass, so it does not add a
`sum(V)` pass or subtract two large means. The reconstructed channel residual
has zero mean. At 16K, its measured maximum channel-mean residual is
`4.47e-8` and RMS is `9.45e-9`.

## Method

Hardware was an NVIDIA GeForce RTX 5090 D v2 (SM120), with PyTorch
2.13.0+cu130 and BF16 inputs shaped `B=1, Hq=16, Hkv=4, D=128`. Lengths were
512, 1024, 2048, 4096, 8192 and 16384. The four V distributions were standard
normal, normal shifted by +3, absolute-normal, and independent per-channel
shifts in [-3, 3]. Each cell averages seeds 0, 1 and 2. Latency used 20 warmups
and 100 paired, alternating-order iterations per seed and distribution.

All 72 cases produced bitwise-identical Q/K codes and scales between RISA and
comfy-kitchen. Consequently the quality comparison isolates the V path.

## Output quality

Each cell is output RMSE reduction relative to comfy-kitchen absmax INT8 V;
positive is better.

| length | normal | shifted +3 | nonnegative | channel shift |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 6.21% | 25.60% | 15.87% | 18.39% |
| 1K | 6.16% | 17.10% | 13.91% | 12.25% |
| 2K | 5.98% | 17.41% | 15.27% | 13.31% |
| 4K | 6.35% | 18.98% | 19.43% | 16.53% |
| 8K | 6.61% | 23.41% | 26.60% | 20.91% |
| 16K | 6.96% | 24.53% | 29.05% | 23.60% |

The original uncorrected midpoint path was 5.11% worse than comfy-kitchen at
16K normal V. The corrected path is 6.96% better, an 11.48% RMSE reduction
relative to the old midpoint implementation. This removes the observed
zero-centered long-sequence regression rather than merely specializing for
biased or nonnegative V.

## Latency

The fused column is the mean paired latency change versus comfy-kitchen over
all distributions and seeds. Small differences near 1% are treated as ties
because GPU clocks were not locked.

| length | fused latency change | quantization-only change |
| ---: | ---: | ---: |
| 512 | +2.15% | +10.09% |
| 1K | +6.26% | +11.06% |
| 2K | +3.83% | +8.68% |
| 4K | +0.84% | +7.01% |
| 8K | -0.18% | +6.26% |
| 16K | +0.11% | +2.36% |

The extra reduction remains visible when quantization is timed alone, but is
amortized by attention at the long sequence lengths used by the target
ComfyUI workloads. No length threshold or user-facing hyperparameter was
added. A short-length fallback would recover latency but discard measured
quality gains, so it is not part of this change.

The command in `bench/README.md` reproduces the benchmark matrix.
