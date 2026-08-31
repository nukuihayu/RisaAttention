# Design notes

RISA combines a dense INT8 attention kernel with an optional frozen block
support. This document separates inherited kernel behavior from the changes
made here, then states the approximations and their costs.

## 1. Reference operation

For one attention head, scaled dot-product attention is

$$
S = \gamma QK^\mathsf{T}, \qquad
P_{ij} = \frac{\exp S_{ij}}{\sum_\ell \exp S_{i\ell}}, \qquad
O = PV,
$$

where the default scale is $\gamma=1/\sqrt d$. RISA keeps the online-softmax
maximum, denominator, exponential and output dequantization in FP32. Q, K and V
are signed INT8 inside the tensor-core path; softmax probabilities are
represented as unsigned INT8 for the P x V multiplication.

The dense Q/K pipeline, integer MMA layout and online-softmax structure come
from SageAttention and the CUDA backend in comfy-kitchen. RISA does not claim
ConvRot itself as a new contribution relative to that baseline.

## 2. Stabilizing Q/K before quantization

### Shared orthogonal rotation

Let $H$ be the normalized block-Hadamard transform used by ConvRot, so
$HH^\mathsf{T}=I$. Applying the same transform to Q and K leaves exact scores
unchanged:

$$
(QH)(KH)^\mathsf{T}
= QHH^\mathsf{T}K^\mathsf{T}
= QK^\mathsf{T}.
$$

The transform spreads isolated channel outliers across a block. Symmetric INT8
quantization of a vector $x$ then uses

$$
s_x=\max\left(\frac{\lVert x\rVert_\infty}{127},\varepsilon\right),
\qquad
q_x=\mathrm{clip}_{[-127,127]}\!\left(\mathrm{round}(x/s_x)\right).
$$

Reducing the maximum relative to the typical coordinates reduces the step
$s_x$ and therefore their absolute rounding error. Orthogonality protects the
floating-point dot product; it does not make the quantized dot product exact.

The name "rotation-stabilized" refers to this numerical transform. It has no
relationship to RoPE, which encodes token position.

### Representative-key translation

For each batch and KV head, the quantizer may choose a representative key
$k_a$ and translate every key by it:

$$
k'_j = k_j-k_a.
$$

For a fixed query $q_i$ this subtracts the same scalar from every score:

$$
q_i^\mathsf{T}k'_j
=q_i^\mathsf{T}k_j-q_i^\mathsf{T}k_a.
$$

Softmax is invariant to a row-wise constant,
$\mathrm{softmax}(s-c\mathbf 1)=\mathrm{softmax}(s)$, so the
translation is exact before quantization. The implementation enables it only
when the sampled representative reduces the K quantization range. The scratch
state is one INT32 index per batch/KV-head pair.

## 3. Residual-zero midpoint-affine INT8 V

The inherited comfy-kitchen path uses a symmetric range for each V channel:

$$
a_d=\max_n |V_{nd}|, \qquad s_d^{\rm sym}=\max(a_d/127,\varepsilon).
$$

That range wastes levels when a channel is shifted away from zero. RISA first
uses the midpoint and half-range instead:

$$
v_d^- = \min_n V_{nd}, \quad
v_d^+ = \max_n V_{nd}, \quad
c_d^{(0)}=\frac{v_d^-+v_d^+}{2}, \quad
r_d=\frac{v_d^+-v_d^-}{2},
$$

$$
s_d=\max(r_d/127,\varepsilon), \qquad
q_{nd}=\mathrm{clip}_{[-127,127]}
\left(\mathrm{round}\frac{V_{nd}-c_d^{(0)}}{s_d}\right).
$$

Because

$$
\frac{v_d^+-v_d^-}{2}
\leq \max(|v_d^-|,|v_d^+|),
$$

the ideal affine step cannot exceed the symmetric step. Ignoring clipping and
floating-point roundoff, nearest rounding gives
$|V_{nd}-(s_{d}\,q_{nd}+c_{d}^{(0)})|\leq s_{d}/2$.

The midpoint minimizes the worst-case step, but its rounding residual can have
a nonzero channel mean. Normalized attention preserves that DC error instead
of averaging it away. With codes and scale fixed, RISA corrects the center to

$$
c_{d,\mathrm{opt}}=\frac{1}{N}\sum_{n}V_{nd}
-s_{d}\frac{1}{N}\sum_{n}q_{nd}.
$$

Equivalently, the kernel accumulates the small normalized rounding residual

$$
\epsilon_{nd}=\frac{V_{nd}-c_{d}^{(0)}}{s_{d}}-q_{nd}, \qquad
c_{d,\mathrm{opt}}=c_{d}^{(0)}+s_{d}\frac{1}{N}\sum_{n}\epsilon_{nd}.
$$

This form avoids a separate `sum(V)` reduction and avoids subtracting two
large means. The corrected residual
$e_{nd}=V_{nd}-(s_{d}\,q_{nd}+c_{d,\mathrm{opt}})$ satisfies
$\sum_{n}e_{nd}=0$ up to floating-point reduction error. For a probability row
$p$ and uniform row $u_{n}=1/N$, its V-induced output error therefore obeys

$$
\sum_{n}p_{n}\,e_{nd}=\sum_{n}(p_{n}-u_{n})\,e_{nd}, \qquad
\left|\sum_{n}p_{n}\,e_{nd}\right|
\leq \lVert p-u\rVert_{2}\,\lVert e_{d}\rVert_{2}.
$$

Thus the correction is exact for uniform attention and removes the component
most likely to survive broad long-sequence attention. It does not claim to
minimize error for every nonuniform probability row.

The center does not require another output kernel. For any normalized
probability row, including a sparse row renormalized over its selected keys,

$$
\sum_{n=1}^{N} p_{n}\left(s_{d}q_{n,d}+c_{d,\mathrm{opt}}\right)
=s_{d}\sum_{n=1}^{N}p_{n}q_{n,d}+c_{d,\mathrm{opt}}.
$$

The CUDA epilogue restores $c_{d,\mathrm{opt}}$ with one FP32 fused multiply-add per output
element. V min, max, center, scale, quantization and MMA permutation are fused
in `csrc/kernels/quant_v_int8.cu`. The correction changes neither the packed
ABI nor the attention kernel. Its only extra runtime work is a reduction of
eight small residual accumulators in the V quantizer block. Measurements are
recorded in the repository-level `bench/MIDPOINT_V_BENCHMARK.md` report.

## 4. Retained-mass sparse support

The sparse path follows LoSA's central idea: measure dense attention mass in
blocks, keep the smallest high-mass support, then freeze that support across
later diffusion calls.

Let $Q_r$ be a query block and $K_b$ a key tile. Construction aggregates

$$
M_{r,b}=\sum_{i\in Q_r}\sum_{j\in K_b}P_{ij}.
$$

For each row $r$, key tiles are sorted by decreasing $M_{r,b}$. The selected
set $\mathcal S_r$ is the shortest prefix satisfying

$$
\sum_{b\in\mathcal S_r}M_{r,b}
\geq \theta\sum_bM_{r,b}.
$$

Selected tile indices are sorted back into execution order and stored as CUDA
CSR arrays. A row corresponds to one
`[batch, query_head, query_block]`; query blocks contain 128 tokens and key
tiles contain 64 or 128 tokens, matching the existing INT8 CTA geometry.

At a later call, the kernel evaluates only keys in $\mathcal S_r$ and computes

$$
\widehat P_{ij}=
\begin{cases}
\displaystyle
\frac{\exp S_{ij}}{\sum_{\ell:\,b(\ell)\in\mathcal S_r}\exp S_{i\ell}},
& b(j)\in\mathcal S_r,\\[8pt]
0,&\mathrm{otherwise}.
\end{cases}
$$

Q, K and V are recomputed for the current call. The sparse path is therefore
not a KV cache and does not reuse an old attention output.

### Construction paths

`construct_sparse_int8_attention()` is the production constructor. Its dense
call produces the output needed at the construction step, then a QK-only replay
emits normalized mass for each CUDA key tile. Sorting, prefix selection and CSR
assembly remain on the GPU.

`build_retained_mass_pattern()` is the slower reference constructor. It uses
FP32 score and softmax calculations in 128-query chunks and exists for tests.
`measure_pattern_recall()` independently measures exact FP32 mass retained by
a frozen pattern after the inputs drift.

### What theta guarantees

The threshold applies to aggregate mass for a 128-token query block. It does
not guarantee that every query token retains $\theta$ of its own probability
mass. A few tokens can lose more mass than the block average. This is an
important quality limit of the current support representation.

The fast constructor also selects from quantized Sage probability mass, so its
measured construction mass need not equal exact FP32 recall. The benchmark
reports both quantities.

### Cost model

Let $\rho$ be CSR coverage, the selected key-tile fraction. The dominant
attention work changes from
$O(BH_qL_qL_kd)$ to approximately
$O(\rho BH_qL_qL_kd)$. The speedup is lower than $1/\rho$ because Q/K/V
quantization, launch overhead and sparse index traversal remain.

With $R=BH_q\lceil L_q/128\rceil$ CSR rows and $N$ selected tiles, index memory
is exactly

$$
4(R+1+N)\ \mathrm{bytes}.
$$

If construction costs $C_{\rm build}$, a dense call costs $C_d$, and each later
sparse call costs $C_s$, reuse across $n$ later calls is profitable only when

$$
C_{\rm build}+nC_s < (n+1)C_d.
$$

Nearly dense patterns fail this condition. The Python and ComfyUI paths use a
fixed 5% minimum sparsity boundary and execute dense INT8 below it.

### Difference from LoSA

This is a LoSA-inspired backend, not a paper-identical implementation.

- LoSA uses 128 x 32 support blocks; RISA uses 128 x 64 or 128 x 128 blocks to
  match the Sage CTA.
- LoSA executes sparse BF16/FP16 attention through FlashInfer; RISA executes a
  signed-INT8 Sage-derived kernel.
- RISA's fast builder measures quantized probability mass. The reference
  builder is the only FP32 construction path.
- The library does not choose the paper's construction timestep $t_0$.
  Pattern lifetime belongs to the integration layer.

The included ComfyUI node constructs each pattern on its first eligible
attention call. It does not reproduce the paper's $t_0=3$ schedule, so a
pattern may be selected from a noisier step than the paper uses. Delaying or
refreshing patterns would add lifecycle work and must be justified by an
end-to-end speed and quality measurement before entering the runtime path.

These differences are why the public API says "retained-mass pattern" rather
than presenting the backend as a LoSA reproduction.

## 5. Output storage

ComfyUI commonly flattens `[B,H,L,D]` into `[B,L,H D]` before the output
projection. HND-contiguous output requires a transpose copy. With
`output_layout="nhd"`, RISA returns the same logical shape with physical strides
chosen so that the transpose and reshape are views for unpadded D64, D128 and
D256. Attention arithmetic is unchanged.

## 6. ComfyUI state ownership

The node calls `ModelPatcher.set_model_optimized_attention()`. Sparse patterns
are keyed by sampling run, conditioning identity, attention-call ordinal,
device, dtype, Q/K/V shape and scale. They are cleared in `finally` blocks at
the end of sampling. Masks, cross-attention, compilation and incompatible
shapes use dense INT8; there is no implicit PyTorch fallback inside the INT8
modes.

This protects pattern compatibility, but it cannot prove temporal validity.
Freezing support assumes that important attention blocks remain stable across
the later diffusion trajectory. Model-level image and video metrics are still
required.

## 7. Performance and numerical error

### Test conditions

Results below were measured on an NVIDIA GeForce RTX 5090 D v2 (SM120),
PyTorch 2.13.0 with CUDA 13.0, and BF16 input. Every case uses
`B=1,Hq=16,Hkv=4,D=128`. Timed attention calls have 30 warmup iterations and
200 samples. Construction has five samples. CUDA events measure device time;
the median and percentile columns include Q/K/V quantization for fused calls.

The input seed resets to `0` before every case and pattern. This matters for
the structured sparse generator: without a per-case reset, adding an earlier
case changes later Q/K tensors and can change their selected coverage. GPU
clocks were not locked, so small differences below roughly 1% should be
treated as ties.

### Dense normal-input comparison

The bracketed interval is p20-p80 latency. NRMSE is measured against PyTorch
SDPA using identical tensors.

| length | PyTorch SDPA | RISA dense | comfy-kitchen | SageAttention 2.2 | RISA TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.062 [0.062, 0.063] ms | 0.077 [0.074, 0.078] ms | 0.070 [0.070, 0.071] ms | 0.099 [0.097, 0.102] ms | 112.06 |
| 4K | 0.842 [0.841, 0.842] ms | 0.353 [0.352, 0.354] ms | 0.350 [0.349, 0.350] ms | 0.351 [0.345, 0.392] ms | 389.53 |
| 8K | 2.940 [2.937, 2.944] ms | 1.146 [1.100, 1.267] ms | 1.102 [1.099, 1.262] ms | 1.195 [1.055, 1.258] ms | 479.56 |
| 16K | 10.826 [10.824, 10.831] ms | 4.351 [4.349, 4.354] ms | 4.352 [4.350, 4.355] ms | 4.518 [4.512, 4.524] ms | 505.39 |

| length | RISA / SDPA | RISA NRMSE | comfy-kitchen NRMSE | SageAttention NRMSE |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 0.81x | 0.01419 | 0.01514 | 0.03825 |
| 4K | 2.39x | 0.01528 | 0.01629 | 0.03901 |
| 8K | 2.57x | 0.01577 | 0.01690 | 0.03989 |
| 16K | 2.49x | 0.01563 | 0.01683 | 0.03944 |

At 1K the fused quantization overhead makes every tested INT8 path slower than
PyTorch SDPA. RISA crosses over by 4K. RISA and comfy-kitchen remain within
about 1% at 4K and 16K; the wide 8K timing interval also prevents a dense speed
claim over comfy-kitchen. Residual-zero midpoint V lowers NRMSE at every tested
length. The isolated three-seed benchmark measures a 5.98%-6.96% output RMSE
reduction on zero-centered normal V and larger gains on biased V.

### Retained-mass sparse comparison

Sparse data use `video_blocks`, `theta=0.99`, drift `0.05`, two clusters, and
prototype norm `8.0`. Brackets show p10-p90. Construction produces the dense
output for its step and the CSR pattern; later sparse calls reuse only that
pattern and recompute Q/K/V.

| length | construction | dense fused | sparse fused | dense / sparse | coverage | CSR index |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.296 [0.285, 0.308] ms | 0.074 [0.073, 0.075] ms | 0.077 [0.075, 0.078] ms | 0.97x | 57.0% | 0.005 MiB |
| 4K | 0.792 [0.769, 0.815] ms | 0.350 [0.349, 0.352] ms | 0.213 [0.212, 0.214] ms | 1.64x | 53.0% | 0.035 MiB |
| 8K | 2.184 [2.175, 2.201] ms | 1.247 [1.079, 1.266] ms | 0.622 [0.620, 0.624] ms | 2.01x | 50.0% | 0.129 MiB |
| 16K | 7.495 [7.476, 7.502] ms | 4.338 [4.329, 4.340] ms | 2.500 [2.492, 2.505] ms | 1.74x | 54.7% | 0.554 MiB |

The pattern index is small relative to the packed attention buffers. Peak
incremental memory for fused dense and sparse calls is effectively identical:
7.0, 28.1, 56.1, and 112.3 MiB from 1K through 16K. Sparsity reduces visited
K/V tiles and latency; it does not remove the current-step quantized Q/K/V
storage.

Construction must be amortized. Using the measured costs in
$C_{\rm build}+nC_s < (n+1)C_d$ gives:

| length | later reuses needed | 8-call attention-only speedup | 20-call attention-only speedup |
| ---: | ---: | ---: | ---: |
| 1K | never | 0.71x | 0.85x |
| 4K | 4 | 1.23x | 1.45x |
| 8K | 2 | 1.53x | 1.78x |
| 16K | 2 | 1.39x | 1.58x |

The 8-call and 20-call columns assume one construction followed by seven or
nineteen compatible sparse calls. They exclude the rest of the diffusion
model, so they are upper bounds on end-to-end generation speedup. A changed
shape, branch, scale, mask, or pattern refresh adds another construction.

### Sparse numerical error

For reference output $y$ and approximation $\hat y$, the reported normalized
error and signal-to-quantization-noise ratio are

$$
\mathrm{NRMSE}=\frac{\sqrt{\mathrm{mean}((\hat y-y)^2)}}
{\sqrt{\mathrm{mean}(y^2)}}, \qquad
\mathrm{SQNR}=20\log_{10}\frac{\sqrt{\mathrm{mean}(y^2)}}
{\sqrt{\mathrm{mean}((\hat y-y)^2)}}.
$$

`vs dense` isolates the additional sparse-support error from the existing INT8
error. Exact recall is measured on the drifted Q/K tensors with FP32 softmax.

| length | exact recall | NRMSE vs SDPA | NRMSE vs dense | MAE | max abs | SQNR | cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.99206 | 0.02551 | 0.01757 | 0.001656 | 0.09766 | 31.87 dB | 0.999701 |
| 4K | 0.99339 | 0.02536 | 0.01518 | 0.000867 | 0.03914 | 31.92 dB | 0.999704 |
| 8K | 0.99558 | 0.02311 | 0.01014 | 0.000578 | 0.02661 | 32.72 dB | 0.999743 |
| 16K | 0.99318 | 0.02610 | 0.01572 | 0.000455 | 0.02100 | 31.67 dB | 0.999684 |

Construction recall and drifted current recall differ by less than 0.001
percentage point in these synthetic cases. That verifies pattern stability for
this generator and drift value only. It does not show that real diffusion
attention has the same stability. PSNR and SSIM are intentionally absent:
attention tensors are not bounded images, and meaningful image/video metrics
require end-to-end model trajectories.

Random-normal attention is a negative control: it retains nearly every block
at `theta=0.99`, so sparse execution provides no useful acceleration.

### Rejected kernel variants

The following bitwise-equivalent kernel variants were measured and removed:

| candidate | RTX 5090 result |
| --- | --- |
| 256-thread V quantizer instead of 512 | 4-9% slower quantization |
| split D128 Q and K ConvRot kernels | about 7-10% slower after paired warm runs |
| interleaved V scale/center metadata | slower at 4K and 16K |
| special last-tile loop for divisible lengths | no stable gain |

The acceptance rule is end-to-end: a kernel change must improve median and p90
latency on the target GPU without violating its numerical contract or moving
cost into quantization, construction, output layout or Python dispatch.

## References

1. Zhang et al., [SageAttention: Accurate 8-Bit Attention for Plug-and-play
   Inference Acceleration](https://arxiv.org/abs/2410.02367), 2024.
2. Zhang et al., [SageAttention2: Efficient Attention with Thorough Outlier
   Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958),
   2024, and the [official implementation](https://github.com/thu-ml/SageAttention).
   The vendored kernel lineage is pinned in source headers to commit
   [`d1a57a5`](https://github.com/thu-ml/SageAttention/commit/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5).
3. [LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion
   Acceleration](https://arxiv.org/html/2608.12032v1), 2026.
4. [FlashInfer](https://github.com/flashinfer-ai/flashinfer), whose CUDA
   primitives are present in the SageAttention lineage.
5. [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), the baseline for
   the local ConvRot INT8 integration.
6. [H3-Optimizations](https://github.com/Zironic/H3-Optimizations), consulted
   for ComfyUI output-layout and model-patching behavior. Its model-specific
   sparse routing is not included here.
