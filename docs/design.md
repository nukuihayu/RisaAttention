# Design notes

RISA combines a dense INT8 attention kernel with an optional frozen block
support. This document describes quantization, normalization and sparse support.

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

## 3. V quantization and CUDA execution

V uses per-channel symmetric INT8 quantization:

$$
a_d=\max_n |V_{nd}|,\qquad
s_d=\max(a_d/127,10^{-12}),\qquad
q_{nd}=\mathrm{round}(V_{nd}/s_d).
$$

The quantizer fuses absmax reduction, quantization, padding and the INT8 MMA
permutation. Output dequantization multiplies the normalized accumulator by
$s_d$. With `v_quantization="midpoint_affine"`, the center starts at
$(\min_n V_{nd}+\max_n V_{nd})/2$ and the INT8 step covers half the range.
After fixing codes and scale, the mean quantization residual is added to the
reconstruction center. Output restores this center after normalization.
Both modes use the same rounded U8 probability operand for numerator and
denominator, preserving constant V on short and long sequences.

The denominator sums the same packed U8 probabilities consumed by P x V,
with FP32 online rescaling. This applies to short sequences, tails, masks,
sparse execution and mass capture. It preserves constant V under probability
rounding; it does not remove quantization error relative to floating-point SDPA.

CUDA scheduling depends on architecture, shape and features:

- SM120 can use 64-query dense tiles for suitable D128 shapes, a TMA K/V ring
  for long contiguous BF16 D128 inputs, and a 1024-thread V quantizer for large
  D128 inputs. Other shapes and mask/LSE modes select the general kernel.
- Long query-broadcast masks compact visible tile indices once per head and reuse
  the CSR tile iterator to skip excluded tiles without a separate attention loop.
  Nonpositive scales apply scaling before selecting the softmax maximum.
- Ampere uses bounded U8 conversions. Batch/head bases and captured-statistic
  addresses use 64-bit arithmetic.

Sparse CSR rows and construction statistics remain 128 queries wide. Capture
measures the constructor's executed probability mass without replaying QK.

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

`construct_sparse_int8_attention()` is the production constructor. It captures
the sufficient statistics of the online softmax while producing the dense
output; it does not replay QK.

For query $q$ and key tile $t$, let $\mu_{t,q}$ be the tile maximum and let
$u_{t,q,\ell}$ be the local numerator accumulated by lane $\ell$ into the
softmax denominator. Let $(m_q,d_q)$ be the final online-softmax maximum and
denominator. The tile mass executed by the quantized kernel is

$$
M_t=\sum_q
\frac{\sum_{\ell}u_{t,q,\ell}\,2^{\mu_{t,q}-m_q}}{d_q}.
$$

This follows directly from the online update. When a later tile raises the row
maximum from $\mu_{t,q}$ to $m_q$, every earlier contribution is multiplied by
$2^{\mu_{t,q}-m_q}$; division by the final $d_q$ normalizes the row. Therefore

$$
\sum_t M_t=\sum_q 1=|Q_r|
$$

for each query block $Q_r$, up to FP32 reduction error.

For a complete tile, the probabilities used by the denominator are packed U8
codes. Their subgroup sum is an exact integer no larger than
$255\,\mathrm{CTA}_K$, so construction stores one FP32 numerator and
one FP32 tile maximum per query. For the fused U8 path, conversion of the
integer numerator to FP32 is exact. Short-K and boundary tiles also sum the
packed U8 operand, while keeping the generic FP32 exponentiation schedule.
They store one FP32 numerator and one FP32 maximum.
Two further FP32 values store the final $(m_q,d_q)$. A second CUDA kernel maps
one CTA to each `(query block, key block)` pair and reconstructs all $M_t$ in
parallel. The workspace is

$$
4BH_qL_q(2N_k+2)
$$

bytes for statistics, plus the FP32 mass matrix, where $N_k$ is the number of
key tiles. At 16K tokens with four query heads this is 64.75 MiB.

The selector sorts key tiles by mass, scans for the shortest retained prefix,
and packs selected indices into CSR in execution order. FP64 reductions measure
mass; CUB scans integer row counts. Selection does not impose a fixed Top-K.

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

See [benchmarks](../benchmarks/README.md) for measured latency, output error and
memory. Quantization error depends on Q/K/V distributions. Sparse error also
depends on retained mass and how inputs change after construction.

The 512-token path is sensitive to CPU submission overhead. Ordinary strided
tensors are exported directly through PyTorch's DLPack capsule API; subclasses,
conjugate views, other layouts and mismatched active CUDA devices keep their
protocol behavior. Storage ownership, tensor strides and stream ordering are
preserved. No quantization or attention arithmetic changes.

Long absmax V quantization assembles four destination INT8 bytes into
one 32-bit store. The forward MMA permutation maps each byte to its source
token and inserts zero codes for padding. Word stores require at least four
tokens per thread; shorter sequences retain the original thread mapping.
Range reduction and scales are unchanged. Affine quantization retains its
residual accumulation order. The optimization adds no workspace or kernel
launch. See [benchmarks](../benchmarks/README.md) for measurement commands.

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
7. [CUTLASS architecture support](https://github.com/NVIDIA/cutlass#building),
   which distinguishes SM100 datacenter kernels from SM120 RTX kernels.
