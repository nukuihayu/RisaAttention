# RISA Attention：面向扩散模型的旋转稳定 INT8 稀疏注意力

## 摘要

RISA（Rotation-stabilized INT8 Sparse Attention）是一套用于扩散模型推理的 CUDA attention 实现。计算骨架来自 SageAttention/comfy-kitchen 的 INT8 tensor-core 路径，修改针对两类误差来源和一类性能瓶颈：Q/K 的通道离群值、V 的非零中心量化残差，以及长序列中低概率质量 attention block 的重复计算。对应的方法是共享正交 Q/K 旋转与 key translation、residual-zero midpoint-affine V 量化，以及按概率质量构造并复用的 block-sparse support。

本文从 attention 不变量和量化误差出发推导这些设计，随后讨论 CUDA 数据布局、稀疏模式生命周期和构造成本。RTX 5090 实验表明，dense RISA 在 4K-16K token 上相对 PyTorch SDPA 达到 2.39x-2.57x；V 量化相对 comfy-kitchen 持续降低 NRMSE。对具有块结构的输入，`theta=0.99` 的 sparse 路径在 4K-16K 上相对 dense RISA 达到 1.64x-2.01x steady-state 加速。所有稀疏结果均单独计入 pattern 构造成本和 drift 后的 FP32 exact recall。

## 1. 问题背景与设计目标

扩散图像和视频模型正在使用更长的 token 序列。分辨率、帧数或 patch 数增加后，标准注意力的主要计算量为

```math
O(BH L_qL_kd),
```

其中 $B$ 是 batch size，$H$ 是 query head 数，$L_q$ 和 $L_k$ 是序列长度，$d$ 是 head dimension。当 $L_q=L_k=L$ 时，序列长度翻倍会使 QK 和 PV 的主体计算量增加到约四倍。图像分辨率、视频帧数和 patch 密度因此会直接放大 attention 开销。

现有方案分别解决了问题的一部分：

- PyTorch SDPA 提供完整精度和通用接口，计算量仍随 $L_qL_k$ 增长。
- SageAttention 和 comfy-kitchen 将 QK/PV 主体映射到 INT8 tensor core；量化误差中，V 的通道偏置会直接传递到输出。
- LoSA 利用扩散轨迹中 attention support 的时间稳定性复用稀疏结构；其 BF16/FP16 执行路径、128 x 32 support block 和构造调度与现有 Sage INT8 CTA 不同。
- 端到端 attention 延迟还包括 Q/K/V 量化、support 构造、CSR 遍历和输出布局转换。单独的 MMA kernel 吞吐不能代表完整调用成本。

RISA 将研究问题限定为三项：

1. 在不改变浮点注意力语义的前提下，让 Q/K 更适合 INT8 量化。
2. 降低 V 量化中会被归一化注意力保留下来的系统偏差。
3. 在长序列且 attention support 稳定时，以较小索引开销减少实际访问的 K/V tile。

实验以 fused end-to-end 调用的 median 和分位延迟为性能判据，以相对 SDPA 的 NRMSE、SQNR、cosine similarity 和稀疏 exact recall 为数值判据。候选实现只有在完整路径上产生稳定收益才进入生产代码。

方法与计算代价的对应关系如下：

| 问题 | 数学依据 | 实现位置 | 主要新增代价 |
| --- | --- | --- | --- |
| Q/K 通道离群值 | 共同正交变换保持点积 | Q/K ConvRot quantizer | block-Hadamard 变换 |
| K 动态范围偏移 | softmax 对行常数平移不变 | representative-key translation | 每个 batch/KV-head 一个索引 |
| V 非零中心 | midpoint 最小化通道最大半径 | fused V quantizer | min/max reduction |
| V 残差均值 | 固定 code/scale 下的最小二乘中心 | V quantizer 与 attention epilogue | 小规模残差 reduction 和一次 FMA |
| 长序列冗余 tile | 最小 tile 数的 retained-mass 前缀 | QK mass replay、GPU sort、CSR kernel | 一次构造与 CSR traversal |

## 2. 计算路径与数值边界

对单个 attention head，标准 scaled dot-product attention 为

```math
S=\gamma QK^\mathsf{T},
\qquad
P_{ij}=\frac{\exp S_{ij}}{\sum_l\exp S_{il}},
\qquad
O=PV,
```

默认缩放为 $\gamma=1/\sqrt d$。

RISA 的 tensor-core 主路径将 Q、K、V 表示为有符号 INT8，将 softmax 概率表示为无符号 INT8，以整数 MMA 完成 QK 和 PV。数值敏感部分仍保留 FP32：

- online softmax 的行最大值；
- 指数与归一化分母；
- 输出反量化和 V 中心恢复。

INT8 由此只覆盖计算密集的矩阵乘法；softmax 的动态范围处理和最终反量化保持 FP32。后续各节讨论的误差均发生在 Q/K/V 与 probability 的整数表示中，而不是由低精度 softmax 累加引入。

## 3. Q/K 的等价变换与量化残差

### 3.1 共享正交旋转

INT8 对离群值很敏感。若一个通道的绝对值远大于其他通道，对称量化的步长会被这个最大值决定：

```math
s_x=\max\left(\frac{\lVert x\rVert_\infty}{127},\epsilon\right),
\qquad
q_x=\mathrm{clip}_{[-127,127]}\left(\mathrm{round}(x/s_x)\right).
```

大步长会让其余大多数坐标承受更大的舍入误差。RISA 沿用 ConvRot 思路，对 Q 和 K 使用相同的归一化 block-Hadamard 变换 $H$。由于 $H$ 是正交矩阵，$HH^\mathsf{T}=I$，因此浮点分数保持不变：

```math
(QH)(KH)^\mathsf{T}=QHH^\mathsf{T}K^\mathsf{T}=QK^\mathsf{T}.
```

旋转把少量通道离群值分散到一个块内，通常能降低量化 scale。这里的“旋转稳定”指量化前的数值变换，与用于位置编码的 RoPE 无关。

设旋转后并反量化的向量为

```math
\widetilde q=qH+e_q,
\qquad
\widetilde k=kH+e_k,
```

其中 $e_q,e_k$ 是量化残差。量化 score 与精确 score 的差为

```math
\widetilde q\widetilde k^\mathsf{T}-qk^\mathsf{T}
=qHe_k^\mathsf{T}+e_qH^\mathsf{T}k^\mathsf{T}+e_qe_k^\mathsf{T}.
```

正交旋转消除了变换本身的 score 误差；剩余三项完全由量化残差决定。ConvRot 的作用点因而可以精确表述为：通过降低旋转后向量的峰均比来减小 $e_q$ 和 $e_k$，而非改变 attention 公式。

### 3.2 softmax 不变的 representative-key translation

RISA 还可以为每个 batch 和 KV head 选择一个代表 key $k_a$，并平移全部 key：

```math
k'_j=k_j-k_a.
```

对固定 query $q_i$，所有分数都减去同一个常量：

```math
q_i^\mathsf{T}k'_j=q_i^\mathsf{T}k_j-q_i^\mathsf{T}k_a.
```

softmax 对行常数平移不变：

```math
\mathrm{softmax}(s-c\mathbf{1})=\mathrm{softmax}(s).
```

因此 key translation 在量化前不改变 probability。实现比较平移前后的 K 动态范围，只在候选 $k_a$ 缩小量化范围时采用平移；额外状态为每个 batch/KV-head 一个 INT32 索引。

共享旋转利用点积对共同正交基变换的不变性，key translation 利用 softmax 对行常数的不变性。两者都在浮点 attention 的等价类内选择量化条件更好的表示。

## 4. V 的非零中心与残差均值

Q/K 的误差影响 softmax 权重，V 的误差则直接进入加权输出。comfy-kitchen 基线对每个 V 通道采用以零为中心的对称范围：

```math
a_d=\max_n|V_{nd}|,
\qquad
s_d^{\mathrm{sym}}=\max(a_d/127,\epsilon).
```

当某个通道整体偏离零时，对称范围的一侧不会被数据使用。例如数值落在 $[2,4]$ 时，对称量化仍覆盖 $[-4,4]$，有效数据只占量化区间的四分之一。此时误差下限主要由区间中心的选择决定。

### 4.1 midpoint-affine INT8

RISA 先按通道计算最小值和最大值：

```math
v_d^-=\min_nV_{nd},
\qquad
v_d^+=\max_nV_{nd}.
```

以区间中点作为初始中心，以半区间作为量化半径：

```math
c_d^{(0)}=\frac{v_d^-+v_d^+}{2},
\qquad
r_d=\frac{v_d^+-v_d^-}{2},
```

```math
s_d=\max(r_d/127,\epsilon),
\qquad
q_{nd}=\mathrm{clip}_{[-127,127]}
\left(\mathrm{round}\frac{V_{nd}-c_d^{(0)}}{s_d}\right).
```

中点也是使最大中心距离最小的解：

```math
c_d^{(0)}
=\underset{c}{\arg\min}\max_n|V_{nd}-c|
=\frac{v_d^-+v_d^+}{2}.
```

由于

```math
\frac{v_d^+-v_d^-}{2}
\leq \max(|v_d^-|,|v_d^+|),
```

理想 affine 步长不会大于对称步长。忽略浮点舍入与 clipping 时，最近邻量化误差满足

```math
|V_{nd}-(s_dq_{nd}+c_d^{(0)})|\leq s_d/2.
```

因此 midpoint-affine 量化在给定有符号 INT8 码域下取得最小的通道最大半径。该结论约束逐元素量化误差；attention 输出还取决于残差沿 token 维度的均值。

### 4.2 residual-zero center correction

midpoint 的舍入残差可能具有非零均值。由于每一行 attention probability 的和为 1，这个 DC 分量会原样进入加权输出。

固定 INT8 code $q_{nd}$ 和 scale $s_d$ 后，可将中心写成一个一维最小二乘问题：

```math
J_d(c)=\sum_n\left(V_{nd}-s_dq_{nd}-c\right)^2.
```

令导数为零，

```math
\frac{\mathrm dJ_d}{\mathrm dc}
=-2\sum_n\left(V_{nd}-s_dq_{nd}-c\right)=0,
```

得到唯一最优中心

```math
c_{d,\mathrm{opt}}
=\frac{1}{N}\sum_nV_{nd}
-s_d\frac{1}{N}\sum_nq_{nd}.
```

实现中等价地累计归一化舍入残差

```math
\epsilon_{nd}
=\frac{V_{nd}-c_d^{(0)}}{s_d}-q_{nd},
\qquad
c_{d,\mathrm{opt}}=c_d^{(0)}+s_d\frac{1}{N}\sum_n\epsilon_{nd}.
```

CUDA 实现不直接计算两个均值，而是累计归一化舍入残差。这样省去额外的 `sum(V)` reduction，也避免两个大均值相减。修正后的残差

```math
e_{nd}=V_{nd}-(s_dq_{nd}+c_{d,\mathrm{opt}})
```

满足 $\sum_ne_{nd}=0$，误差只与 attention probability 相对均匀分布的偏离有关：

```math
\sum_np_ne_{nd}=\sum_n(p_n-u_n)e_{nd},
\qquad u_n=1/N,
```

```math
\left|\sum_np_ne_{nd}\right|
\leq \lVert p-u\rVert_2\lVert e_d\rVert_2.
```

当 $p=u$ 时，输出中的 V 量化残差为零。对一般 $p$，上式给出的误差界随 $\lVert p-u\rVert_2$ 增长。因而 $c_{d,\mathrm{opt}}$ 的严格最优域是固定 code/scale 下的 token 维均方重建；attention probability 越接近均匀分布，该目标与输出误差越一致。高度集中的 probability 对少数 token 残差赋予更大权重，其输出误差由右侧范数界刻画。

中心恢复不需要额外输出 kernel。对任意归一化概率行：

```math
\sum_np_n(s_dq_{n,d}+c_{d,\mathrm{opt}})
=s_d\sum_np_nq_{n,d}+c_{d,\mathrm{opt}}.
```

CUDA epilogue 为每个输出元素增加一次 FP32 fused multiply-add，即可恢复中心。V 的 min/max、中心、scale、量化和 MMA permutation 被融合在同一个量化 kernel 中，packed ABI 与 attention kernel 保持不变。

## 5. 由概率质量定义稀疏 support

固定 block density 只规定计算预算，不能规定被删除的 probability mass。不同层和 query block 的 attention entropy 不同，相同 density 对应的遗漏质量可能相差很大。

RISA 采用 LoSA 启发的 retained-mass support。设 $Q_r$ 是一个 query block，$K_b$ 是一个 key tile，构造阶段聚合完整 attention probability：

```math
M_{r,b}=\sum_{i\in Q_r}\sum_{j\in K_b}P_{ij}.
```

对每个 query block 按 $M_{r,b}$ 从大到小排序，选择满足下式的最短前缀 $\mathcal S_r$：

```math
\sum_{b\in\mathcal S_r}M_{r,b}
\geq \theta\sum_bM_{r,b}.
```

由于 $M_{r,b}\geq0$，按质量降序取得的前 $k$ 个 tile 在所有大小为 $k$ 的集合中具有最大总质量。因此上述最短前缀也是达到阈值所需 tile 数最少的集合。默认 $\theta=0.99$；该值直接定义保留质量，不参与模型相关的多项式拟合。

选择完成后，tile 索引重新按执行顺序排列，并保存为 GPU 上的 CSR。每一行对应一个 `[batch, query_head, query_block]`；query block 固定为 128 token，key tile 为 64 或 128 token，与已有 Sage INT8 CTA 几何保持一致。

后续调用只遍历选中的 key tile，并在 support 内重新归一化：

```math
\widehat P_{ij}=
\begin{cases}
\dfrac{\exp S_{ij}}
{\sum_{l:\,b(l)\in\mathcal S_r}\exp S_{il}},
& b(j)\in\mathcal S_r,\\
0,&b(j)\notin\mathcal S_r.
\end{cases}
```

后续步骤沿用 $\mathcal S_r$，但重新量化当前 Q、K、V，并重新计算 support 内的 score、softmax 和 PV。时间复用仅发生在离散 block 索引上，数值张量和 attention 输出均来自当前步骤。

### 5.1 与 LoSA 的实现差异

RISA 保留了 LoSA 的 retained-mass selection，但执行路径有四项差异：

- LoSA 使用 128 x 32 support block；RISA 使用 128 x 64 或 128 x 128，以复用 Sage CTA。
- LoSA 通过 FlashInfer 执行 BF16/FP16 sparse attention；RISA 使用有符号 INT8 tensor-core kernel。
- 生产构造器使用量化后的 Sage probability mass；只有参考构造器使用 FP32 score 和 softmax。
- RISA 库不决定论文中的构造时刻 $t_0$，pattern 生命周期由集成层负责。

公开 API 因此使用描述实际数据结构和执行方式的 `RetainedMassPattern` 与 `sparse_int8_attention`。

### 5.2 质量约束的粒度与输出误差

对 query token $i$，定义 support 内质量

```math
m_i=\sum_{j:\,b(j)\in\mathcal S_r}P_{ij}.
```

再分别定义 support 内外归一化后的 V 均值 $y_i^{S}$ 和 $y_i^{T}$。完整输出与稀疏输出可写为

```math
y_i=m_i y_i^{S}+(1-m_i)y_i^{T},
\qquad
\widehat y_i=y_i^{S}.
```

两者之差为

```math
\widehat y_i-y_i
=(1-m_i)(y_i^{S}-y_i^{T}).
```

若 $\lVert V_j\rVert_2\leq V_{\max}$，则

```math
\lVert\widehat y_i-y_i\rVert_2
\leq 2(1-m_i)V_{\max}.
```

这给出了 retained mass 与输出扰动之间的直接关系。在上述浮点 probability 定义下，$\theta$ 约束以 128-token query block 为单位。由于每个 probability row 的总质量为 1，构造条件等价于

```math
\frac{1}{|Q_r|}\sum_{i\in Q_r}m_i\geq\theta.
```

因此 block 内 token 的平均遗漏质量不超过 $1-\theta$，但单个 $m_i$ 没有相同的下界。该粒度来自 CTA 和 CSR 行的设计：若改为逐 token support，索引量与遍历分歧都会显著增加。

冻结 support 还要求 $m_i$ 在后续扩散步骤中保持稳定。benchmark 通过 drift 后的 FP32 exact recall 直接测量这一量；真实模型上的判断则需要逐层捕获扩散轨迹。

### 5.3 GPU pattern 构造

生产接口 `construct_sparse_int8_attention()` 在一次调用内完成两项工作。首先执行 dense INT8 attention，返回构造时刻本来就需要的模型输出；随后进行一次 QK-only replay，为每个 CUDA key tile 输出归一化 probability mass。tile 排序、最短前缀选择和 CSR 组装全部留在 GPU 上。

构造步骤不重复 PV，但必须重新遍历 QK。设量化、dense QK、PV 和 tile selection 的成本分别为 $C_{\mathrm{quant}}$、$C_{QK}$、$C_{PV}$ 和 $C_{\mathrm{select}}$，可写成

```math
C_{\mathrm{build}}\approx C_{\mathrm{quant}}+2C_{QK}+C_{PV}+C_{\mathrm{select}}.
```

这解释了构造延迟高于单次 dense 调用的原因。将 mass 复制到 CPU 后排序会增加同步和传输，生产路径没有采用这种实现。

项目另提供 `build_retained_mass_pattern()` 作为数值参考，按 128-query chunk 使用 FP32 score 和 softmax 测试 support selection，不进入运行时热路径。生产构造器依据量化后的 Sage probability mass 选 tile；`measure_pattern_recall()` 再以 FP32 softmax 独立测量构造时刻或 drift 后的 exact recall。构造依据与评估依据由此分离。

## 6. 稀疏路径的成本模型

设 CSR coverage 为 $\rho$，主体计算量从

```math
O(BH_qL_qL_kd)
```

下降到近似

```math
O(\rho BH_qL_qL_kd).
```

量化、kernel launch 和 CSR 遍历与 coverage 不成正比，因此 $1/\rho$ 只是主体 MMA 的理想上界。若 CSR 有 $R$ 行、共选择 $N$ 个 tile，索引内存为

```math
4(R+1+N)\ \mathrm{bytes}.
```

设构造调用耗时 $C_{\mathrm{build}}$，dense 调用耗时 $C_d$，后续 sparse 调用耗时 $C_s$。构造步骤同时返回该步所需的 dense 输出，因此一次构造加 $n$ 次复用应与 $n+1$ 次 dense 调用比较：

```math
C_{\mathrm{build}}+nC_s<(n+1)C_d.
```

由此得到两个运行时判据：

1. 将 sparsity 低于 5% 的 pattern 标记为 dense，后续调用不再支付 CSR traversal 成本；
2. 短序列是否启用 sparse 由实测 $C_{\mathrm{build}}$、$C_d$ 和 $C_s$ 决定。RTX 5090 的 1K case 不满足回本条件。

## 7. 面向 ComfyUI 的工程设计

### 7.1 pattern 的所有权

节点通过 `ModelPatcher.set_model_optimized_attention()` 安装 attention callable，不修改全局 attention 函数。稀疏状态以以下信息作为 key：

- sampling session；
- conditioning/CFG 分支标识；
- 每次 diffusion forward 内的 attention 调用序号；
- device、dtype、Q/K/V shape；
- attention scale。

该键空间将不同层、CFG 分支和 tensor 配置隔离。采样结束时，pattern store 在 `finally` 中清理，异常退出采用同一释放路径。

masked attention、cross-attention、编译捕获和不兼容 shape 执行同一 RISA 后端的 dense INT8 路径；INT8 模式不会在内部切换到 PyTorch attention。

### 7.2 输出布局也是热路径的一部分

ComfyUI 常把 `[B,H,L,D]` 转成 `[B,L,H D]` 后送入输出投影。如果 kernel 固定返回 HND contiguous，随后 transpose 和 reshape 可能触发一次完整拷贝。

RISA 的 `output_layout="nhd"` 返回相同逻辑 shape，并选择使后续 transpose/reshape 成为 view 的物理 stride。对未 padding 的 D64、D128 和 D256，该布局消除了独立的输出转置分配，attention 算术保持不变。

### 7.3 只编译目标 GPU

构建脚本读取当前 GPU compute capability，并使用 `<SM>-real`。RTX 5090 对应 `120-real`，产物只包含 `sm_120.cubin`，不包含 PTX 或其他架构的 cubin。扩展部署到不同 SM 时需要重新编译。

## 8. RTX 5090 实验结果

### 8.1 测试条件

- GPU：NVIDIA GeForce RTX 5090 D v2，SM 120；
- PyTorch：2.13.0，CUDA 13.0；
- 输入：BF16，`B=1, Hq=16, Hkv=4, D=128`；
- 每个 timed call：30 次 warmup，200 次 CUDA event 计时；
- 每个 case 和输入 pattern 都重新设置 seed 0；
- GPU 时钟未锁定，小于约 1% 的差异视为持平。

核心命令如下；四个 `--case` 分别设置为 1K、4K、8K 和 16K：

```bash
python bench/benchmark_attention.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern normal --dtype bfloat16 --seed 0 \
  --warmup 30 --iterations 200 \
  --compare-comfy-kitchen --compare-sage-attention

python bench/benchmark_sparse.py \
  --case 1,16,4,1024,1024,128 \
  --case 1,16,4,4096,4096,128 \
  --case 1,16,4,8192,8192,128 \
  --case 1,16,4,16384,16384,128 \
  --pattern video_blocks --theta 0.99 --drift 0.05 \
  --seed 0 --warmup 30 --iterations 200 \
  --construction-iterations 5
```

本节报告 attention 张量上的 kernel-level 性能与数值误差。端到端生成质量在第 10 节单独界定。

### 8.2 Dense RISA 对比 SDPA、comfy-kitchen 和 SageAttention

| 长度 | PyTorch SDPA | RISA dense | comfy-kitchen | SageAttention 2.2 | RISA / SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.062 [0.062, 0.063] ms | 0.077 [0.074, 0.078] ms | 0.070 [0.070, 0.071] ms | 0.099 [0.097, 0.102] ms | 0.81x |
| 4K | 0.842 [0.841, 0.842] ms | 0.353 [0.352, 0.354] ms | 0.350 [0.349, 0.350] ms | 0.351 [0.345, 0.392] ms | 2.39x |
| 8K | 2.940 [2.937, 2.944] ms | 1.146 [1.100, 1.267] ms | 1.102 [1.099, 1.262] ms | 1.195 [1.055, 1.258] ms | 2.57x |
| 16K | 10.826 [10.824, 10.831] ms | 4.351 [4.349, 4.354] ms | 4.352 [4.350, 4.355] ms | 4.518 [4.512, 4.524] ms | 2.49x |

方括号为 p20-p80。speedup 使用 median 计算。

1K case 中，RISA 的 0.077 ms 高于 SDPA 的 0.062 ms，量化与 launch 开销尚未被矩阵乘法节省抵消。交叉点出现在 1K 与 4K 之间；4K-16K 的 RISA/SDPA speedup 为 2.39x-2.57x。RISA 与 comfy-kitchen 在 4K、16K 的差异约为 1%，8K 的分位区间存在重叠，因此 dense kernel 的性能按持平处理。

| 长度 | RISA NRMSE | comfy-kitchen NRMSE | SageAttention NRMSE |
| ---: | ---: | ---: | ---: |
| 1K | 0.01419 | 0.01514 | 0.03825 |
| 4K | 0.01528 | 0.01629 | 0.03901 |
| 8K | 0.01577 | 0.01690 | 0.03989 |
| 16K | 0.01563 | 0.01683 | 0.03944 |

residual-zero midpoint V 在四个长度上都降低了对 SDPA 的 NRMSE。相对 comfy-kitchen，降幅分别为 6.27%、6.20%、6.69% 和 7.13%。独立三 seed 实验测得零中心 normal V 的输出 RMSE 降低 5.98%-6.96%，带通道偏置的 V 降幅更高。该差异来自 V quantizer 与 epilogue center restoration；执行路径没有增加独立输出 pass。

### 8.3 `theta=0.99` 的 retained-mass sparse 结果

稀疏输入使用可重复的 `video_blocks` 结构、drift 0.05、两个 cluster、prototype norm 8.0。

| 长度 | 构造 | dense fused | sparse fused | dense / sparse | coverage | CSR 索引 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.296 [0.285, 0.308] ms | 0.074 [0.073, 0.075] ms | 0.077 [0.075, 0.078] ms | 0.97x | 57.0% | 0.005 MiB |
| 4K | 0.792 [0.769, 0.815] ms | 0.350 [0.349, 0.352] ms | 0.213 [0.212, 0.214] ms | 1.64x | 53.0% | 0.035 MiB |
| 8K | 2.184 [2.175, 2.201] ms | 1.247 [1.079, 1.266] ms | 0.622 [0.620, 0.624] ms | 2.01x | 50.0% | 0.129 MiB |
| 16K | 7.495 [7.476, 7.502] ms | 4.338 [4.329, 4.340] ms | 2.500 [2.492, 2.505] ms | 1.74x | 54.7% | 0.554 MiB |

方括号为 p10-p90。构造列包含该步 dense 输出和 pattern 生成。

4K-16K 的 steady-state dense/sparse 比值为 1.64x、2.01x 和 1.74x。1K 的比值为 0.97x，此时 CSR traversal 抵消了减少 tile 带来的收益。

稀疏路径仍为当前步骤生成量化 Q/K/V buffer，显存下降仅来自未物化完整 attention matrix 的共同在线 softmax 设计，而非 coverage。dense 和 sparse fused 调用的峰值增量显存基本一致：1K、4K、8K、16K 分别约为 7.0、28.1、56.1、112.3 MiB。CSR 索引分别为 0.005、0.035、0.129 和 0.554 MiB。

### 8.4 稀疏误差和 support 稳定性

以 PyTorch SDPA 输出 $y$ 为参考，NRMSE 和 SQNR 定义为

```math
\mathrm{NRMSE}
=\frac{\sqrt{\mathrm{mean}((\hat y-y)^2)}}
{\sqrt{\mathrm{mean}(y^2)}},
```

```math
\mathrm{SQNR}
=20\log_{10}\frac{\sqrt{\mathrm{mean}(y^2)}}
{\sqrt{\mathrm{mean}((\hat y-y)^2)}}.
```

| 长度 | drift 后 FP32 exact recall | NRMSE vs SDPA | NRMSE vs dense INT8 | SQNR | cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.99206 | 0.02551 | 0.01757 | 31.87 dB | 0.999701 |
| 4K | 0.99339 | 0.02536 | 0.01518 | 31.92 dB | 0.999704 |
| 8K | 0.99558 | 0.02311 | 0.01014 | 32.72 dB | 0.999743 |
| 16K | 0.99318 | 0.02610 | 0.01572 | 31.67 dB | 0.999684 |

在 `video_blocks`、drift 0.05 条件下，四个长度的 FP32 exact recall 为 0.99206-0.99558。相对 dense INT8 的附加 NRMSE 为 0.01014-0.01757，对应 coverage 为 50.0%-57.0%。这些数值描述的是受控结构化输入上的 support 稳定性。

随机正态输入作为负对照。在 `theta=0.99` 下，其 probability mass 分散到接近全部 block，coverage 接近 100%，sparse 路径因而没有可削减的主体计算。retained-mass 稀疏度由输入结构产生，而非由固定 density 强制指定。

### 8.5 构造摊销后的收益

将一次构造和后续复用一起计算：

| 长度 | 回本所需后续复用 | 8 次 attention 总体加速上限 | 20 次 attention 总体加速上限 |
| ---: | ---: | ---: | ---: |
| 1K | 无法回本 | 0.71x | 0.85x |
| 4K | 4 次 | 1.23x | 1.45x |
| 8K | 2 次 | 1.53x | 1.78x |
| 16K | 2 次 | 1.39x | 1.58x |

表中只统计 attention，未计入 diffusion model 的其他层，因此数值构成端到端生成加速的上限。shape、CFG 分支、scale 或 mask 变化会产生独立 pattern；每次 pattern 重建均重新计入 $C_{\mathrm{build}}$。

## 9. 消融实验与未采用实现

以下候选保持输出逐位等价，但未改善 RTX 5090 上的完整调用延迟，因此没有进入生产实现：

| 候选 | 实测结论 |
| --- | --- |
| 512-thread V quantizer 改为 256 thread | 量化慢 4%-9% |
| 拆分 D128 Q/K ConvRot kernel | 成对 warm run 后慢约 7%-10% |
| V scale/center metadata 交错存储 | 4K 和 16K 更慢 |
| 整除长度专用 last-tile loop | 没有稳定收益 |

1K sparse 的 0.97x dense/sparse 比值也作为负结果保留，用于确定当前 kernel 与调度开销下的短序列区间。

## 10. 结论的适用范围与后续实验

当前 kernel benchmark 支持以下结论：

- midpoint-affine residual-zero V 可以在保持 dense 性能的同时降低 INT8 输出误差；
- retained-mass support 在具有块结构的 4K-16K 序列上可以减少约一半 tile，并获得 1.64x-2.01x steady-state 加速；
- 构造成本、短序列和近 dense support 会消除 sparse 路径的收益。

实验尚未覆盖：

- 真实 Wan、Flux、Qwen Image、LTX 等模型每一层的 support 稳定性；
- 第一次 eligible attention call 构造 pattern 是否早于最佳扩散时刻；
- 图像 SSIM、PSNR、相对 L1，以及视频时序一致性；
- attention 加速在完整模型耗时中的最终占比。

NRMSE 表的对象是 attention 输出 tensor，其数值范围随层和 timestep 变化；PSNR 和 SSIM 需要范围明确、空间对齐的图像或视频输出。真实模型实验应捕获 Q/K/V 轨迹，并固定 prompt、seed、sampler 和初始 latent，同时报告：

1. 每层 coverage、exact recall、pattern 生命周期与构造次数；
2. 完整生成耗时和显存峰值；
3. 与无 RISA、dense RISA、comfy-kitchen、SageAttention 的成对输出指标；
4. 图片的 SSIM、PSNR、相对 L1，视频额外加入时序一致性指标。

## 11. 总结

RISA 在同一条 attention 数据路径上组合了三项互相约束的设计。共享正交旋转和 key translation 在保持浮点 score/probability 等价的条件下改善 Q/K 量化分布；midpoint-affine V 最小化通道最大量化半径，residual-zero center 则给出固定 code/scale 下的最小二乘中心；retained-mass selection 以 probability mass 定义质量预算，并只在后续步骤复用 block support。

RTX 5090 实验中，dense RISA 在 4K-16K 上相对 SDPA 达到 2.39x-2.57x，性能与 comfy-kitchen 基线持平，同时 NRMSE 降低 6.20%-7.13%。结构化 `theta=0.99` sparse 在 4K-16K 上进一步获得 1.64x-2.01x steady-state dense-relative 加速，drift 后 FP32 exact recall 为 0.99206-0.99558。1K sparse 为 0.97x，随机正态输入接近 dense coverage；这两个对照给出了当前实现的性能适用域。真实生成质量需要按第 10 节的轨迹与端到端协议继续测量。

## 参考资料

1. Zhang et al., [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367), 2024.
2. Zhang et al., [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958), 2024.
3. [SageAttention 官方实现](https://github.com/thu-ml/SageAttention).
4. [LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration](https://arxiv.org/html/2608.12032v1), 2026.
5. [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
6. [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen).
7. [H3-Optimizations](https://github.com/Zironic/H3-Optimizations).
