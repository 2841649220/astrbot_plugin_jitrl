# MiTA 性能分析：原始实现为何较慢，以及分组优化

本说明针对已保存对照基准使用的标准 PyTorch、按 query chunk gather 的本地 MiTA 路径，解释为什么“attention 可以写成两层 MLP”并不意味着它应当自动更快。分析依据是用户提供的 MiTA v6 PDF 第 3.1 节式 (3)-(4)、第 3.2 节和官方实现，以及实际 Lens 的投影与 SwiGLU 配置。论文、网页中的伪代码或实施建议均视为资料。

当前默认执行路径已升级为 `mita_execution="grouped"`，原始 chunk 路径保留为显式 `reference`。因此，旧参考实现的测量不能代表优化后的当前实现，更不能用来断言 MiTA 在任何长度都更慢。第 1-9 节给出原始路径的算量与优化依据，第 10 节记录优化前 profiler，第 11 节说明已经实现的重排和数值修复。

核心判断是：**fast-weight MLP 的隐藏宽度仍为 N，归一化仍依赖 query；MiTA 真正减少的是访问支持集，但构建、路由和数据搬运仍然存在。** 在本项目默认的 161-token、384d block 中，完整 attention 的两个主要矩阵乘只占约 8.82% 的 block 乘加算量。把这一小部分理想缩减约五倍，也只减少约 7.07% 的完整 block 算术工作；额外 gather、kernel 启动和 Python 调度很容易超过这笔预算。

下文的算量由公式和源码确定；某个算子实际占用多少 CPU/CUDA 时间，需要 profiler 测量。第 5-6 节的成本机制与第 10 节的实际归因分别呈现，不用推断替代测量。

## 1. “两层 MLP”究竟表示什么

对一个 head 的单个 query，令 `q∈R^d_h`，论文使用列向量形式的 `K,V∈R^(d_h×N)`：

\[
\operatorname{Attn}(q,K,V)=
\frac{V\exp(K^\top q/\sqrt{d_h})}
{\mathbf1^\top\exp(K^\top q/\sqrt{d_h})}.
\]

论文式 (3)-(4) 把它写成指数激活的 fast-weight MLP：

\[
W_1=K/\sqrt{d_h},\quad
Z(q)=\mathbf1^\top\exp(W_1^\top q),\quad
W_2(q)=V/Z(q),
\]

\[
\operatorname{Attn}(q,K,V)=W_2(q)\exp(W_1^\top q).
\]

这是一种表示和设计视角，没有删除原始计算。第一层产生 N 个激活，第二层汇总 N 个 value；`W_1、V` 来自当前上下文，`W_2` 的归一化又依赖当前 query。指数、求和和除法仍对应 softmax，隐藏宽度也没有从 N 变成某个固定小常数。

单 query 的宽度图是 `d_h → N → d_h`：第一步计算对 N 个 keys 的响应，第二步沿同一 N 维汇总 values。普通固定宽度网络则是 `d_h → f → d_h`，f 由架构选定、慢权重在输入之间复用。这两个图都可以称为“两层”，但层数没有说明中间宽度、权重生成方式和归一化成本。

| 对象 | 隐藏宽度与“权重” | 单 query 主要成本 |
|---|---|---|
| 普通固定宽度两层 MLP | 固定 f，慢权重不随输入上下文变化 | 约 `2 d_h f` 次 FMA |
| 完整 attention 的 fast-weight MLP | 宽度 N；K/V 来自当前上下文，归一化依赖 q | 约 `2 N d_h` 次 FMA，加归一化 |
| MiTA 的共享与 routed 支持 | 每个 query 访问 m+k 项，另需从 N 项构建专家 | 访问变少，构建不为零 |

因此，N 个 query 运行该 N-width fast-weight MLP 仍有 `O(N² d_h)` 交互成本。把名称从 attention 改成 MLP 不会使这两次乘法变成 `O(N d_h)` 的固定总成本。若真的换成普通小宽度、只在单个 token 内运行的 MLP，还需要另一个机制传递不同事件和经验之间的信息；计算和模型能力已经改变。

MiTA 的实际贡献是通过 landmark compression 与可复用 top-k KV experts 限制 fast-weight 的访问宽度，并保留共享概况。这是一项近似与稀疏机制，而不是对完整 attention 做无损代数改名后自动获得加速。[MiTA v6，第 3.1-3.2 节](https://arxiv.org/abs/2602.01219v6)

## 2. GQA 减少的部分与未减少的部分

设模型宽度 d、query heads 为 Hq、KV heads 为 Hkv，`d_h=d/Hq`，`ρ=Hkv/Hq`。这里实际配置为：

```text
d=384, Hq=6, Hkv=2, d_h=64, ρ=1/3
N=161, m=8, k=16
SwiGLU 分支宽度 f=2d=768
```

Q 和输出投影的宽度仍为 d；K/V 各只投影到 `Hkv*d_h=128`。所以每个 token 的投影 FMA 为

\[
d^2+2d(Hkv\,d_h)+d^2=2d^2(1+\rho).
\]

相同 d 的普通 MHA 是 `4d²`，本配置 GQA 的投影乘加量为其 2/3；投影后的紧凑 KV 头容量为六头 KV 的 1/3。

但仍有六个不同 query heads，各自计算匹配和输出。完整 GQA 的 QK 与 AV 合计 FMA 仍为

\[
2HqN^2d_h=2N^2d.
\]

不能把这个二次项乘以 `Hkv/Hq`。共享 KV 投影与减少 query 的数量是两件事。当前对照路径还通过 `repeat_interleave` 把 KV 广播到六个 query heads；这不会恢复被省掉的 K/V 线性投影，但可能产生物理复制。优化 kernel 可以保留紧凑 KV，具体仍取决于实现。官方 FlashAttention 接口也明确区分 query heads 与更少的 KV heads。[官方 GQA 接口](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)

## 3. MiTA core 的完整乘加账目

以下为 batch 1；B>1 时乘 B。一个 FMA 记为两个 FLOPs。这是通用矩阵乘常用的算术计数，不能将 CPU/CUDA 的一次指令、exp 的代价或访存字节等同于一个 FLOP。padding 仍占用物理张量槽位时按物理 N 计数。

共享 landmark 查询完整 K 的 `mN` 阶段不能省略。当前实现将 query→landmark 的 logits 同时用于路由与最终共享分支，避免再次计算这一项：

| MiTA 阶段 | FMA | 本配置 FMA |
|---|---:|---:|
| Landmark 与完整 K 匹配 | `mNd` | 494,592 |
| Landmark softmax 汇总完整 V | `mNd` | 494,592 |
| Query 与 landmarks 匹配，供路由/共享 logits 复用 | `Nmd` | 494,592 |
| Query 与被路由的原始 K 匹配 | `Nkd` | 989,184 |
| 最终共享 weights 汇总 landmark values | `Nmd` | 494,592 |
| 最终 routed weights 汇总原始 V | `Nkd` | 989,184 |
| 合计 | `2Nd(2m+k)` | 3,956,736 |

完整 GQA 的 QK、AV 各需 `N²d=9,953,664` FMA，合计 19,907,328 FMA。因此，两种 core 分别为：

```text
Full GQA core: 39,814,656 FLOPs
MiTA core:      7,913,472 FLOPs
理想算术比：161/(2*8+16)=5.03125
```

这里还没有计算 pooling、mask、缩放、top-k、argmax、softmax、索引、gather、输出 scatter 或框架调度。MiTA 的 top-k 至少需要扫描每个 landmark 的 N 个候选，选择算法还有自己的成本；不能只计 `Nk` 最终访问就断言总成本。

完整 attention 需要约 `HqN²=155,526` 个 softmax 指数项；MiTA 的压缩与最终 joint softmax 共约 `HqN(2m+k)=30,912` 项，另外有路由 argmax。归一化工作也减少了，但并未消失。两分支仍必须共同归一化，不能为提速换成独立 softmax 后相加。

## 4. 放回完整 block：最大的计算并未被替换

实际 `SwiGLUBlock` 使用 `Linear(d,4d)` 后拆成两个宽度 `f=2d` 的分支，再做 `Linear(2d,d)`。相当于 gate、value、down 三个投影，而不是两个宽度很小的线性层。

在 384d 配置中，gate 与 value 各自都是 `384 → 768`，合并实现为一次 `384 → 1536`，拆分后做 SiLU 与逐元素乘，最后 `768 → 384`。这里的 768 是通道宽度，不是 attention 的 token 数 N。其 FMA 为：

\[
N(2df+fd)=3Ndf=6Nd^2.
\]

完整 block 的主要矩阵乘计数如下，暂不包括 norm、bias、激活、残差以及 MiTA 额外操作：

| 部分 | FMA | FLOPs |
|---|---:|---:|
| Q 投影 | 23,740,416 | 47,480,832 |
| K 投影 | 7,913,472 | 15,826,944 |
| V 投影 | 7,913,472 | 15,826,944 |
| 输出投影 | 23,740,416 | 47,480,832 |
| Q/K/V/O 小计 | 63,307,776 | 126,615,552 |
| SwiGLU 三个投影 | 142,442,496 | 284,884,992 |
| Full GQA core | 19,907,328 | 39,814,656 |
| MiTA core | 3,956,736 | 7,913,472 |
| Full GQA block 合计 | 225,657,600 | 451,315,200 |
| MiTA block 合计 | 209,707,008 | 419,414,016 |

默认短序列下，SwiGLU 的主要乘加本身约为完整 attention core 的 7.16 倍。“MLP 总比 attention 便宜”在这一具体宽度、长度下也不是正确的算量判断。

若假定各部分的吞吐完全相同、非 attention 时间保持不变、MiTA 没有额外成本，则完整 block 的理想算术加速为

\[
451{,}315{,}200/419{,}414{,}016\approx1.0761.
\]

这相当于减少约 7.07% 的主要矩阵乘工作。若 attention core 完全免费，在同一简化假设下也只有

\[
451{,}315{,}200/(126{,}615{,}552+284{,}884{,}992)
\approx1.0968.
\]

这是**按乘加量加权的 Amdahl 估算**，不是未经测量就成立的真实运行时上界。真实公式应使用 profiler 的时间比例 p、core 加速 s 和新增开销相对比例 o：

\[
S=1/[(1-p)+p/s+o].
\]

attention 与 MLP 的算力利用率、带宽需求和融合程度不同，因此 `p≠8.82%` 完全可能；8.82% 只是这里的 FMA 比例。若 `o` 已超过被节省的时间，实际 S 就低于 1。完整 Lens 还包含 encoder、Engram、compiler、adapter 与输出头；远端请求又是另一个计时边界。

固定 d、m、k，只改变 N，同一算术估算如下：

| N | Full attention core 占 block 主要乘加比例 | MiTA 相对 full-GQA 的理想 block 加速 |
|---:|---:|---:|
| 32 | 1.89% | 1.000× |
| 128 | 7.14% | 1.057× |
| 161 | 8.82% | 1.076× |
| 512 | 23.53% | 1.283× |
| 1024 | 38.10% | 1.585× |
| 8192 | 83.12% | 5.811× |

core 算术开始减少的简单条件是 `N>2m+k`，本例为 N>32；实际时延交叉点还必须偿还额外开销。不能把这个条件当作 N=33 起就会更快的预测。

本表是原始 reference 复用共享 logits、无额外 packing padding 时的算术估算。当前 grouped SDPA 的实际计算略有不同：路由先算一次 `Q·landmark`，随后 SDPA 在拼接支持集上重新计算共享 logits。若 P 是所有 batch/head 的 packed query 槽位总数，其主要 FMA 为 `3 B Hq N m d_h + 2 P(m+k)d_h`；完全没有 padding 时为 `B Hq N d_h(5m+2k)`。默认例子的 grouped core 因此约 8.903 MFLOPs，理想 block 比约 1.074×；packed 空槽还会增加工作。它可以用略多算术换取更少搬运和更好的执行效率，不能继续套用 reference 的精确操作数。

## 5. 为什么稀疏 eager 路径可能输给完整 SDPA

`GroupedSelfAttention` 调用 PyTorch SDPA，由框架根据设备、dtype、mask 和形状选择实现。CUDA 上可能选择融合路径，也可能因限制而回退；必须检查真实 kernel，不能只看到 SDPA 函数名就断言使用了 FlashAttention。[PyTorch 2.8 SDPA 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)

FlashAttention 的要点是精确 attention 的分块与 IO 优化：减少 HBM 与片上存储之间的读写，而不是删除 `N²d` 的数学交互。它可以不把完整 attention 矩阵物化到 HBM。用“稀疏 score 项更少”直接推断比该实现省显存，忽略了基准已经避开的临时数据。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

对照 MiTA 路径逐个执行 pooling/scatter、两个较小矩阵乘、top-k、argmax、多次 mask、gather、乘法、reduction、cat、softmax 和每个 query chunk 的循环。以下都是需要测量的额外成本候选：

- 小矩阵或 elementwise 操作不一定达到稠密 GEMM 的吞吐，减少算术量也可能同时降低硬件利用率。
- Python 循环与每个独立 CUDA kernel 的启动都有固定成本；N=161、C=64 已需要三轮 chunk，四个 block 还会重复该流程。
- KV 的非连续 gather、临时张量、内存分配和 autograd 保存值消耗带宽；top-k 不会因输出很短就不扫描输入。
- KV 头广播、dtype 转换、布局变化或不连续张量可能引入额外复制，不能仅由张量的逻辑 shape 判断是否免费。
- 首次编译/初始化、线程池调度和 profiler 自身都有成本，冷启动与稳定阶段应分开。

这些陈述解释成本来源的机制，不表示已经测出某一项占总延迟的具体百分比。

## 6. 路由 KV 的重复搬运比 score 数更值得检查

逻辑上同一个 expert 的 KV 可供许多 query 复用。原 chunk 实现却对每个 query gather 一份被选中的 K/V，分别产生 `[B,Hq,C,k,d_h]`；随后以逐元素乘法和 sum 计算 query-local 匹配。它没有利用“一个 expert 对多个 query”的矩阵乘共享方式。

按 float32、B=1、N=161、Hq=6、d_h=64、m=8、k=16 计算，以下是**逻辑张量字节数**，不是测量到的 allocator 峰值：

| 张量 | 容量 |
|---|---:|
| 若显式物化的一份完整 self-attention scores `[Hq,N,N]` | 0.593 MiB |
| 可复用 expert K/V 两份 `[Hq,m,k,d_h]` | 0.375 MiB |
| 单个 C=64 chunk 的 routed K/V 两份 | 3.000 MiB |
| 全部 query 各复制 routed K/V 的累计两份容量 | 7.547 MiB |

训练 autograd 可能保留不同 chunk 的 gather 或相关中间值直到 backward；chunk 限制瞬时工作集，并不保证训练图只保留一个 chunk。N=1024 时，全部 query 的两份 routed KV 容量达到 48 MiB，而一份显式稠密 scores 为 24 MiB；融合 SDPA 又不一定需要保存后者。

因此，没有完整 `N×N` self-score 张量与低显存不是同一个结论。具体峰值还取决于哪些张量同时存活、反向保存策略、allocator 和 fused kernel，应测量而非将这些容量相加冒充实测值。

## 7. 保持 MiTA 算法的优化方向

当前 grouped 路径已经采用按 `(batch, query_head, expert_id)` 对 query 分组，复用同一 expert 的 KV；其执行重排为：

1. 保留原 landmark pooling、压缩与 top-k；不改变路由决定。
2. 依据 `argmax` 结果排序或打包 query，生成每个 expert 的 query 组 `Q_g`。
3. 为该组准备一次 `[landmark_keys; original_expert_keys]` 及对应 values。
4. 用矩阵乘或合适的 SDPA/变长内核计算组内所有 query，最后恢复 query 原顺序。

这样可以避免按 query 重复搬运相同的 routed KV，并把许多 elementwise dot 改成更适合硬件的批量矩阵工作。代价是排序、打包、输出 scatter、空组和负载不均；小 N 时不保证净收益。若为每组 padding 到最大 query 数，严重不平衡也会浪费工作。

如果共享与 routed 分支分别计算，必须保留各自 log-sum-exp，并按联合分区函数合并：

\[
O=\frac{e^{L_s}O_s+e^{L_r}O_r}{e^{L_s}+e^{L_r}},
\quad L_s=\log Z_s,\ L_r=\log Z_r.
\]

可减去 `max(L_s,L_r)` 稳定计算。给每组直接拼接共享与 routed KV 后做一次 SDPA，则天然保持一个共同分母。等权相加两个已归一化输出会改变算法，不能作为等价加速路径。

论文 Algorithm 1 和[作者实现](https://github.com/QishuaiWen/MiTA/blob/95460f703ef5bd5d814c45a6ff52d9582207d01c/mita/mita_attention.py)采用按 expert 排列 query 与变长 attention；[MixedAttention](https://github.com/QishuaiWen/MiTA/blob/95460f703ef5bd5d814c45a6ff52d9582207d01c/mita/mixed_attention.py)负责联合归一化。这里可以借鉴该执行策略，同时保留本项目的 mask-aware 1D pooling、GQA 和论文标准缩放，明确记录与官方视觉代码的差异。

随后可检查避免物理 KV 头复制、融合 mask/softmax/reduction、稳定形状的编译以及减少中间值保存。它们都需要前向、有效梯度、padding 和 joint-normalizer oracle 对照。优化方向不构成已获得加速的事实，也不授权把默认 MiTA 静默换成 GQA 或 compression-only 模块。

## 8. 区分匹配基线与旧单 query 模型

新 token 主干的 full GQA 与 MiTA 都处理 `[state;events;memory]`，同样四个 attention+SwiGLU blocks、同样初始权重和投影参数，适合比较算子效率。

`legacy` 的 memory attention 只有一个 state query，主要 attention 交互是 `O(Kd)`；随后四个 MLP 只处理一个向量。新主干让 N 个 token 在每个 block 都交互和执行 MLP。将新 MiTA 与 legacy 的整体耗时相比，包含了 query 数、深度布局和模型容量变化，不能将全部差异归因于 MiTA 算子。

同样，“4/4 attention 使用 MiTA”仅描述本地新主干的层数占比，没有替换远端黑盒 LLM 的 attention，也没有减少远端生成本身的计算量。

## 9. 接下来如何把推断变成可复核证据

公平计时至少固定相同输入/权重、B/N/d、head 数、m/k/C、有效 mask、dtype、设备、线程和训练/推理模式。区分三个层级：不含投影的 core、含投影的 attention module、包含 MLP/encoder 等的完整 Lens；再单独记录端到端检索与远端调用。不要把一个层级的收益放大到另一个层级。

CUDA 执行是异步的，应使用设备 events 或在计时边界同步，分别记录冷启动、热身后分布和 forward/backward；相关要求见[PyTorch CUDA 异步执行说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#asynchronous-execution)。profiler 用于归因，另行计时用于报告性能，避免把 profiling 开销当作正常延迟。

应采集 SDPA 实际 kernel 名称，MiTA 的 compression、top-k、routing、gather、joint attention、projection/MLP 各阶段的 CPU/CUDA self time、调用次数和内存分配。确认低效点后再比较原 chunk 与 query-grouped 路径，并检查输出/梯度误差。测 allocator 峰值时区分 allocated、reserved、相对驻留基线增量和进程显存。

N 从短序列扩展到较长序列、m/k 固定与缩放、C 的扫描、不同 batch 和 padding 分布都要单独报告。较长 N 能提高算术节省的潜力，但无法预先保证实际交叉点。旧参考实现的原始结果位于 [CPU 基准](../benchmarks/mita-attention.json)、[CUDA float32 基准](../benchmarks/mita-attention-cuda.json)与[受控任务消融](../benchmarks/mita-memory-ablation.json)；第 10 节补充旧 reference 的归因，第 12 节报告当前 grouped 路径的同轮测量。

## 10. 优化前 profiler 已经确认的事实

[诊断脚本](../scripts/profile_mita.py)与[原始 profile JSON](../benchmarks/mita-profile.json)分别保存无 instrumentation 的计时和 Kineto CPU/CUDA 归因。它们记录的是历史 reference attention 源码 SHA-256 `eab67ada56298a1a6cb5000a50a5f298ed186fbefd11ea89e1bb5d1ec8b181ab`；该哈希与当前优化后源码不同是有意保留历史证据，而不是把旧测量冒充新实现。

这次 float32 CUDA profile 中，GQA 实际出现 `aten::_scaled_dot_product_efficient_attention`，GPU kernel 名包含 `fmha_cutlassF_f32...PyTorchMemEffAttention...`。因此可具体归因为 CUTLASS memory-efficient attention 路径，不能笼统称为使用了 FlashAttention。下表的 CUDA kernel 启动次数按两次采样的计数除以 2，归一到一次前向；不是把整份 capture 的总数误作单次：

| 原始模块，C=64 | N=161 的 CUDA kernel 启动/前向 | N=1024 的 CUDA kernel 启动/前向 |
|---|---:|---:|
| Full GQA | 20 | 17 |
| MiTA reference | 134 | 405 |

N=1024 的 reference 每次前向只调用一次 top-k，profile 的平均 self-device time 约 65.20 μs。对比同一份 instrumented capture 的算子族统计：gather 约 747.02 μs、mul 约 696.49 μs、sum 约 548.27 μs；这些数值按每次前向归一。CPU 侧还有大量 view/slice/mask 调度，profile 中 shape-view 族约 1,425 次调用/前向。这些事实支持“许多小操作和重复 gather/reduction 是值得优先优化的路径”，不支持“主要时间都花在一次 top-k”。算子族统计是归因信息，不应与无 instrumentation 的 wall latency 相减来计算所谓精确损失占比。

同一诊断协议的无 instrumentation CUDA 同步 wall 中位数为：

| N | 模块 | C | 同步 wall 中位数 |
|---:|---|---:|---:|
| 161 | GQA | — | 0.7304 ms |
| 161 | MiTA reference | 64 | 4.1616 ms |
| 161 | MiTA reference | 1024 | 2.6393 ms |
| 1024 | GQA | — | 2.3147 ms |
| 1024 | MiTA reference | 64 | 10.0012 ms |
| 1024 | MiTA reference | 1024 | 4.1825 ms |

N=1024 时仅增大 chunk，reference 的启动次数就从 405 降到 89，前向输出最大绝对差约 `5.96e-8`。这是在没有更换 MiTA 数学机制的情况下显著改变实际耗时的直接证据；但该版本仍慢于这轮 GQA，不能据此宣称只改 chunk 就解决了全部问题。CPU 上该扫描没有相同收益：C=64/1024 的中位数分别约 50.48/56.59 ms，说明 chunk 的选择也具有设备相关性。

CUDA events 记录的是 stream elapsed 区间，可能包含 CPU 两次提交间的 GPU 空闲；raw kernel device time 求和又是另一个量。instrumented CPU/CUDA self time、未插桩 host-dispatch wall、未插桩同步 wall 与 CUDA-event elapsed 必须分列。不能把不同运行模式或不同批次的数字互相相减后，当作已经测出的 Python 或 kernel 空闲时间。

## 11. 当前 grouped 执行与 FP16 稳定性修复

当前 `mita_execution="grouped"` 在 landmark 压缩、可复用 top-k KV 与硬路由之后，对相同 expert 的 query 排序并打包成 query blocks。每个 block 只准备一份共享+原始 expert KV，调用批量 SDPA 完成联合归一化，再将输出恢复到原 query 顺序。它保留原算法的支持集，没有采用完整 N×N attention 回退，也没有把共享与 routed 输出各自 softmax 后直接相加。

动态 packing 以一次标量尺寸同步确定 block 数；较长 CUDA 序列可以使用已知 block 数上界的 static packing，以 dummy expert 填充未使用块，避免该尺寸的主机往返。短序列采用较小的 query block，以限制空槽浪费。排序、填充、scatter 和 SDPA 重算共享 logits 仍然有成本，所以必须以当前 grouped 路径的同轮数据寻找与 GQA 的交叉点。优化前的慢速表不再是当前默认实现的性能结论；未归档的初测也不替代最终匹配比较。

这次重排还暴露了 FP16 landmark pooling 的数值问题：半精度 `scatter_add` 累加损失足够多有效位时，相同输入的重复前向可能改变临界 top-k 或 argmax 路由，离散选择会放大该差异。当前对 FP16/BF16 query 先用 float32 累加 pooled sums 和 counts，求均值后再转换回输入 dtype，修复所观察到的重复路由漂移。该修复不改变 landmark 窗口或数学平均定义，也不等于保证任意硬件、所有 tie 和所有 CUDA 算子都逐位确定。

当前优化结果应同时给出：reference→grouped 的同 Q/K/V 输出及梯度误差、当前 grouped 相对同轮 GQA 的前向/反向时延与内存，以及原始完整 block、设备和 dtype。不能把 reference→grouped 的优化倍数说成 grouped→GQA 的优势，也不能把一个较长 N 的收益迁移成默认短序列的保证。

## 12. 当前匹配基准：交叉点存在，但不在默认短序列

[基准脚本](../scripts/bench_mita.py)在同一次运行中复用相同的投影权重、输入、mask、参数量和设备，分别比较 fused GQA、MiTA reference 与当前 grouped MiTA。CUDA 每个阶段热身 3 次、计时 7 次并在样本边界同步；下表为 attention module 前向与前向+反向的中位数。硬件为 RTX 3050 Ti Laptop、PyTorch 2.8.0+cu128，`B=1,d=384,Hq=6,Hkv=2,m=8,k=16`。这些是合成输入的算子微基准，不是完整 Agent 请求。

| dtype / N | fused GQA 前向 / 前反向 | grouped MiTA 前向 / 前反向 | grouped 对 GQA |
|---|---:|---:|---|
| fp32 / 161 | 0.717 / 1.904 ms | 2.818 / 5.320 ms | 前向慢 3.93×，前反向慢 2.79× |
| fp32 / 1024 | 2.246 / 7.747 ms | 2.657 / 5.755 ms | 前向慢 1.18×，前反向快 1.35× |
| fp32 / 8192 | 79.024 / 254.151 ms | 9.145 / 25.680 ms | 前向快 8.64×，前反向快 9.90× |
| fp16 / 161 | 0.702 / 1.847 ms | 2.633 / 5.496 ms | 前向慢 3.75×，前反向慢 2.98× |
| fp16 / 1024 | 0.982 / 3.099 ms | 2.806 / 5.884 ms | 前向慢 2.86×，前反向慢 1.90× |
| fp16 / 8192 | 15.448 / 63.630 ms | 6.093 / 12.593 ms | 前向快 2.54×，前反向快 5.05× |

短序列上 grouped MiTA 的主要收益是相对原 reference 的执行优化，而不是胜过 fused GQA；N=1024 的 fp32 只在训练测量中反超。到 N=8192，固定大小的 landmark/expert 支持集开始明显压过稠密 attention 的二次项，才出现稳健的本机延迟交叉。该交叉点取决于 GPU、dtype、batch、padding、SDPA 内核、chunk 和路由分布，不能直接推导为其他机器或完整四层 Lens 的倍率。

这也解释“attention 可写成两层 MLP”与实测并不矛盾：这种 fast-weight MLP 的隐藏宽度是 N，query 依赖的归一化仍要处理注意力分数；默认 N=161 时 attention core 只有整个 block 主要矩阵乘约 8.82%，而 grouped 还要排序、打包、scatter，并在局部 SDPA 里重算共享 logits。将 core 的理论节省乘到整层，无法抵消这些固定工作。长 N 时稠密项按 N² 增长，收益预算才大得多。

相同基准还将整网输入特征预先生成，再分别运行四层完整 Lens forward。N=161、fp32 时四层 GQA 为 7.391 ms、四层 grouped MiTA 为 14.828 ms；fp16 分别为 8.180 ms 和 17.044 ms。计时包含网络前向与四个 attention block，但不包含检索、特征构建、训练或远端 API；它确认在当前实际工作负载长度下，attention-only 的开销会反映到整网前向。旧 `legacy` 只有一个 state-query attention，其总耗时混合了架构差异，不作为同容量 MiTA/GQA 比较。

CUDA profile 的平均 kernel launch 数/前向（两次 capture 归一）为：N=161，GQA/grouped/reference 为 20/123/132；N=1024 为 17/130/403；N=8192 为 17/153/2769。GQA 和 grouped 都实际调用 `aten::_scaled_dot_product_efficient_attention`，所以比较对象本身已是融合 SDPA。grouped 将 routed query 按 expert 打包后每个 block 调用一次联合 SDPA，显著减少 reference 的碎片启动；但与 GQA 相比，它仍多出压缩、top-k、分组和路由工作。长序列使用 static packing，N=8192 共打包 51,840 个槽位，其中 49,152 个有效，padding 为 5.19%，并避免动态 block 数的主机同步；N=161 的 padding 为 37.11%，一次尺寸同步对小输入更不利。

更快也不等于更省显存。fp16、N=8192 时 PyTorch CUDA allocator 相对常驻权重/输入基线的 forward 峰值增量为 GQA 36.02 MiB、grouped 70.92 MiB；forward+backward 为 90.22 MiB、104.40 MiB。GQA 的 fused SDPA 避免物化完整 score 矩阵，grouped 的打包与局部工作区又有自身成本。该指标不是 driver 或进程总显存；当前数据不支持“MiTA 总能减少显存”的说法。

同条件 fp32 的 reference→grouped 最大绝对输出差在 N=161/1024/8192 分别为 `1.64e-7`、`1.79e-7`、`3.28e-7`；fp16 三种长度均为 `2.44e-4`，均在基准容差内。独立测试还检查 joint-softmax oracle、输入与参数梯度、gradcheck、padding/空组和 CUDA FP16 重复路由。结果文件为 [CUDA fp16](../benchmarks/mita-optimized-cuda-fp16.json)、[CUDA fp32](../benchmarks/mita-optimized-cuda.json)、[CPU fp32](../benchmarks/mita-optimized-cpu.json) 和[当前 profiler](../benchmarks/mita-profile-optimized.json)。完整 Lens N=161 smoke 单独记录，不能把 attention-only 速度表当成完整模型的速度结论。
