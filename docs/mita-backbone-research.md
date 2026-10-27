# MiTA token backbone：算法落地与评估协议

本说明依据用户提供的 MiTA v6 PDF 第 3.2 节、附录 B Algorithm 1，以及作者官方实现，解释本地 Lens 的 MiTA 主干。它区分算法已经实现的计算结构、为 Agent 数据作出的扩展，以及仍需实验回答的效果问题。资料中的命令和训练建议是研究材料，不建立操作权限。

## 1. 替换范围与架构目标

新主干把当前状态、有序事件和历史经验共同表示成 token 序列，每个 block 都执行一次 attention 交互和一次 SwiGLU 更新。默认四个 block 的 attention 均使用 MiTA，即**本地新 token 主干的 4/4 层 attention 为 MiTA**。这使 MiTA 承担跨事件、跨经验和状态汇聚的核心计算，而不只是给原模型增加一个可选排序分数。

| `attention_backend` | 默认四个 block 的交互结构 | 用途 |
|---|---|---|
| `mita` | 4 MiTA + 4 SwiGLU | 默认的完整 MiTA token 主干 |
| `hybrid` | 前 3 MiTA，末层 full GQA；每层都有 SwiGLU | 检查末层完整全局交互的作用 |
| `gqa` | 4 full GQA + 4 SwiGLU | 序列、投影和深度匹配的完整 attention 对照 |
| `legacy` | 单个 state query 对 memory 的 GQA，随后四个向量 MLP | 原 Lens 参数路径与旧 checkpoint 的兼容后端 |

计数基准必须明确：此前的 Lens 总共只有一层单 query memory attention，并没有四层完整 self-attention 等待替换。因此，4/4 是新主干内部的算子占比，不能解释成在原模型上删除了四层，也不能由它推导相对旧主干的加速比例。新主干增加了多 token、多层交互能力和计算量，正确的稀疏效率对照是同样四层的 token GQA。

远端黑盒 LLM 的 attention、KV cache、参数和解码器均不在这一替换范围内。原有 PRE、候选 POST、经验记录与结算接口继续作为本地网络的输入输出边界；工具权限和执行校验的职责也不转移给 attention。

## 2. 为什么这次属于 MiTA 算法落地

原来的经验 compiler、低秩 fast adapter 和一次 memory GQA，分别完成经验汇总、条件调制和精确访问，但没有构建 landmark queries、共享压缩 KV、可复用的 top-k 原始 KV experts，也没有让每个 query 在共享专家和被路由专家之间共同归一化。它们只能归类为对“压缩与路由”思想的借鉴。

当前 `jitrl/neural/attention.py` 与 `jitrl/neural/mita_grouped.py` 实现以下完整计算链：

```text
输入状态 / 事件 / 经验 tokens
    └─ Q、分组 K/V 投影
         ├─ masked landmark pooling
         │    └─ landmark 查询完整 K
         │         ├─ softmax 汇总 V → 共享压缩 values
         │         └─ top-k 原始 K/V → 可复用 experts
         └─ 每个 query 与 landmarks 匹配 → 硬路由一个 expert
              └─ 共享压缩 KV + 被路由原始 KV
                   └─ 一次 joint softmax → 更新 tokens
```

“原始 KV”在这里指当前本地 block 的 K/V 投影，绝不是远端 LLM 的 KV cache。expert 随输入重新构建，没有各自独立的一套 MoE 慢权重。所有查询到同一个 expert 的 query 复用同一组 top-k 索引，这与为每个 query 独立搜索 top-k 不同。

## 3. 论文公式到张量路径

以下省略 batch 和 head，采用 token 位于行的记号：`Q,K,V∈R^(N×d_h)`。论文采用转置记法，算法相同。所有 padding 都通过 `valid` mask 处理。

### 3.1 Landmark queries

将有效 query 按原顺序分成 `m` 个窗口，得到

\[
\widetilde q_j=\frac{\sum_{i\in W_j} M_i q_i}
{\max(1,\sum_{i\in W_j}M_i)},\qquad
\widetilde M_j=\mathbf1[\sum_{i\in W_j}M_i>0].
\]

`masked_landmarks` 对有效位置计算 rank。有效数量为 `L` 时，rank `r` 属于 `floor(r*m/L)` 号窗口。空窗口无效，不作为额外零向量参与注意力。该定义保证插入或删除 padding 不改变有效 token 的窗口归属。

### 3.2 用同一份 scores 压缩与构建 experts

\[
S_{ji}=\widetilde q_j^\top k_i/\sqrt{d_h},\quad
\widetilde v_j=\sum_i\operatorname{maskedsoftmax}_i(S_{ji})v_i,
\]

\[
I_j=\operatorname{TopK}_i(S_{ji}),\quad
K_j=K[I_j],\quad V_j=V[I_j].
\]

这对应论文式 (6)-(8) 和 Algorithm 1 第 4-9 行。`compression_scores` 同时用于压缩和 top-k，`landmark_values` 表示所有已知有效 tokens 的共享概况；expert 保留被选中原始 KV 的信息。实际 top-k 宽度为 `min(k,N)`，不足 k 个有效 token 的槽位继续保持 mask。

共享 expert 的 keys 是 `landmark_queries`，values 是 `landmark_values`。把压缩后的 value 再当作 key，或者用 memory importance 标量替代 landmark 与 K 的匹配，都改变了算法。

### 3.3 每个 query 硬路由一个 expert

\[
R_{ij}=q_i^\top\widetilde q_j/\sqrt{d_h},\qquad
e(i)=\arg\max_{j:\widetilde M_j=1}R_{ij}.
\]

对应论文式 (5)、式 (7) 后的路由定义和 Algorithm 1 第 13 行。缩放是正的常数，不改变 argmax。`s=1` 与论文的实际实现设置一致；本分支没有冒称实现任意 s 的多 expert 路由。

### 3.4 拼接后共同归一化

\[
K_i^\star=[\widetilde Q;K_{e(i)}],\qquad
V_i^\star=[\widetilde V;V_{e(i)}],
\]

\[
o_i=\operatorname{maskedsoftmax}
\left(q_iK_i^{\star\top}/\sqrt{d_h}\right)V_i^\star.
\]

这就是论文式 (10)-(12)。`reference` 路径复用 `routing_scores` 作为共享分支的 logits，再计算被路由原始 KV 的 logits；拼接两者后调用一次 softmax。默认 `grouped` 路径按 expert 打包 query，拼接共享与原始 expert KV 后调用一次融合 SDPA；它重算共享 logits，但保持同一支持集和共同分母。

若分别计算共享输出 `O_s` 和路由输出 `O_r`，必须保留两个分区函数：

\[
O=\frac{Z_sO_s+Z_rO_r}{Z_s+Z_r}.
\]

简单相加、等权平均，或再加入一个任意学习的混合门，都不等价。论文 Algorithm 1 的 `Combine` 使用 online softmax，本实现直接对拼接分数共同归一化，实现同一数学目标。

原始 token 同时出现在压缩摘要的来源与某个原始 expert 中，是该支持集设计的一部分。即使设 `k=N`，还存在 m 个压缩 KV，通常也不会退化为 full attention；正确 oracle 是上述拼接公式，不能用 full-attention 相等性代替算法验证。

## 4. Agent 序列与论文设置的差异

输入序列为 `[state token; ordered event tokens; memory tokens]`，使用类型标识区分三种对象，事件另带位置表示。默认预算下 `N≤1+32+128=161`。`memory_tokens=128` 表示可交互的内部历史经验容量，与渲染少量 PRE evidence 的数量和文本预算分开；128 条经验不会全部被写进提示。

事件包含当前决策以前已知的工具结构与观察，memory 包含当前可见的历史已结算回报。这里是对已知信息做判别式编码，可以在有效序列内双向交互；它不是自回归语言模型的未来 token 预测。避免标签泄漏仍依赖历史时间与快照合同，不能单靠 attention mask 解决。

本实现保留 MiTA 的核心机制，同时作出明确扩展：

- 作者官方视觉模块使用 2D spatial pooling，并带图像 token 布局假设；本地采用 1D 有效 rank 窗口。它保留局部有序压缩，但在不可整除长度上不等同于可能产生重叠窗口的 adaptive pooling。
- 本地采用 6 query heads / 2 KV heads 的 GQA 投影，再广播到 query heads。每个 query head 分别生成自己的 landmarks、experts 和路由；共享的是 KV 投影，不是假定六个头的 top-k 索引相同。
- 本地增加 padding、空窗口和空样本处理。无有效支持的行输出零，不在全 `-∞` 上直接运行 softmax。mask 也必须覆盖被 top-k 补齐的无效 KV。
- 使用标准 PyTorch 与 autograd；默认按 expert 打包 query、复用 KV 并调用局部融合 SDPA，保留逐 query chunk gather 的 `reference` 对照。不依赖官方 FlashAttention 2.6.3 的变长 CUDA 内核，也不宣称获得相同吞吐。

官方代码还存在一个应公开的数值差异：`mita_attention.py` 将 `router * d_h^-0.5` 传为共享 keys，而 `MixedAttention` 内部再次使用 `d_h^-0.5`。这使共享分支比字面论文公式多一层平滑。本分支采用论文式 (10)-(12) 的统一标准缩放，属于基于论文核心的独立实现，不能宣称逐位复刻官方数值路径。

## 5. 梯度、可学习性与路由诊断

`topk(...).indices` 和 `argmax` 是离散选择，普通 PyTorch 不对索引决定反向传播；本实现没有使用 straight-through estimator，也没有把路由伪装成可微采样。

有效梯度仍通过共享压缩的 softmax、landmark pooling、共同归一化、被选中的原始 K/V、输出投影和残差路径传播。一个未被某个 routed expert 选中的 token，也可能通过共享压缩获得梯度。因此硬路由不意味着整个模块无法训练，但也不保证 experts 会自动形成良好分工。

应检查有效 query 的 expert 使用率、最大 expert 占比、各 head 路由差异、原始 KV 覆盖率、共享分支与 routed 分支质量占比，以及不同经验来源的被选比例。长期只使用少数 experts、所有 head 总是选择同一组 KV、或者几乎全部质量落在共享分支，都值得与对照实验一起解释。诊断结果是行为观察，不是任务提升证明。

梯度数值对照应在 top-k 与 argmax 排序间隔明确的输入上进行。排序边界处的有限差分包含离散路径切换，不能据此要求不存在的索引导数。相同分数的 tie 也可能导致不同设备的索引顺序不同，应将随机种子、设备和 dtype 一并记录。

## 6. 原始 reference 的复杂度与当前 grouped 执行

设 batch 为 B、query heads 为 H、head width 为 d_h，m 为 landmarks，k 为 expert 宽度，C 为 query chunk。下表先列原始 `reference` 路径，不能直接当作当前 grouped 的物理执行量。每层的主要工作包括：

| 阶段 | 主要计算或容量 |
|---|---|
| Landmark pooling | 约 `O(B H N d_h)` |
| Landmark 查询完整 K | `O(B H m N d_h)`，产生 `[B,H,m,N]` scores |
| 压缩 V | 额外 `O(B H m N d_h)` |
| 每个 landmark 的 top-k | 必须扫描其完整候选轴；选择算法另有成本，不能算免费 |
| Query 到 landmarks | `O(B H N m d_h)`，logits 同时供共享 attention 使用 |
| 路由原始 KV 的 query logits | `O(B H N k d_h)` |
| Joint weights 汇总 values | `O(B H N(m+k)d_h)` |
| 可复用 expert K/V | 每份约 `[B,H,m,k,d_h]` |
| Chunk gather 的路由 K/V | 每份约 `[B,H,C,k,d_h]`；有非连续访存和复制成本 |

因此，完整分析必须包含 `mN` 构建阶段，不能只报最后 `N(m+k)` 的访问成本。若 m、k 固定且 N 增长，attention 交互相对 N 近似线性；若把 m 也设成随 N 线性增长，则该阶段仍可回到二次规模。Q/K/V 和输出投影，以及每个 token 的 SwiGLU 计算，也不因稀疏访问消失。

由于 reference 复用共享 logits，一层形成的主要点积分数数目为

\[
BHN(2m+\min(k,N)),
\]

而 full GQA 为 `BHN²`。以预算上限 `N=161,m=8,k=16` 为例，单 head 分别为 5,152 和 25,921 个分数；这是算术数量，**不是端到端 FLOPs 或速度测量**。若 `N=8,m=8,k≥8`，则分别为 192 和 64，MiTA 的分数工作反而更多。

默认 grouped 路径在 SDPA 内重算共享 logits，并对 packed 空槽计算。若 P 为所有 batch/head 的 query 槽位总数，其 score 构建量为 `2BHNm+P(m+k)`，主要 FMA 为 `d_h[3BHNm+2P(m+k)]`。它用额外的部分算术和 padding 换取更少 KV 搬运与更好的矩阵执行；两种路径的实际操作数需分开报告。

`score_elements` 描述主要点积分数构建量，`peak_score_elements` 描述最大单个 score 张量的逻辑容量。它们都不等于 allocator 峰值：compression scores、weights、routing scores、expert vectors、chunk gather 和输出可以同时存活，融合 SDPA 也未必物化完整分数矩阵。GQA 的 KV 广播也会增加实际内存。短序列开销、长序列实测交叉点与完整算量见 [MiTA 性能分析](mita-performance-analysis.md)。

Chunking 限制推理时单次 gather 的临时规模，但训练 autograd 可能保留各 chunk 的中间值直到 backward，累计保存量仍可随 `B H N k d_h` 增长。不能用 C 替代 N 宣称训练峰值已经固定，也不能从“没有显式完整 N×N self-score 张量”推出一定省内存。

## 7. 数据兼容与 checkpoint 迁移

动作记录、结算事件、来源标记、PRE/POST schema、scope 与冻结检索表示继续沿用既有合同。新 backbone 可以重放旧经验，但必须按记录原有快照读取：旧快照只有 32 条就使用这 32 条，不能为了凑满 128 条从更新后的库回填未来经验。

数据版本与模型架构版本需要分开。仅更换本地 attention 不应无故改写历史 normalizer、embedding 或 renderer 身份；但 checkpoint 必须绑定 backend、block 数、head 数、landmark/top-k/chunk 配置、memory token 预算及相关参数布局。旧向量 Lens 与新 token 主干不是同构网络，不能靠 `strict=False` 静默吞掉缺失权重并声称已迁移成功。

`legacy` 后端保留原参数命名和计算路径，供旧模型明确加载。MiTA、hybrid 与 full-GQA checkpoint 需要明确区分；切换后端后重新训练或显式选择对应 checkpoint。拒绝不兼容权重与删除原始 checkpoint 是不同操作，不应自动销毁旧训练结果。发布新模型仍通过不可变文件与原子切换，在途决策继续绑定原 checkpoint 与原快照。

扩大内部 memory token 预算并不扩大 scope 权限。跨用户检索、跨 scope 训练、未结算回报进入输入、未来 outcome 回流和已删除记录通过旧权重复现，仍由原存储、训练与 reset 机制约束。

## 8. 验证与有意义的实验

算法验证应首先证明当前代码执行论文机制：与逐 query 显式拼接 `K*、V*` 的小型 oracle 对照前向及有效梯度；分别验证共享压缩、top-k 索引、query 路由和共同归一化。改变两分支分区函数规模的输入能识别错误的独立 softmax 相加。还要覆盖任意位置 padding、全空行、`m>L`、`k>L`、ragged batch 和不同 C，并验证 padding 不改变有效 token 的输出。

主干层面的验证包括：默认实例确有四个 MiTA blocks；hybrid 与 GQA 的层数正确；每层有有效梯度；参数保存重载保持输出；旧模式仍可加载自己的 checkpoint；新旧架构互不误加载；历史快照与 ProviderRequest 临时上下文合同不变。仅通过 shape、随机前向或训练损失下降，不能代替上述机制测试。

随后用相同任务与时间切分分别回答两个问题：新增 token 主干是否改善决策，以及在同等主干中 MiTA 能否更有效利用容量。对照至少包含：

1. 原 legacy Lens，控制旧方案本身的变化。
2. 四层 full GQA，匹配 token 集、d、head、MLP、训练数据和训练步数。
3. 四层 MiTA 与三层 MiTA + 一层 GQA，评估末层全局访问的取舍。
4. Compression-only 与 routing-only，验证两分支组合的实际贡献。
5. 固定 token 预算下扫描 m、k；固定 m、k 下扫描 32/64/128 memory tokens；C 的扫描单独研究实现成本。
6. 去掉 Engram、compiler、fast adapter 的独立消融，避免把整个网络增加容量的收益全部归给 MiTA。

测量应包含完整 operator 构建与 gather 的 forward、forward+backward 延迟，热身后多次重复的分布，实际分配器或进程内存，以及主干和端到端请求延迟。报告 m、k、C、有效 N、padding、B、dtype、线程数、PyTorch 版本和设备。逻辑容量估算与实测峰值分别呈现。

任务效果用可验证成功率、回报、额外 API 成本、PRE 提示量、无支持弃权、错误标签干扰、新工具与旧任务遗忘衡量。使用只在过去数据上训练的 checkpoint 评估后续时间窗口；检索时间正确但模型已看过未来标签，仍然属于泄漏。解释模型参数、有效交互数量和外部库规模的差异，不把“4/4 使用 MiTA”直接当作胜过 full GQA 的证据。

## 9. 资料定位

- 用户本地 `MiTA.pdf`，17 页，第 3.2 节（第 5-6 页）和 Appendix B Algorithm 1（第 16 页）；第一页标注 arXiv:2602.01219v6，2026-09-27。SHA-256：`18aeca2c72f3632a220685f1d8a346c4e640e5cc37f222c9b3c77bb913b5e9d3`。公开版本：[论文 v6](https://arxiv.org/abs/2602.01219v6)。
- [作者官方仓库](https://github.com/QishuaiWen/MiTA)，读取时 `main` commit 为 `95460f703ef5bd5d814c45a6ff52d9582207d01c`（2026-10-07）。公式与代码的区别按固定提交核对，不以持续变化的 README 替代论文。
- [官方 MiTA 模块](https://github.com/QishuaiWen/MiTA/blob/95460f703ef5bd5d814c45a6ff52d9582207d01c/mita/mita_attention.py)：pooling、landmark 压缩、top-k、s=1 路由与共享 key 平滑。
- [官方 MixedAttention](https://github.com/QishuaiWen/MiTA/blob/95460f703ef5bd5d814c45a6ff52d9582207d01c/mita/mixed_attention.py)：两分支的 log-sum-exp 合并与对应反向传播。当前实现按论文公式独立构建标准 PyTorch 张量路径。

## 10. 已完成实验的实际结果

已按预先确定的两种种子、三种主干协议运行受控记忆任务，并在当轮 model/config/learner 源码冻结后原样重跑（原始 reference 内核，具体源码哈希保留于 JSON）。任务使用十个 code，每个回合随机改变五个 alpha / 五个 beta 的绑定；snapshot 中包含二十条已结算历史记录和打乱顺序的干扰项。配对样本具有相同当前状态与实际执行动作，但历史 rulebook 互为补集、观察回报相反，因此固定的 state→action 策略不能超过配对的 50% 排序基线。训练只接受实际执行动作的 ±3 回报；完整候选真值只用于评估。

每种种子使用 128 条训练样本、64 条未见完整绑定赋值的评估样本；词汇表共享，回合、经验 ID 与完整绑定赋值不跨训练/评估。网络为四层、hidden 48、embedding 64、m=4、k=4，100 次更新、batch 16、学习率 0.001、CPU 两线程。三个后端每种种子的初始参数逐字节相同，均为 129,082 个参数；训练后都有非零参数变化和训练损失下降。以下是两种种子的描述性均值，MSE 使用原始回报单位的平方：

| 主干或基线 | 留出观察回报 MSE | 留出候选排序准确率 |
|---|---:|---:|
| 四层 MiTA | 11.5301 | 47.66% |
| 四层 full GQA | 9.0012 | 50.00% |
| 三层 MiTA + 一层 GQA | 10.7774 | 38.28% |
| 零回报预测 / 与 memory 无关的配对策略 | 9.0000 | 50.00% |

这些结果展示了训练链路和真实参数更新，**没有展示本设定下的记忆条件化泛化收益，也没有展示 MiTA 的任务优势**。去掉 memory 后三个后端的配对排序准确率都为 50%；完整原始预测、补集 rulebook 诊断、每种种子的损失/计时和源码 SHA-256 保存在 [受控记忆任务结果](../benchmarks/mita-memory-ablation.json)，可由 `scripts/ablate_mita.py` 复现。最终六次训练累计约 36.31 秒。评估结果没有被用于选择新学习率、训练步数或重试筛选更好种子；两种种子和一个小型合成任务不足以给出普遍架构结论。

另已测量优化前 reference 实现的 CPU/CUDA 时延，以下保留为历史基线；当前 grouped 路径的同轮结果见 [性能分析](mita-performance-analysis.md)。以下数据只适用于记录中的机器和工作负载：Windows、PyTorch 2.8.0+cu128、float32、batch 1、两次热身与七次采样；CPU 单线程，GPU 为 RTX 3050 Ti Laptop GPU。完整网络测量使用 384 hidden、32 个事件、128 条 memory、三个候选、共 161 个主干 tokens，输入特征预先编码；不含检索、哈希、训练或远端 API。

| 测量对象 | Full GQA 前向中位数 | MiTA 前向中位数 |
|---|---:|---:|
| CPU，四层完整本地网络 | 29.711 ms | 61.307 ms |
| CUDA float32，四层完整本地网络 | 9.939 ms | 17.141 ms |
| CUDA float32，单层 attention，N=1024 | 2.432 ms | 9.702 ms |

N=1024 的 CUDA attention forward+backward 测得 PyTorch allocated tensor 峰值相对基线增量分别为 GQA 16.139 MiB、MiTA 68.399 MiB。这是已驻留参数/输入之上的分配器测量，不是 reserved cache、驱动显存或进程总显存。数据见 [CPU 基准](../benchmarks/mita-attention.json) 和 [CUDA float32 基准](../benchmarks/mita-attention-cuda.json)。这些历史观测中，原 reference 的实际时延与该训练内存指标都高于优化后的稠密 SDPA；较少的理论点积分数并未转化为这里的加速或节省。CUDA 支持提供可选执行设备，不代表已经实现官方专用 FlashAttention 内核的性能。
