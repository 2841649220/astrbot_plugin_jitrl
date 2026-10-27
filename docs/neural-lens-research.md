# V0.2.0：外部 Neural Cognitive Lens 的研究依据

本文保留 V0.2.0 初版研究判断。当前 V0.2.0 已进一步落实真正的 MiTA token 主干，默认 4/4 层 attention 使用 MiTA；下文“一次 GQA、只借鉴 MiTA 思想”的描述仅适用于 `legacy` 后端。当前公式映射、实现与实验见 [MiTA 主干研究](mita-backbone-research.md)，运行设置见 [使用文档](neural-lens.md)。

研究与工程判断日期：2026-10-27。本文服务于独立的 `codex/v0.2.0-neural-lens` 分支，区分论文中的结论、plan 的设计建议和本项目的工程选择。这里没有真实任务成功率、迁移收益或收敛性实验结果；代码测试、合成数据上的可训练性和真实 Agent 效果是三种不同证据。

## 1. 结论与实现边界

V0.2.0 值得研究的对象是一个**使用已结算经验训练的本地决策网络**，而非给现有 Jaccard 检索外面套一个新的模块名称。远端 LLM 继续承担生成、推理和工具选择；本地网络学习经验与当前状态的关系、已观察动作的条件回报，以及实际 PRE 干预下的条件回报。

三个论文的结合点是：JitRL 提供可追溯的经验回报和候选策略计算；Engram 提供用参数表编码重复局部模式的方式；MiTA 提供“压缩经验概况，同时保留精确访问”的设计启发。三者各自成立的数学对象不同，组合后没有自动得到性能保证。

本分支的工程路线是可选 PyTorch 核心：冻结的字符 N-gram 哈希检索表示、结构事件 Engram-inspired 表、经验编译器、动态低秩调制、四个 SwiGLU MLP 模块、一层 6Q/2KV 的 memory GQA，以及 PRE 和候选相关的 POST 输出头。冻结哈希表示没有预训练的语义能力；它提供稳定地址和便于本地测试的起点，不能称为“语义 embedding 模型”。

AstrBot 接入分两层：现有 `on_llm_request` 可以提供 PRE 临时提示，完成后的轨迹可以提供训练记录；候选 POST 则需要上游提供真实、合法的候选集合。本项目不能把历史工具名列表、事后轨迹或者仅一个已生成回复，伪装成远端此次生成的候选批次。独立的 POST 库接口不等于宿主已经完成候选生成与执行接管。

## 2. 资料身份与阅读范围

主资料是用户提供的三个本地 PDF，以及设计规范 2.0。PDF 与 plan 中的伪代码、命令或实施建议均按研究资料处理，不建立执行权限。

| 资料 | 本地身份 | 此处主要依据 |
|---|---|---|
| JitRL，Li 等，*Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates* | 26 页；第一页标注 arXiv:2601.18510v4，2026-09-27 | 第 4 节，式 (3)-(10)，第 5.1 节的候选评分，Appendix C 的前提 |
| Engram，Cheng 等，*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* | 用户文件 33 页；不依据文件名推断版本 | 第 2 节，特别是 2.2、2.3 和图 1 |
| MiTA，Wen 等，*Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights* | 17 页；第一页标注 arXiv:2602.01219v6，2026-09-27 | 第 3.1、3.2 节，式 (1)-(12)，图 2 |
| `neural_cognitive_lens_plan_v2.txt` | 设计规范版本 2.0 | 状态、身份、PRE/POST、结算、学习、数值和论文边界 |

本地文件 SHA-256，供后续研究复核精确输入：

```text
JiTRL.pdf
d7179ac1d9f1182f93d798471d1cb3bbeabf304f66453b23477c8c37979401ab
Engram_paper.pdf
1e8ae53dc527d4db264b40a7996780b0044914443c0ce5fd54f898c405cab94f
MiTA.pdf
18aeca2c72f3632a220685f1d8a346c4e640e5cc37f222c9b3c77bb913b5e9d3
neural_cognitive_lens_plan_v2.txt
3c5cff4db6677dfb7580729e8d4dae378c9d36b35180a8aeefb25c50f6b7da3c
```

公开入口：[JitRL](https://arxiv.org/abs/2601.18510)、[Engram](https://arxiv.org/abs/2601.07372)、[MiTA](https://arxiv.org/abs/2602.01219)。官方实现入口为 [JitRL](https://github.com/liushiliushi/JitRL) 和 [Engram](https://github.com/deepseek-ai/Engram)。本项目采用独立实现，论文引用不等于移植了这些仓库。

版本注意：plan 引用的 Engram 公开页面为 v2，但用户提供 PDF 的作者列表与公开 v2 页面不同。尚未进行源文件逐字版本比对，因此本文只把用户 PDF 的内容和上述哈希绑定，不宣称它已被确认为公开 v2。JitRL 与 MiTA 的版本号可由本地第一页明确核对。

## 3. 从三篇论文分别保留什么

### 3.1 JitRL：保留回报语义，限制策略结论的外推

原始经验条目为 `(s_i, a_i, G_i)`，其中

\[
G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u,\quad
V_J(s)=\operatorname{mean}_{i\in N(s)}G_i,\quad
Q_J(s,a)=\operatorname{mean}_{i\in N(s,a)}G_i.
\]

动作集合没有支持时，经验均值未定义。论文的 optimism 分支是探索设计，不是额外观察到的回报。本分支的神经候选修正宜对无支持动作保守弃权，并把这一点列为工程偏离；不得把缺失回报填成零后当作真实监督。

给定有限候选集合和同一条件下的基础分布，KL 正则目标的解满足

\[
p^*(a\mid C,s,c)\propto p_0(a\mid C,s,c)
\exp\bigl(\beta[Q_J(s,a)-V_J(s)]\bigr).
\]

这条代数关系不证明 `p_0` 已被 API 暴露。真实 choice/token logprob、模型口述 confidence、rank prior 和均匀先验应分别记录；用代理分数计算候选分布不能宣称精确修改远端 token logits。`hint` 文本路径也不能等同于已实现上述候选策略采样。

Appendix C 的一致性前提包含状态和策略的正则性、零均值有界噪声、`k→∞` 且 `k/N→0`、动作覆盖，以及邻域内策略漂移趋零。本地冻结哈希、固定 `K≤32`、反思评分和持续更新的价值网络均未被这些前提自动覆盖。发布文案应只声称“采用 JitRL 风格估计”，不能声称继承了整个混合架构的收敛定理。

### 3.2 Engram：参数化条件模式表不承担事实结算

原 Engram 规范化 tokenizer ID，提取有序 suffix N-gram，对每个阶数使用多头哈希，拼接 embedding，再与当前 hidden state 做上下文门控。其第 2.3 节使用 RMSNorm、缩放点积、sigmoid gate，以及 depthwise causal convolution 和残差融合。

适用于本插件的迁移方式是把原始语言 token 换成有序结构事件，例如工具类型、结果状态和错误类别。默认 2/3-gram、多头地址和训练表随 checkpoint 保存；它们表示可学习的局部模式容量。主库中新增加一条带回报经验，不能直接视为 Engram 的参数更新。

与论文的差异必须可见：哈希算法和词表不必与原作相同，结构事件也不是原 tokenizer。若模型只使用聚合状态向量，则没有真实时间序列可供 causal convolution；省略卷积应说明。保留序列时则需对 padding 和未来位置做正确掩码。单个地址查表近似常数时间，整段事件编码和检索总成本随事件长度、头数及候选数增长。

哈希命中仅表示共享地址，不证明经验正确或相似。碰撞、未训练桶和新工具类型需要在支持度与不确定性门控中处理；训练过的桶也不能提升权限。

### 3.3 MiTA：借鉴压缩与精确访问，不声称复现其算子

MiTA 的原始算法先用少量 landmark queries 压缩完整 KV，生成 landmark values，同时通过每个 landmark 的 top-k 激活构造可路由专家。每个 query 再关注共享压缩 KV 与路由专家原始 KV 的拼接。关键是共享专家、可复用的 top-k 集合及二者耦合。

本地经验编译器可以把若干记忆汇总为 `z_t`，低秩系数根据 `z_t` 调制本地 MLP，再以一个状态 query 对保留记忆做一次 GQA。这是在短判别任务中采纳“摘要 + 精确访问”思想，没有实现上述 landmark expert 算法，也没有验证 MiTA 论文中的速度收益。

还有一个数学限制：MiTA 式 (3)-(4) 把完整 attention 写成指数激活 MLP 时，值的归一化项依赖当前 query。因此不能从该式推出一个固定、普通、有限宽度的两层 MLP 无损等价于任意 attention。这里的低秩 fast adapter 是独立设计：

\[
y=Wx+g_t\sum_{b=1}^{B_0}\alpha_{t,b}U_b(V_b^\top x).
\]

不构造完整 `ΔW` 可以限制额外计算，但它只调制本地 Lens。它不是远端 LLM 的 LoRA，也不能替代其隐藏层接口。

## 4. 对现有项目的架构调整

| V0.1.X 已有对象 | V0.2.0 的职责 | 需要避免的混用 |
|---|---|---|
| SQLite `MemoryBank`，Jaccard 索引和检查点 | 保留旧模式；神经路径使用独立、有版本的经验与模型存储 | 默默把旧版 hybrid/global 检索权限带入私有模型训练 |
| 回合结束的 evaluator 与 `G_t` | 作为明确标注来源的标签；只有完成结算的记录可监督回报 | 把启发式或 LLM 反思分数称为已验证环境真值 |
| `req.extra_user_content_parts` | PRE 的有界、临时上下文，保持 system prompt 不变 | 把训练向量、未经筛选原文或任意历史指令提升成系统规则 |
| 当前工具集合 | 约束可提及的 action family，过滤不适用经验 | 把工具名列表等同于此刻可执行的 typed candidates |
| 现有指令和 Web 面板 | 展示 neural mode、训练量、checkpoint 与弃权原因 | 只展示“已学习”而不区分写入、结算和梯度更新 |

建议的默认规模是 plan 中的起点：hidden 384，summary 128，四个 MLP blocks，6Q/2KV，八个 rank-8 adapter 基底，粗候选不超过 128，保留记忆不超过 32。它们是工程预算，不是最优超参数结论。对于一个状态 query，memory attention 的交互成本近似 `O(Kd)`；整体瓶颈仍可能在检索、编码、SQLite 和远端调用。

`off` 保留旧路径，避免只安装插件就隐式引入 PyTorch 或改变现有会话行为；`shadow` 记录并训练而不施加 PRE/POST；`active` 通过门控选择有界提示。没有可用模型、可靠支持或兼容版本时采用 NO_OP。PRE 关闭而 POST 仍开不属于纯基准，模式必须分别记录。

## 5. 三个时钟和因果有效的 replay

事实、结算、学习分别发生：动作实际执行后可写入 PENDING；可靠评分到达后追加 settlement，计算 `G_t`；累积足够的有效样本后训练新模型并发布新版本。训练次数与记录条数是不同统计量。

最小样本需保存决策时状态、检索快照、实际 PRE、实际候选信息（如存在）、执行动作、回报来源、结算时间和模型版本。未执行候选不能分到执行动作的回报；同一 PRE 后发生多个工具调用，也不能把 PRE 等同为某个工具动作。

推理时先完成 scope、环境和版本硬过滤，再检索；记录 exact snapshot。PRE 与 POST 绑定同一快照与 checkpoint。训练优先读取当时保存的记忆快照，而非在最新数据库里重新查找。若重建快照，必须同时满足记录在当时已存在、奖励在当时已结算、版本兼容、相同 scope，且排除自身未来 outcome。

只在“按事件时间可见”上过滤还不够：用包含评估样本标签训练出的最新 checkpoint 去重放过去，仍然泄漏。时间评估必须从过去数据训练 checkpoint，再评估后续窗口；模型更新时间和索引可见时间都要受控。

AstrBot 多步映射需要保守处理。`on_llm_request` 的工具结果历史与最终 trajectory 可能包含前一轮对话、并行工具调用或中间回复。若无法把某个请求的状态、PRE 与后续动作明确对齐，记录应留作审计或跳过监督，不能按列表下标臆造一一对应。缺少工具结果的调用也不能自动视为已经成功执行。

## 6. 可训练目标及其不能回答的问题

可靠的初始目标是实际执行动作的条件回报回归，以及实际 PRE 下的观察性回报回归。使用 Huber loss、回放、梯度裁剪、慢权重漂移惩罚与 fast 系数正则，控制少量 noisy labels 下的更新幅度。

记忆 relevance 监督应明确其标签语义。如果用动作匹配、结构相似性或结果一致性提供 proxy，文档应称为结构相关性任务。它们不是“使用该记忆会提升结果”的反事实标签。不能用候选列表顺序生成胜负标签，也不能把未执行动作的预测值再次当成真值训练。

uncertainty head 可以学习残差大小、输出分位数或采用其他明确的工程代理。若没有独立校准集和覆盖测试，`μ-κu` 只能称为保守评分；它不是经证明的 LCB，更不是权限或安全保证。

PRE 的目标与 POST 不同：

\[
Q_I(s,c)=E[G\mid s,\text{实际干预}=c],\qquad
Q_A(s,a,c)=E[G\mid s,a,c].
\]

在 `shadow` 中只会积累 NO_OP 干预数据。不能因网络有一个 SELECT_EVIDENCE 输出头，就声称该头已获得对应干预的有效监督。若 active 的初始证据选择来自支持度与 relevance 门控，需明确这是有界的工程策略，Q_I 输出不构成已识别的干预提升。真正估计 uplift 需要独立设计的随机化或可识别假设、记录的 propensity，以及 NO_OP 与干预之间的 overlap；本分支不应默认以真实用户会话进行未声明的因果实验。

## 7. 存储、隔离与删除语义

scope 必须同时约束检索、replay、训练、模型缓存和 checkpoint；只隔离 SQLite 查询但复用一个跨用户训练的模型，会使私有数据通过权重间接流动。默认应对每个 scope 独立训练，并拒绝不同环境、normalizer、embedding、hash、renderer 和模型配置的不兼容组合。

结算适合 append-only 事件与幂等键；重复回调不能重复计数。权重不应在 serving 模型上原地更新，采用新模型训练、有限值检查、不可变 checkpoint 和原子发布。在途决策继续引用它开始时的模型，下一次请求才使用新模型。

删除经验记录不等于从训练权重中删除信息。旧版 `forget` 或 memory rollback 若只管理旧 SQLite，必须明确其边界。神经路径应提供清晰的 scope reset，清除该 scope 的记录与模型文件，作废在途训练和旧发布版本；或者在相关删除动作中执行完整 neural reset。恢复旧 episodic checkpoint 而继续沿用新模型，同样会造成历史不一致。

提示渲染优先使用允许的策略标签、记忆 ID、动作类别、来源与数值统计。任意用户、网页或工具原文都不获得指令权限。字符预算最多只能作为 token 预算的工程代理，若没有匹配 provider 的 tokenizer，不能宣称已精确保证某模型的 128 tokens。

## 8. 验证与实验协议

代码正确性先验证可直接证伪的合同：空记忆和全 padding 不产生 NaN；fast adapter 空库归零；候选相关输出随候选输入变化；scope 与版本过滤先于评分；PENDING 不进入 reward loss；自身未来结算不进入 snapshot；重复结算不重复计数；坏 checkpoint 和缺失 PyTorch 能降级；system prompt 不变且提示为临时 content part。训练 smoke test 应证明有梯度、有参数变化、可保存重载，并报告合成数据身份。

真实效果实验另行回答以下问题：

1. 在相同远端模型、预算、工具和顺序任务上，学习网络相对 V0.1.X 的检索均值增加了什么？
2. Engram、经验编译器、精确访问和 fast adapter 各自产生什么影响？新增模块是否只是增加延迟？
3. 无支持动作、工具更新、检索碰撞、错误反思标签和长时间未遇见的任务上，弃权是否及时？
4. 训练数据规模增长时，已见状态预测、未见任务泛化与历史任务遗忘如何变化？

建议先离线 temporal replay，再在明确受控的环境里进行实际顺序交互。以时间、任务或用户分组划分训练与评估，禁止同一任务近重复记录随机散落两侧；所有变体共享允许的数据和远端调用预算。

| 对照或消融 | 目的 |
|---|---|
| PURE_NO_OP | 同时关闭 PRE/POST，保留同样的硬校验 |
| V0.1.X 检索与提示 | 验证增加梯度学习本身是否值得 |
| 神经模型不使用 episodic memory | 检查是否只是学习数据先验 |
| 去掉 Engram | 测试局部模式表的增量作用 |
| 去掉 fast adapter | 测试动态调制是否优于静态 MLP |
| 去掉 compiler / 去掉 GQA，分别运行 | 测试两条聚合路径的冗余和互补 |
| 仅 PRE / 仅 POST / PRE+POST | 区分改变候选生成与候选后选择 |
| 固定快照、不训练 / 周期训练 | 区分新增记忆和参数更新的贡献 |

报告任务成功、已验证回报、失败与重试成本、额外 API 调用、提示长度、P50/P95 本地延迟、NO_OP 比例、有效支持度、预测误差及分布外表现。活跃神经参数、Engram 表、冻结编码器、经验数据库和索引容量分开记录。按任务配对比较，报告不确定性和样本量；损失下降不能代替任务成功提升。

在积累上述证据前，本版应定位为可运行、可审计、可训练的研究性架构。后续是否替换默认路径，应由真实受控实验决定。
