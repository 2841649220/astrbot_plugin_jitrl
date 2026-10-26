# JitRL 持续学习 · astrbot_plugin_jitrl

[![CI](https://github.com/2841649220/astrbot_plugin_jitrl/actions/workflows/ci.yml/badge.svg)](https://github.com/2841649220/astrbot_plugin_jitrl/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.2.1-blue)](CHANGELOG.md)

> 在 V0.1.2 经验学习基础上，增加可选的本地 Neural Lens。远端大模型始终冻结；神经扩展默认关闭，旧配置与旧 API 保持兼容。
>
> V0.1 路径无需训练；V0.2 神经路径使用可选 PyTorch 训练会话隔离的小模型。当前默认四层 token 注意力全部使用真正的 MiTA 压缩与路由机制，结合 JitRL 经验回报和 Engram 条件模式记忆。

## V0.2.1 快速入口

开发分支：`codex/v0.2.0-neural-lens`，基于 V0.1.2；没有合入 V0.1 分支。现有插件入口、配置键、指令、经验库、检查点及面板路由继续保留。`neural.mode: off` 为默认值，使用原来的行为，无需安装额外依赖。

- [神经架构的研究依据与论文边界](docs/neural-lens-research.md)
- [MiTA 主干：公式落地、4/4 层替换与研究实验](docs/mita-backbone-research.md)
- [启用方法、API 合同、数据与权重管理](docs/neural-lens.md)
- 开启采集：显式安装 `requirements-neural.txt`，配置 `neural.mode: shadow`；管理员用 `/jitrl lens train` 训练，再切为 `active`。
- 模型：384 hidden、4 层 MiTA + SwiGLU、事件 Engram 条件表、经验编译器和 rank-8 动态低秩基底；状态、事件和最多 128 条历史经验形成 token 序列。MiTA 真正构造 landmark 压缩 KV、可复用 Top-k KV experts 和每个 query 的路由，并对拼接分支共同归一化。
- `neural.attention_backend` 默认 `mita`（4/4 层）；可显式选择 `hybrid`（3/4）、`gqa`（匹配的四层完整注意力对照）或 `legacy`（V0.2.0 原单 query 主干）。V0.2.0 经验与快照继续使用，升级 MiTA 需重训本地权重。
- `neural.device` 支持 CPU、CUDA 或 auto，默认 CPU。正式 384d/四层 MiTA 已分别完成 CPU 与 CUDA 的训练、保存重载和候选评分闭环。
- 已接入 AstrBot 的 PRE 临时提示与完成后采集；POST 为独立的候选评分 API，仍需上游候选生成和执行前硬校验。
- 保留 `/jitrl feedback` 与原反馈关键词入口；神经模式将反馈作为追加结算，保留旧奖励历史。旧面板的备份、删除、导出和回退只管理 V0.1 经验；神经事实和模型使用 `/jitrl lens reset confirm` 独立重置。

开发仓库可运行 `python scripts/neural_lens.py smoke --steps 20` 验证正式尺寸模型的采集、训练、原子检查点与候选评分闭环。此演示为合成数据，不代表真实任务收益；结果见 [smoke 报告](benchmarks/neural-smoke.json)。

CPU 算子与完整主干对照见 [MiTA 基准](benchmarks/mita-attention.json)，复现命令 `python scripts/bench_mita.py --model`。当前纯 PyTorch CPU 路径尚未取得相对优化 GQA 的提速，稀疏交互数量与实际延迟分开报告。

以下原有算法与配置说明描述 `neural.mode: off` 的 V0.1 兼容路径。其中“无梯度更新”指该路径及冻结的远端模型，不适用于新增本地神经训练器。

---

## 1. 它解决什么问题

LLM Agent 的能力在部署那一刻就被冻结了。它会把同一个错误犯第二次、第三次，因为**没有任何机制把「上次那样做不行」沉淀下来**。

传统解法是强化学习（PPO / GRPO / WebRL），代价是：要海量环境交互、要 GPU 在线更新、还会灾难性遗忘。论文实测 WebRL 训练一轮约 **$9,900**，而 JitRL 只要 **$290**，性能反而更好。

JitRL 的思路是：**不动模型参数，只动推理时的输出分布。**

```
每个 episode 结束后：
  轨迹 τ → LLM 评估器给出步级奖励 r_t → 折扣回报 G_t = Σ_{u≥t} γ^(u−t)·r_u
                                        ↓
                            写入非参数记忆 M = {(s_i, a_i, G_i)}

下一次请求到来时：
  当前状态 s → 用 Jaccard 相似度检索 top-k 近邻 N(s)
             → V̂(s) = mean(G_i)                       （式 4，状态价值 = 基线）
             → Q̂(s,a) = mean(G_j)  over N(s,a)        （式 5，已知动作）
             → Q̂(s,a) = V̂(s) + α/|N(s)|  以概率 λ    （式 6，未见动作的乐观探索）
             → Â(s,a) = Q̂(s,a) − V̂(s)               （式 7，优势）
             → z'(s,a) = z(s,a) + β·Ã(s,a)            （式 10，闭式策略更新）
             → Softmax(z') 重新采样                  （式 9 的等价实现）
```

定理 4.1 证明了这个加法式 logit 更新**就是** KL 约束策略优化目标的精确闭式解：

```
π* = argmax_{π'} ( E_{a~π'}[Â(s,a)] − (1/β)·D_KL(π' ‖ π_θ) )
    ⟹  π*(a|s) ∝ π_θ(a|s)·exp(β·Â(s,a))
```

换句话说：**优势估计 + 温度 β 的 KL 约束 = 一个加性 logit 偏置**。没有梯度，没有反向传播，没有参数更新。

---

## 2. 本插件如何落地到 AstrBot

AstrBot 的对话是黑盒 API，拿不到原始 logits。论文 §5.1 正好给了这个情况的官方方案 —— **Verbalized Logit** 变体：让模型对候选动作输出 0–100 的置信度，把置信度转成 logit 参与闭式更新。插件完整实现了这条路径。

### 2.1 论文概念 → AstrBot 概念

| 论文 | 本插件 | 代码位置 |
| --- | --- | --- |
| 环境 observation `o` | 用户消息 + 最近动作 + 工具结果摘要 | `jitrl/state.py: abstract_state` |
| 结构化状态 `s`（Regularized URL） | 脱敏 + 标识符正则化后的稳定签名 | `jitrl/state.py: regularize / redact` |
| 动作 `a`（semantic normalization） | 工具调用 / 回复，参数归一化为签名 | `jitrl/state.py: normalize_action` |
| 检索相似度 Jaccard（式 24） | 中英混合 token 集合（中文 bigram） | `jitrl/state.py: tokenize / jaccard` |
| 记忆 `M = {(s_i,a_i,G_i)}` | SQLite 表 `experiences` | `jitrl/memory.py: MemoryBank` |
| 式 4/5/6/7/25 价值估计 | `ValueEstimate` | `jitrl/estimator.py: estimate_values` |
| 式 9/10 闭式更新 | `PolicyDecision` | `jitrl/policy.py: update_logits / select_action` |
| Verbalized Logit 采样 | 置信度 → logit | `jitrl/reranker.py` + `policy.verbalized_to_logit` |
| LLM 评估器（附录 G.1.2） | −3..+3 步级奖励 + 规则信号融合 | `jitrl/evaluator.py` |
| 轨迹 `(s_t,a_t)` 重建 | 从 `run_context.messages` 还原 | `jitrl/trajectory.py` |
| 策略输出 `π*` | 通过 `extra_user_content_parts` 注入用户消息的临时证据块 | `jitrl/hint.py` |

### 2.2 运行时数据流

```
                      ┌──────────── on_llm_request ────────────┐
  用户消息 ──────────►│ 1. abstract_state()  构造状态 s         │
                      │    （同回合内命中状态缓存）             │
                      │ 2. MemoryBank.topk() 检索 N(s)          │
                      │    （倒排索引 + 前缀剪枝，无损）         │
                      │ 3. estimate_values() 全动作探针 V̂ Q̂ Â  │
                      │ 4. 取 |Â| 最强的 |C| 个做正式估计        │
                      │ 5. select_action()   z' = z + β·Ã       │
                      │ 6. 证据门控 + 注入临时块                │
                      │    （决策按 状态/记忆版本/工具集 缓存）  │
                      └────────────────────────────────────────┘
                                        │
                                        ▼
                                  主模型作答 / 调工具
                                        │
                      ┌──────────── on_agent_done ─────────────┐
                      │ 7. extract_transitions() 重建轨迹       │
                      │ 8. StepRewardEvaluator  打分 r_t        │
                      │ 9. discounted_returns   G_t             │
                      │10. MemoryBank.add_episode() 落库        │
                      └────────────────────────────────────────┘
```

学习默认在**后台任务**里跑（`learning.async_learn`），用户不会多等一毫秒。

### 2.3 相对论文的工程改进（算法与执行效率）

论文是评测框架，本插件是**每次对话都要跑一遍**的运行时。落地时做了 8 处数学/工程上的改动，
每一处都写清了"为什么"；要严格复现论文数值时，把 `retrieval.estimator_weighting` 与
`policy.exploration_estimate` 切回 `uniform` / `sample` 即可回到论文原式。

| # | 论文原式 | 本插件实现 | 为什么改 |
| --- | --- | --- | --- |
| 1 | 检索：对全表逐条算 Jaccard（论文 Table 19：2500 条时 15–47 ms） | `jitrl/index.py`：倒排索引 + 长度过滤（`t·\|q\| ≤ \|x\| ≤ \|q\|/t`）+ 重叠下界（≥ `ceil(t·\|q\|)`）+ 前缀过滤（PPJoin），再逐条精确校验 | 检索在每次 LLM 请求的关键路径上，线性扫描不可接受。剪枝是**无损**的（不是近似 ANN），`tests/test_index.py` 用暴力参考实现做全矩阵交叉验证 |
| 2 | 式 (4)(5)：`V̂`、`Q̂` 都是邻居的等权平均 | V 和每个 Q 组共用同一套核权重（相似度 × 时效） | 低相似邻居不该和精确匹配同权；更关键的是 V/Q 必须同权——否则 `Â = Q̂ − V̂` 在比较两个不同的估计量（旧实现里某动作组权重和为 0 时会退化成等权均值） |
| 3 | 低于 `min_count` 的动作视为"未见"，可能拿到随机乐观值 | 向状态基线收缩：`Q = (n·Q̄ + κ·V̂)/(n+κ)`，κ = `min_count` | 一次偶然观测不该让优势在 +1 / −1 之间跳变；κ = 0 即论文原式，默认 `min_count = 1` 时行为不变 |
| 4 | 式 (6)：以概率 λ 取 `Q = V̂ + α/\|N\|`，否则 `Q = 0` | 取闭式期望并修正中性基线：`Q = V̂ + λ·α/\|N\|`，即 `Â = λ·α/\|N\|` | 论文的 `Q = 0` 分支给出 `Â = −V̂`：在正价值状态里，未尝试动作比任何已知动作罚得都重；在负价值状态里反而被奖励。且逐请求采样会让同一情景的建议在 +1 / −1 间抖动，而插件并不能像论文那样采样动作 |
| 5 | 式 (25)：`Ã = A/(max\|A\|+ε)` | 同式 + `max\|A\| < 1e-3` 时直接输出 0 | 所有回报完全相同时（评估器降级到纯规则很常见），式 (25) 会把浮点舍入噪声放大成 ±1 的策略抖动 |
| 6 | 候选集 `C = C_LLM ∪ N(s)` 的动作，按 `\|C\|` 截断 | 先对**全部**检索到的动作估优势，再按 `\|Â\|` 取 `\|C\|` 个重算归一化 | 支持度低但证据强的动作（尤其带强负优势的工具）不该在估计前就被丢掉；工具门控也因此能看到全量证据 |
| 7 | 直接改 logits | 证据门控（邻居 ≥ 2 且 `max\|Â\| ≥ 0.05` 才注入）+ 更短的块 + 同回合决策缓存 | 黑盒模型只能走文本；弱证据注入既费 token 又误导，且同一回合内应给出一致建议 |
| 8 | 附录 G.1.2 步级打分（完整轨迹） | JSON 打分 + 规则融合 + 超时/异常降级 + 12000 字符提示预算 | 40 步轨迹的评估提示从约 28k 字符降到 12k 以内，评估成本减半而不损失打分粒度 |

执行效率的实测（同一台机器、同一份基准脚本，详见 [benchmarks/README.md](benchmarks/README.md)）：

| 场景（p50） | 0.1.1 | 0.1.2 | 加速 |
| --- | ---: | ---: | ---: |
| top-k 检索（k=10, r=0.8, 1000 条记忆） | 57.6 ms | 0.31 ms | ~184× |
| top-k 检索（k=10, r=0.8, 5000 条记忆） | 124.0 ms | 0.39 ms | ~319× |
| top-k 检索（k=10, r=0.8, 20000 条记忆） | —（>20 min 未完成） | 0.69 ms | >300× |
| 一次请求的策略前段（检索+估计+渲染，5000 条） | 118.7 ms | 0.47 ms | ~253× |
| 写入一个 Episode（5000 条库） | 22.4 ms | 0.66 ms | ~34× |

两条曲线形状不同才是重点：0.1.1 的耗时随记忆条数线性增长（每条都要 JSON 解码 + 重建
`frozenset` + 算一次 Jaccard），0.1.2 基本平坦（倒排索引只碰"可能过阈值"的候选，token 以
整数数组常驻内存）。

---

## 3. 安装

```bash
cd AstrBot/data/plugins
git clone https://github.com/2841649220/astrbot_plugin_jitrl.git
```

或者直接把 `astrbot_plugin_jitrl` 目录放进 `data/plugins/`，然后在 WebUI 的「插件」页重载。

要求 **AstrBot `>=4.29,<5`**：学习钩子 `filter.on_agent_done` 从 4.24 起提供，
而 `views/` 面板目录从 4.29 起才会被宿主扫描，因此整体门槛是 4.29（该版本要求
Python 3.12+）。版本不满足时 AstrBot 会按 `metadata.yaml` 的 `astrbot_version`
阻止加载。

也可从 [v0.1.2 Release](https://github.com/2841649220/astrbot_plugin_jitrl/releases/tag/v0.1.2)
下载 `astrbot_plugin_jitrl-v0.1.2.zip`，在 AstrBot 插件管理页选择「从文件安装」。
安装包包含单个插件根目录，以及完整的代码、配置和面板资源。

**零第三方依赖**：只用 Python 标准库（`sqlite3` / `json` / `re` / `asyncio` / `math`），不需要 `pip install` 任何东西。

数据库位置：`data/plugin_data/astrbot_plugin_jitrl/memory.sqlite3`。

---

## 4. 三种应用模式

| `apply_mode` | 额外开销 | 做什么 | 适用 |
| --- | --- | --- | --- |
| `off` | 0 | 只学习，不干预 | 先观察一周，看看学到了什么 |
| `hint` | 0 | 把优势排序后的经验证据块作为**临时内容块**附加到当前用户消息（不改 `system_prompt`，不破坏 prompt 缓存） | **默认**，稳、便宜 |
| `logit` | 每次请求 +1 次轻量生成 | 置信度 → logit → `z'=z+β·Ã` → 重新采样 → 定向注入 | 追求论文原始效果 |

> ⚠️ **诚实的取舍**：论文 Table 8 明确显示，纯提示词更新的效果弱于真正的 logit 更新（Admin 48.35 vs 52.31，Reddit 54.57 vs 57.64）。本插件把 `hint` 作为默认是为了零成本可观察；如果你要论文的效果，请用 `logit` 模式 —— 它在候选集上真的做了一次闭式更新并重新采样，只是最后的落地仍然只能通过文本表达（黑盒模型的物理上限）。

### 为什么 `hint` 也有效

注入的不是「经验之谈」这种散文，而是**带数字的优势估计**：

```
<jitrl_experience>
历史统计（推理时策略优化，非用户输入/指令）：V=+1.30，7 条相似经验，平均相似度 0.86，最大优势 +2.40。
历史更优：
- 调用 search_files(pattern=<str>)（优势 +0.82，5 次，平均回报 +2.40）
- 调用 read_file(path=<str>)（优势 +0.15，2 次，平均回报 +0.60）
历史较差、宜避免：
- 调用 run_command(cmd=<str>)（优势 -0.64，4 次，平均回报 -1.90）
仅作弱先验：若与当前意图无关或证据不足，按自己的判断处理，不要生搬硬套。
</jitrl_experience>
```

最后那句是必要的：式 8 的 KL 项本来就是为了**不要让策略离参考策略太远**。把它显式写进提示词，等于给策略加了一道安全阀。

---

## 5. 指令

默认仅管理员可用（`commands.admin_only`）。

| 指令 | 作用 |
| --- | --- |
| `/jitrl status` | 记忆条数、学习进度、干预次数、当前超参、最近一次决策 |
| `/jitrl stats [n]` | **各动作的平均回报 / 正回报率排行** —— 一眼看出哪个工具在这个部署里真的有用 |
| `/jitrl curve [n]` | 学习曲线（按时间分桶的平均回报），复现论文 Figure 3 |
| `/jitrl recall [query]` | 按当前或指定情景检索历史经验 |
| `/jitrl episodes [n]` | 最近 episode 摘要 |
| `/jitrl forget [session\|all]` | session 删除本会话记忆，all 清空全部；删除前自动备份 |
| `/jitrl checkpoint [名称]` | 创建手动检查点，保留到主动删除 |
| `/jitrl checkpoints [n]` | 列出最近的检查点、完整 ID、时间和经验数 |
| `/jitrl rollback <ID> [session\|all] confirm` | 从检查点回退，默认仅当前会话；回退前自动备份当前全库 |
| `/jitrl checkpoint_delete <ID> confirm` | 永久删除指定检查点，不修改当前记忆 |
| `/jitrl feedback good\|bad` | 手动回填对上一轮的评价 |
| `/jitrl export` | 导出记忆 JSON |

### 检查点与污染恢复

检查点管理指令始终仅管理员可用，即使关闭了 `commands.admin_only`，因为每份快照包含全部会话的数据。

```text
/jitrl checkpoint 可信基线
/jitrl checkpoints
/jitrl rollback <从列表复制的完整ID> session confirm
/jitrl rollback <从列表复制的完整ID> all confirm
```

先在确认表现正常时保存一个手动检查点。发现异常后，可以只回退当前会话，或回退全部会话。
回退命令未带 `confirm` 时只提示确认用法。操作成功后会返回一份「回退前备份」的 ID；
恢复这个 ID 即可撤销刚才的回退，撤销时同样可以选择 `session` 或 `all`。

默认首次打开数据库时保存基线，之后每累计 **50 个完成的 Episode** 自动保存一次。
`memory.checkpoint_interval = 0` 关闭这两类自动保存，删除和回退前的备份仍然生效。
自动保存计数在插件重载或回退后重新开始；自动历史和操作前备份各保留 **20** 份，手动检查点不参与轮换。

快照独立存放在 `data/plugin_data/astrbot_plugin_jitrl/checkpoints/`，使用 SQLite Backup API
保存已提交的 WAL 数据，校验后才发布完整文件。回退先校验格式、表结构、完整性和记录数，
再备份当前数据，最后在一个事务内替换记忆；校验、备份或写入失败时，当前记忆保持不变。
会话回退保留其他会话的数据。快照包括奖励、折扣回报、episode 元数据及原有脱敏内容，
不回退插件配置和进程累计计数。

删除或回退会使旧请求和旧评估任务的写入失效，并清理反馈、决策缓存，避免旧结果把恢复后的数据库再次污染。
回退后的新请求可以正常学习；如果评估配置仍会产生异常经验，可先关闭 `learning.enabled`，调整后再恢复学习。

### 自然语言反馈

不用打指令。用户回一句「不对」「谢谢」这类短句，插件会在时间窗内（默认 180 秒）自动把它折算成 **±2.0** 的奖励增量，回填到上一轮 episode 的所有步骤，并**重算 G_t**。这是零成本的密集信号来源。

### Agent 工具

插件向模型暴露了一个函数工具 `jitrl_recall(query)`：模型可以在多轮工具调用的中途主动查一下「这种情况我以前是怎么处理的」。返回的是按优势排序的证据，同样带「弱证据」免责声明。

### WebUI 面板

插件自带一个可视化面板（`views/dashboard/`），在 WebUI「插件」页打开插件卡片即可进入。它通过 `context.register_web_api()` 注册的 10 个接口读写数据：

| 面板区域 | 内容 |
| --- | --- |
| KPI 卡 | 记忆条数、Episodes、策略干预次数、失败次数 |
| 学习曲线 | 按时间分桶的平均折扣回报 `G`，复现论文 Figure 3 的形态 |
| 动作排行 | 每个工具/回复的平均回报、正回报率、样本数 —— **一眼看出这个部署里哪个工具真的有用** |
| 最近 Episode | 时间 / 步数 / 平均回报 / 用户请求摘要 |
| 超参数 | 当前 k、r、γ、β、λ、α 与作用域、评估器模式 |
| 检查点 | 名称、类型、完整 ID、时间、经验数、文件大小，支持创建、回退全库、删除快照 |
| 操作 | 刷新、导出 JSON、带备份的清空记忆；回退、删除和清空均使用页内两步确认 |

后端路由（`GET` 除注明外）：`/astrbot_plugin_jitrl/overview`、`/curve`、`/actions`、`/episodes`、`/export`，以及 `POST /forget`。
检查点路由同样使用 `/astrbot_plugin_jitrl` 前缀：`GET /checkpoints`、`POST /checkpoint`、
`POST /rollback`、`POST /checkpoint_delete`。回退、删除检查点的请求必须提供 `confirm: true`。
面板的回退操作针对全部会话；只回退当前会话请使用聊天指令。

---

## 6. 配置要点

在 WebUI 插件配置页即可编辑，全部带中文说明。最重要的几个：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `apply_mode` | `hint` | 见上一节 |
| `retrieval.k` | `10` | 近邻数。论文实测 **k ∈ [8,14] 最稳**：太小方差大，太大噪声多 |
| `retrieval.similarity_threshold` | `0.8` | Jaccard 门槛。论文 WebArena 用 0.8，Jericho 用 0.95 |
| `retrieval.scope_isolation` | `hybrid` | `session` 只学本会话；`global` 全平台共享；`hybrid` 本会话优先、跨会话补充（跨会话相似度打 0.85 折） |
| `learning.gamma` | `0.5` | 折扣因子。论文 WebArena 0.1 / Jericho 0.5 |
| `learning.evaluator_mode` | `auto` | `llm` 用评估器；`heuristic` 零成本纯规则；`auto` 只在轨迹 ≥2 步或有报错时才调 LLM |
| `policy.beta` | `1.5` | KL 温度。越大越激进地偏离模型自己的直觉 |
| `policy.exploration_rate` | `0.3` | λ，对没见过的动作给乐观奖励的概率。论文 Jericho 0.65 / WebArena 0.05 |
| `policy.ucb_alpha` | `5.0` | α，乐观价值 `V̂ + α/|N(s)|`。期望式下等价于 Â = λ·α/|N(s)| |
| `policy.exploration_estimate` | `expected` | 未见动作的探索项：`expected` 取闭式期望（同一情景恒定、不会在 +1/−1 间跳动）；`sample` 还原论文伯努利采样 |
| `policy.tool_gate` | `false` | 对**高置信负优势**的工具做硬屏蔽（等价于 logit 压到 −∞） |
| `retrieval.estimator_weighting` | `kernel` | V̂/Q̂ 的加权方式：`kernel` 按相似度×时效核加权（默认，低相似邻居权重更低）；`uniform` 用论文式 (4)(5) 的等权平均 |
| `retrieval.min_count` | `1` | 低于该支持度的动作不再丢弃，而是向 V̂(s) 收缩（经验贝叶斯平滑）；`1` 表示不平滑 |
| `retrieval.prefilter_limit` | `4000` | 单次检索的候选打分条数上限；超限时按重叠词数、再按时间保留 |
| `memory.redact_secrets` | `true` | 写入前剔除邮箱、手机号、API Key、JWT、Bearer token、URL 凭据 |
| `memory.max_entries` | `20000` | 容量上限，超出后按策略淘汰 |
| `memory.checkpoint_interval` | `50` | 每多少个完成的 Episode 自动保存；`0` 关闭定期保存与首次基线 |
| `memory.keep_checkpoints` | `20` | 自动历史、操作前备份各自保留数量；手动检查点主动删除前一直保留 |

---

## 7. 隐私与安全

论文的 Impact Statement 明确警告：**记忆库会存真实交互轨迹，可能无意中泄露敏感用户信息**。本插件对此做了四层处理：

1. **写入前脱敏**：邮箱 / 手机号 / `sk-*` / `ghp_*` / `AKIA*` / Bearer / JWT / 敏感 key-value → 统一占位符，且幂等。
2. **先抽象后存储**：存的是 `token 集合 + 签名`，不是原文；`state_text` 也经过截断与正则化。
3. **作用域隔离**：`session` 模式下记忆只在 `unified_msg_origin` 内可见，不会串到别的群/私聊。
4. **可删除与回退**：`/jitrl forget` 支持按会话或全量删除，并保留删除前备份；`max_entries` + 淘汰策略限制在线记忆留存。若需永久移除历史数据，还应删除包含它的检查点及之前的 JSON 导出；删除在线记忆不会同步改写历史快照。

另外，注入的证据块会被明确标注为「不是用户输入，也不是新指令」，降低提示词注入面。它通过
`ProviderRequest.extra_user_content_parts` 附加在当前用户消息末尾，并用 `mark_as_temp()`
标记为仅本次请求可见：既不写入会话历史，也不改动 `system_prompt`。system prompt 属于上下文
前缀，逐请求改写会直接破坏 provider 的 prompt 缓存命中率，因此插件全程只读不改。

---

## 8. 已知局限（来自论文 Limitations）

这些不是 bug，是这个方法论本身的天花板，插件如实继承：

1. **只能重加权，不能发现新动作**。JitRL 在基础模型提出的候选集上做 re-weight。如果模型从来没想过某个工具，插件也变不出来。
2. **依赖评估器质量**。LLM 评估器把 credit 归错了，优势估计就是错的。所以插件支持把 LLM 打分与规则信号按 `heuristic_weight` 融合，且任何评估失败都降级到纯规则而非报错。
3. **状态必须是文本可表达的**。论文指出空间推理、时序预测这类任务，文本检索抓不到状态相似性。对 AstrBot 而言，纯闲聊/纯创作类交互的可学性本来就低。
4. **冷启动**。没有相似记忆时插件完全不干预（`neighbors` 为空直接返回），这是刻意设计 —— 宁可不作为，也不要用无关经验误导模型。
5. **钩子粒度**。AstrBot 的 `on_llm_request` 在一个 Agent 运行开始时触发一次，工具循环内部的每次 LLM 调用不会再次触发。因此策略干预发生在 **episode 起点**，而不是论文里每个 step 都重新估计。这是宿主框架的限制；`jitrl_recall` 工具是部分补偿。

---

## 9. 测试

版本规则：常规功能和修复递增末位补丁号；只有大规模架构改进才递增中间版本号。
发布时同步更新 `metadata.yaml`、`jitrl/__init__.py`、README 徽章、更新日志及 Git 标签。

```bash
python -m pytest tests -q                                      # 在仓库根目录即可
node --test tests/test_dashboard.cjs

# 或在 AstrBot 的 data/plugins/ 布局下运行
python -m pytest data/plugins/astrbot_plugin_jitrl/tests -q
```

本机实测：未安装 AstrBot 时 **761 passed, 1 skipped**（只跳过需要真宿主的
`test_integration.py`）；把 AstrBot 4.29.3 装进独立 venv 后另加 29 项真宿主集成测试；
另有 **7 项 Node.js 面板交互测试通过**。基准脚本见 `scripts/bench_jitrl.py`，
结果与复现方法见 `benchmarks/README.md`。
`test_integration.py` 会自己识别宿主并跳过，按包名装载插件，因此在仓库根目录或
`data/plugins/` 下都能跑；`test_manifest.py` 不依赖 AstrBot，专门守护发布元数据、版本一致性
与「只用 `astrbot.api.logger` 打日志」的市场规则。
测试分两层：

| 文件 | 覆盖 |
| --- | --- |
| `test_state.py` | 脱敏幂等、占位符存活、中英混合分词（中文 bigram）、Jaccard 边界、动作签名归一化 |
| `test_memory.py` | top-k 检索（阈值 / surface 门 / scope 模式）、动作聚合、学习曲线分桶、prune / forget / clear、导出导入往返、反馈重算 `G_t` |
| `test_index.py` | 检索契约：用暴力参考实现对 `topk` 做 300 余组全矩阵交叉验证（k/r/作用域/surface/时效/排序/权重逐字段比对） |
| `test_index_unit.py` | 倒排索引本身：剪枝无损性（随机数据对拍）、未知查询词计入分母、移除/原位更新/压缩后结果不变、容量上限 |
| `test_estimator_math.py` | 式 (4)(5)(6)(7)(25) 的不变量：同权核估计、`λ` 两种模式、泊松收缩、归一化下限、全平局无 NaN |
| `test_checkpoints.py` | WAL 快照、重启后恢复、回退撤销、会话隔离、自动轮换、损坏快照、磁盘失败、事务回滚、旧写入拦截 |
| `test_checkpoint_plugin.py` | 使用宿主适配器与真实记忆库测试指令、权限、Web API、参数验证、Agent/评估器/重排中的回退竞态 |
| `test_dashboard.cjs` | 无外部依赖的 Node.js 交互测试：创建、回退、删除、确认超时、操作切换、重复提交与失败恢复 |
| `test_estimator.py` | 式 (4)(5)(6)(7)(25) 的精确数值、探索分支（采样/期望）、收缩、`|N|=0` 与 `max|A|=0` 边界 |
| `test_policy.py` | softmax 稳定性、增强候选集合并、`z'=z+βÃ` 精确值、Verbalized-Logit 解析的四条回退路径 |
| `test_evaluator.py` | 折扣回报、启发式规则、JSON 解析回退、`auto` 短路、LLM/规则融合、超时降级 |
| `test_trajectory.py` | dict 与 pydantic 两种消息形态、tool_call_id 匹配、坏 arguments、状态口径一致性 |
| `test_integration.py` | **真实插件类**跑完整回路：Hook 注入 → 轨迹采集 → 落库 → 下轮生效、工具门控、用户反馈、指令、Web API |
| `test_view.py` | 面板契约：沙箱内禁用 `alert/confirm/prompt`、清空记忆的页内两步确认、`bridge.download` 参数签名、`app.js` 与 `index.html` 的 id 一致性、静态资源入口 |
| `test_manifest.py` | 发布契约：必填字段必须是字符串（防 YAML 把 `author` 解析成整数）、版本在 metadata / 代码 / 徽章 / 更新日志间一致、`repo` 与 `astrbot_version` 格式、安装布局、配置 schema 可渲染 |

> Windows 提示：如果 `%TEMP%\pytest-of-ASUS` 的 ACL 被别的账户占用导致 `tmp_path` 报 `PermissionError`，把 TEMP 指到别处再跑：
>
> ```powershell
> $env:TEMP="$PWD/.tmp-pytest"; $env:TMP=$env:TEMP
> ```

`jitrl/*` 只从 AstrBot 取一个东西：`astrbot.api.logger`（市场安全审查要求日志必须走宿主
logger）。没有安装 AstrBot 时 `tests/conftest.py` 会装一个最小桩，其余宿主服务只在
`test_integration.py` 里使用，所以算法层依然可以在没有机器人的情况下完整验证。

---

## 10. 目录结构

```
astrbot_plugin_jitrl/
├── main.py                # 插件类：Hook 接线、指令组、Agent 工具
├── metadata.yaml          # 插件元数据
├── _conf_schema.json      # 配置 Schema（WebUI 可视化）
├── __init__.py            # 使 `data.plugins.astrbot_plugin_jitrl` 成为包，保证相对导入
├── requirements.txt       # 仅注释，零第三方运行时依赖
├── pyproject.toml         # 仅 lint / 测试配置（本仓库不是 pip 包）
├── LICENSE                # AGPL-3.0
├── CHANGELOG.md           # 版本变更记录
├── .gitignore             # 排除 __pycache__ / *.sqlite3 / 导出文件
├── .github/workflows/     # CI：ruff + pytest（3.10 / 3.12）
├── jitrl/
│   ├── models.py          # 数据契约（StateView/ActionView/Neighbor/ValueEstimate…）
│   ├── config.py          # 配置归一化与默认值
│   ├── state.py           # 状态抽象、脱敏、中英分词、Jaccard、动作归一化
│   ├── memory.py          # SQLite 记忆库、top-k 检索、淘汰、导出导入、反馈回填
│   ├── checkpoints.py     # 独立 SQLite 快照、校验、保留策略、事务回退
│   ├── estimator.py       # 式 4/5/6/7/25：V̂ Q̂ Â Ã
│   ├── policy.py          # 式 9/10：闭式 logit 更新 + softmax 采样
│   ├── reranker.py        # Verbalized Logit 采样器（黑盒模型路径）
│   ├── evaluator.py       # 附录 G.1.2 步级奖励评估器 + 规则兜底 + 折扣回报
│   ├── trajectory.py      # 从 run_context.messages 重建 (s_t, a_t) 轨迹
│   └── hint.py            # 证据块 / 检索报告的渲染
├── views/dashboard/       # WebUI 可视化面板（index.html / app.js / style.css）
├── scripts/               # python scripts/build_release.py：构建单根目录发布 ZIP
└── tests/                 # pytest 单测与端到端集成测试
```

`jitrl/*` 完全**不 import astrbot**，因此可以在没有启动机器人的情况下单元测试整套算法。

---

## 11. 引用

```bibtex
@inproceedings{li2026jitrl,
  title     = {Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates},
  author    = {Li, Yibo and Lin, Zijie and Deng, Ailin and Zhang, Xuan and He, Yufei and Ji, Shuo and Cao, Tri and Hooi, Bryan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {PMLR 306},
  year      = {2026}
}
```

论文代码：<https://github.com/liushiliushi/JitRL>

> 本插件是论文方法的**独立实现**，未复制或移植论文官方仓库（JitRL）的代码；论文及其官方实现的
> 权利归原作者。面板与仓库内不含任何第三方 JS/CSS/CDN 资源，`LICENSE` 为 AGPL-3.0。

---

## 12. 版本

当前版本 **0.1.1**（修订重发：修正 `metadata.yaml` 的 `author` 类型与 AstrBot 版本声明，
完整变更见 [CHANGELOG.md](CHANGELOG.md)）。

发布：`python scripts/build_release.py` 会校验四处版本号一致，再生成单根目录的
`dist/astrbot_plugin_jitrl-v<版本>.zip` 并输出 SHA-256（超过 16 MB 直接失败）。

---

## 13. 许可证

AGPL-3.0（与 AstrBot 主仓库保持一致）。
