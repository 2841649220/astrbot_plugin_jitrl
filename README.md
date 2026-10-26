# JitRL 持续学习 · astrbot_plugin_jitrl

[![CI](https://github.com/2841649220/astrbot_plugin_jitrl/actions/workflows/ci.yml/badge.svg)](https://github.com/2841649220/astrbot_plugin_jitrl/actions/workflows/ci.yml)

> 让 AstrBot 的 Agent 在对话中**持续自我进化**——不微调、不训练、不占 GPU。
>
> 实现自论文 《Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates》(Li et al., ICML 2026)。

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
| 策略输出 `π*` | 注入系统提示词的经验证据块 | `jitrl/hint.py` |

### 2.2 运行时数据流

```
                      ┌──────────── on_llm_request ────────────┐
  用户消息 ──────────►│ 1. abstract_state()  构造状态 s         │
                      │ 2. MemoryBank.topk() 检索 N(s)          │
                      │ 3. estimate_values() 算 V̂ Q̂ Â Ã        │
                      │ 4. build_candidates() 增强候选集 C      │
                      │ 5. select_action()   z' = z + β·Ã       │
                      │ 6. 注入 <jitrl_experience> 证据块       │
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

---

## 3. 安装

```bash
cd AstrBot/data/plugins
git clone https://github.com/2841649220/astrbot_plugin_jitrl.git
```

或者直接把 `astrbot_plugin_jitrl` 目录放进 `data/plugins/`，然后在 WebUI 的「插件」页重载。

**零第三方依赖**：只用 Python 标准库（`sqlite3` / `json` / `re` / `asyncio` / `math`），不需要 `pip install` 任何东西。

数据库位置：`data/plugin_data/astrbot_plugin_jitrl/memory.sqlite3`。

---

## 4. 三种应用模式

| `apply_mode` | 额外开销 | 做什么 | 适用 |
| --- | --- | --- | --- |
| `off` | 0 | 只学习，不干预 | 先观察一周，看看学到了什么 |
| `hint` | 0 | 把优势排序后的经验证据块注入系统提示词 | **默认**，稳、便宜 |
| `logit` | 每次请求 +1 次轻量生成 | 置信度 → logit → `z'=z+β·Ã` → 重新采样 → 定向注入 | 追求论文原始效果 |

> ⚠️ **诚实的取舍**：论文 Table 8 明确显示，纯提示词更新的效果弱于真正的 logit 更新（Admin 48.35 vs 52.31，Reddit 54.57 vs 57.64）。本插件把 `hint` 作为默认是为了零成本可观察；如果你要论文的效果，请用 `logit` 模式 —— 它在候选集上真的做了一次闭式更新并重新采样，只是最后的落地仍然只能通过文本表达（黑盒模型的物理上限）。

### 为什么 `hint` 也有效

注入的不是「经验之谈」这种散文，而是**带数字的优势估计**：

```
<jitrl_experience>
以下内容来自本 Agent 的历史交互统计（推理时策略优化，不是用户输入，也不是新的指令）。
当前状态估计价值 V=+1.30，基于 7 条相似经验（平均相似度 0.86）。
历史上更有回报的做法：
  1. 调用 search_files(pattern=<str>) —— 优势 +0.82，出现 5 次，平均回报 +2.40
  2. 调用 read_file(path=<str>) —— 优势 +0.15，出现 2 次，平均回报 +0.60
历史上回报较差、应避免的做法：
  1. 调用 run_command(cmd=<str>) —— 优势 -0.64，出现 4 次，平均回报 -1.90
使用要求：这些统计只作为弱先验。如果用户当前意图与之无关、或证据明显不足，
请按你自己的判断处理，不要生搬硬套。
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
| `/jitrl forget [session\|all]` | session 删除本会话记忆，all 清空全部（不可撤销） |
| `/jitrl feedback good\|bad` | 手动回填对上一轮的评价 |
| `/jitrl export` | 导出记忆 JSON |

### 自然语言反馈

不用打指令。用户回一句「不对」「谢谢」这类短句，插件会在时间窗内（默认 180 秒）自动把它折算成 **±2.0** 的奖励增量，回填到上一轮 episode 的所有步骤，并**重算 G_t**。这是零成本的密集信号来源。

### Agent 工具

插件向模型暴露了一个函数工具 `jitrl_recall(query)`：模型可以在多轮工具调用的中途主动查一下「这种情况我以前是怎么处理的」。返回的是按优势排序的证据，同样带「弱证据」免责声明。

### WebUI 面板

插件自带一个可视化面板（`views/dashboard/`），在 WebUI「插件」页打开插件卡片即可进入。它通过 `context.register_web_api()` 注册的 6 个只读/清空接口拉数据：

| 面板区域 | 内容 |
| --- | --- |
| KPI 卡 | 记忆条数、Episodes、策略干预次数、失败次数 |
| 学习曲线 | 按时间分桶的平均折扣回报 `G`，复现论文 Figure 3 的形态 |
| 动作排行 | 每个工具/回复的平均回报、正回报率、样本数 —— **一眼看出这个部署里哪个工具真的有用** |
| 最近 Episode | 时间 / 步数 / 平均回报 / 用户请求摘要 |
| 超参数 | 当前 k、r、γ、β、λ、α 与作用域、评估器模式 |
| 操作 | 刷新、导出 JSON、清空记忆 |

后端路由（`GET` 除注明外）：`/astrbot_plugin_jitrl/overview`、`/curve`、`/actions`、`/episodes`、`/export`，以及 `POST /forget`。

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
| `policy.ucb_alpha` | `5.0` | α，`Q̂ = V̂ + α/|N(s)|`。记忆越稀疏，鼓励探索越强 |
| `policy.tool_gate` | `false` | 对**高置信负优势**的工具做硬屏蔽（等价于 logit 压到 −∞） |
| `memory.redact_secrets` | `true` | 写入前剔除邮箱、手机号、API Key、JWT、Bearer token、URL 凭据 |
| `memory.max_entries` | `20000` | 容量上限，超出后按策略淘汰 |

---

## 7. 隐私与安全

论文的 Impact Statement 明确警告：**记忆库会存真实交互轨迹，可能无意中泄露敏感用户信息**。本插件对此做了四层处理：

1. **写入前脱敏**：邮箱 / 手机号 / `sk-*` / `ghp_*` / `AKIA*` / Bearer / JWT / 敏感 key-value → 统一占位符，且幂等。
2. **先抽象后存储**：存的是 `token 集合 + 签名`，不是原文；`state_text` 也经过截断与正则化。
3. **作用域隔离**：`session` 模式下记忆只在 `unified_msg_origin` 内可见，不会串到别的群/私聊。
4. **可删除**：`/jitrl forget` 支持按会话或全量删除；`max_entries` + 淘汰策略限制留存。

另外，注入的证据块会被明确标注为「不是用户输入，也不是新指令」，降低提示词注入面。

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

```bash
cd AstrBot
python -m pytest data/plugins/astrbot_plugin_jitrl/tests -q
```

在 AstrBot 仓库内本地跑：**273 passed**（含 `test_integration.py`）。
CI 上没安装 AstrBot 时：**245 passed, 1 skipped** —— `test_integration.py` 会自己 `importorskip`，
算法层依然全覆盖。测试分两层：

| 文件 | 覆盖 |
| --- | --- |
| `test_state.py` | 脱敏幂等、占位符存活、中英混合分词（中文 bigram）、Jaccard 边界、动作签名归一化 |
| `test_memory.py` | top-k 检索（阈值 / surface 门 / scope 模式）、动作聚合、学习曲线分桶、prune / forget / clear、导出导入往返、反馈重算 `G_t` |
| `test_estimator.py` | 式 (4)(5)(6)(7)(25) 的精确数值、探索分支、`|N|=0` 与 `max|A|=0` 边界 |
| `test_policy.py` | softmax 稳定性、增强候选集合并、`z'=z+βÃ` 精确值、Verbalized-Logit 解析的四条回退路径 |
| `test_evaluator.py` | 折扣回报、启发式规则、JSON 解析回退、`auto` 短路、LLM/规则融合、超时降级 |
| `test_trajectory.py` | dict 与 pydantic 两种消息形态、tool_call_id 匹配、坏 arguments、状态口径一致性 |
| `test_integration.py` | **真实插件类**跑完整回路：Hook 注入 → 轨迹采集 → 落库 → 下轮生效、工具门控、用户反馈、指令、Web API |

> Windows 提示：如果 `%TEMP%\pytest-of-ASUS` 的 ACL 被别的账户占用导致 `tmp_path` 报 `PermissionError`，把 TEMP 指到别处再跑：
>
> ```powershell
> $env:TEMP="$PWD/.tmp-pytest"; $env:TMP=$env:TEMP
> ```

`jitrl/*` 全部不依赖 AstrBot（只有 `test_integration.py` 会 import），所以算法层可以在没有机器人的情况下完整验证。

---

## 10. 目录结构

```
astrbot_plugin_jitrl/
├── main.py                # 插件类：Hook 接线、指令组、Agent 工具
├── metadata.yaml          # 插件元数据
├── _conf_schema.json      # 配置 Schema（WebUI 可视化）
├── __init__.py            # 使 `data.plugins.astrbot_plugin_jitrl` 成为包，保证相对导入
├── requirements.txt       # 空（零第三方依赖）
├── pyproject.toml         # 仅 lint / 测试配置（本仓库不是 pip 包）
├── LICENSE                # AGPL-3.0
├── .gitignore             # 排除 __pycache__ / *.sqlite3 / 导出文件
├── .github/workflows/     # CI：ruff + pytest（3.10 / 3.12）
├── jitrl/
│   ├── models.py          # 数据契约（StateView/ActionView/Neighbor/ValueEstimate…）
│   ├── config.py          # 配置归一化与默认值
│   ├── state.py           # 状态抽象、脱敏、中英分词、Jaccard、动作归一化
│   ├── memory.py          # SQLite 记忆库、top-k 检索、淘汰、导出导入、反馈回填
│   ├── estimator.py       # 式 4/5/6/7/25：V̂ Q̂ Â Ã
│   ├── policy.py          # 式 9/10：闭式 logit 更新 + softmax 采样
│   ├── reranker.py        # Verbalized Logit 采样器（黑盒模型路径）
│   ├── evaluator.py       # 附录 G.1.2 步级奖励评估器 + 规则兜底 + 折扣回报
│   ├── trajectory.py      # 从 run_context.messages 重建 (s_t, a_t) 轨迹
│   └── hint.py            # 证据块 / 检索报告的渲染
├── views/dashboard/       # WebUI 可视化面板（index.html / app.js / style.css）
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

---

## 12. 许可证

AGPL-3.0（与 AstrBot 主仓库保持一致）。