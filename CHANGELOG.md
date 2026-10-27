# 更新日志

本项目的所有重要变更都记录在这个文件里。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本规则：常规功能和修复递增末位补丁号；大规模架构改进才递增中间版本号。

## [0.1.3] - 2026-10-27

### 新增

- **按机器人实例隔离学习**：按 AstrBot 平台实例 ID 创建独立 SQLite 记忆库。检索、学习写入、统计、用户反馈缓存、运行计数、检查点、导出和清空都在当前机器人范围内；`global` / `hybrid` 仍只影响同一机器人内的跨会话检索。
- WebUI 新增机器人选择器，所有统计、导出、检查点、回退和清空操作只作用于所选机器人。
- 旧版共享 `memory.sqlite3` 与检查点在升级后迁移到第一个触发插件的机器人目录；其他机器人使用独立新库。

### 修复（2026-10-27 重发）

- 修正机器人实例识别：初版把 `get_self_id()` 纳入 `robots/<hash>` 实例键，但 AstrBot 多个适配器的 `self_id` 由消息内容决定（`qq_official` 群聊取首个机器人 mention 或字面量 `qq_official`、C2C 固定 `unknown_selfid`、频道取首个 mention），导致同一机器人在不同消息形态下被当成新机器人，学习数据被拆散。现在实例键只使用平台实例 ID，`self_id` 仅写入 `robot.json` 供展示与排查；缺少平台实例 ID 的宿主只回退到适配器类型名。
- 启动时自动合并初版遗留的 `robots/<platform_id:self_id>/`：按 `robot.json` 的 `platform_id` 分组，数据最丰富的目录提升为规范目录，其余归档到 `robots/.legacy/<robot-hash>/`（不删除）；规范库首次初始化时按 episode 幂等合并归档经验，归档导出不设行数上限，崩溃后重试不会重复。
- 面板机器人选择器与 `robot.json` 读取更稳健：损坏或隐藏目录不再影响列表，缺少 `label` 的旧条目回退显示平台实例 ID，`self_id` 为空时保留上次已知值。
- 增加相应回归测试，并修正真实 AstrBot 集成测试中过期的断言。

### 变更

- 机器人数据存放在 `robots/<robot-hash>/`，运行统计和状态/决策缓存也按机器人区分。
- 版本号保持 0.1.3（本次为重发修正，不改版本号）。

[0.1.3]: https://github.com/2841649220/astrbot_plugin_jitrl/releases/tag/v0.1.3

## [0.1.2] - 2026-10-27

### 新增

- **倒排索引检索（`jitrl/index.py`）**：把论文式 (24) 的 top-k Jaccard 从「每条记忆逐一求交」改成无损候选剪枝——长度过滤（`t·|q| ≤ |x| ≤ |q|/t`）、重叠词下界（至少 `ceil(t·|q|)` 个共同词）与前缀过滤（PPJoin），再用倒排表逐行精确校验。结果与全表暴力扫描逐字段一致（`tests/test_index.py` 以暴力参考实现做 300 余组交叉验证）。索引常驻内存（约 0.55 KB/条），支持墓碑删除、原位回报更新、自动压缩，容量上限 10 万条，超限自动退回精确 SQL 扫描。
- **两段式候选筛选**：第一次估计覆盖全部检索到的动作，第二次只对 |优势|/支持度最强的 |C| 个动作做式 (25) 归一化与决策。带强负优势的工具不再因为支持度低而在估计前就被丢弃。
- 新配置 `retrieval.estimator_weighting`（`kernel` 默认 / `uniform`）与 `policy.exploration_estimate`（`expected` 默认 / `sample`）。
- **决策缓存与状态缓存**：同一（会话，状态签名，记忆版本，工具集）在一次 Agent 回合内复用同一次策略决策；新经验、用户反馈、回退/删除都会立即失效缓存。状态抽象（脱敏 + 正则化 + 分词）也只算一次。
- **证据门控**：邻居少于 2 条或 max|Â| < 0.05 时不注入证据块（既省 token，也避免把归一化后的舍入噪声当成建议）；证据块文本压缩约四成。
- **评估器提示词预算**：单次步级评估的步骤描述总长上限 12000 字符，长轨迹按步数均分（每步下限 120 字符）。40 步轨迹的评估提示由约 28k 字符降到 12k 以内。
- 面板清空记忆接口要求 `confirm: true`（与检查点三条路由一致）；`/jitrl export` 不再把服务器绝对路径回显到聊天，且只保留最近 5 份导出。

### 变更与优化

- **检索语义**：`retrieval.prefilter_limit` 从「只扫描最新 N 条」改为「候选打分条数上限」。旧语义下，一条 100% 相似的老经验在 4000 条新经验之后永远检索不到；新实现先做内容候选，超限时按重叠词数、再按时间保留。
- **估计器一致性**：V(s) 与 Q̂(s,a) 现在始终使用同一套核权重。此前当某个动作组的相似度权重和为 0 时，Q̂ 会退化为等权均值而同一次调用的 V̂ 仍是加权均值，A = Q̂ − V̂ 实际在比较两个不同的估计量。
- **低样本平滑**：低于 `retrieval.min_count` 的动作由「硬阈值丢弃」改为向 V(s) 的经验贝叶斯收缩 `Q = (n·Q̄ + κ·V)/(n+κ)`（κ = min_count）。默认 `min_count = 1` 时行为不变。
- **探索项**：默认改为论文伯努利分支的闭式期望，中性基线取 Q = V（即 Â = λ·α/|N|）。论文的「概率 1−λ 时 Q = 0」会使 Â = −V：在正价值状态里把未尝试动作罚得比任何已知动作都重，在负价值状态里反而奖励它，且逐请求采样会让同一情景的建议在 +1 与 −1 之间跳动。`exploration_estimate = "sample"` 可还原论文原式用于消融。
- **归一化下限**：式 (25) 增加 max|Â| < 1e-3 时输出 0 的保护，避免把所有回报完全相同的噪声放大成 ±1 的策略抖动；未归一化的尺度通过 `ValueEstimate.max_abs_advantage` 暴露给证据门控。
- **SQLite**：新增 `experiences(episode_id)`、`(surface, created_at)`、`(scope, created_at)` 索引（原先每次写 Episode 的两条 DELETE 都是全表扫描）；行数改为缓存计数（原先每次写入都 `COUNT(*)` 全表）；`cache_size` 与 `temp_store` 调优；删除/回退改为索引墓碑 + 自动压缩，不再整表重建。
- `jaccard()` 不再构造并集集合：|a∪b| = |a| + |b| − |a∩b|。
- 宿主 `TextPart` 改为惰性导入：宿主迁移这个私有类时插件仍可加载，只是跳过注入；此前会在 import 阶段直接加载失败。
- 文档与发布：新增 `docs/compliance-audit.md`（AstrBot 插件规范与插件市场审查逐条对照）与 `docs/runtime-compat.md`（对 AstrBot v4.29.0 / v4.29.3 / master 的 32 项宿主 API 逐项核验，无一处不匹配；`astrbot_version: ">=4.29,<5"` 经源码确认准确），以及可复现基准 `scripts/bench_jitrl.py` + `benchmarks/`。`docs/`、`benchmarks/` 通过 `.gitattributes` 排除在源码归档之外。
- 版本号 0.1.1 → 0.1.2（常规功能与修复，遵循补丁号规则）。

[0.1.2]: https://github.com/2841649220/astrbot_plugin_jitrl/releases/tag/v0.1.2

## [0.1.1] - 2026-10-26

### 新增

- 独立 SQLite 检查点：手动命名、首次基线、按 Episode 数自动保存；快照包含已提交的 WAL 数据，校验后原子发布。
- 全库或指定会话事务回退；回退、清空和定向删除前自动备份，支持恢复备份以撤销操作。
- 自动历史和操作前备份分别轮换，默认各 20 份；手动检查点保留到主动删除。
- 管理员指令 `checkpoint`、`checkpoints`、`rollback`、`checkpoint_delete`，以及面板创建、回退和删除快照入口。
- 配置 `memory.checkpoint_interval`（默认 50，0 关闭定期保存/初始基线）和 `memory.keep_checkpoints`（默认 20）。
- 30 项恢复测试、24 项插件行为测试、7 项 Node.js 面板交互测试；CI 自动验证面板交互。

### 修复与优化

- 删除和回退后，拒绝旧 Agent 请求、后台评估和排队反馈的写入，清理反馈和策略缓存，防止恢复后的记忆再次被旧结果污染。
- 拒绝错误的删除作用域、非法检查点 ID、缺少明确确认的 Web 请求及损坏/不兼容的快照。
- 备份失败时中止删除和回退；回退写入失败时整体回滚，不留下半恢复数据。
- 面板并行加载统计，忽略过期刷新响应；变更处理中禁用重复操作，回退和删除使用沙箱兼容的页内两步确认。
- 发布元数据修正：`metadata.yaml` 的 `author` 改为带引号字符串，避免 YAML 解析为整数导致 AstrBot 元数据校验失败、插件身份降级为 `unknown/unknown` 与面板入口 404；`astrbot_version` 由 `">=4.16,<5"` 修正为 `">=4.29,<5"`（`filter.on_agent_done` 需 4.24+，`views/` 面板目录需 4.29+）。
- 集成测试改为按包名装载插件，不再依赖 `data/plugins/` 布局；新增 `tests/test_manifest.py` 守护发布契约（字段类型、版本链、`repo` 与版本范围格式、安装布局），CI 安装 `pyyaml` 后运行。
- 新增 `scripts/build_release.py`：校验版本链后生成单根目录发布 ZIP，拒绝超过 16 MB 的包。
- 配置 schema 清理：修正 `jitrl_note` 描述，移除未实现的 `debug.expose_decision` 与宿主不渲染的 `apply_mode.subtitles`，`float` 默认值统一为浮点字面量。
- README 更新安装要求（AstrBot `>=4.29`）、测试说明与「独立实现、未移植论文官方代码」许可声明；`.gitignore` 补充 `*.db` 家族。
- 市场复审修复（日志）：`jitrl/memory.py`、`jitrl/evaluator.py`、`jitrl/checkpoints.py`、`jitrl/reranker.py` 全部改用 `from astrbot.api import logger`，删除 `import logging` 与 `logging.getLogger()` 兜底；`StepRewardEvaluator` / `VerbalizedReranker` 不再接收外部 logger。新增契约测试禁止插件源码再引入标准库 `logging`。
- 市场复审修复（缓存）：经验证据块不再改写 `req.system_prompt`，改为 `req.extra_user_content_parts.append(TextPart(...).mark_as_temp())`，附加到当前用户消息且不写入会话历史；system prompt 作为上下文前缀逐字节不变，provider 的 prompt 缓存命中率不再被按请求改写破坏。
- 集成测试新增断言：注入落在用户消息的临时内容块中、`system_prompt` 保持原值、`_no_save` 已置位；另加一项不依赖宿主的注入路径测试。
- 版本号保持 `0.1.1`（本次为修订重发），维持零第三方运行时依赖。

## [0.1.0] - 2026-10-26

首个公开版本：把论文 *Just-In-Time Reinforcement Learning: Continual Learning in
LLM Agents Without Gradient Updates*（Li et al., ICML 2026）的推理时策略优化完整
落地到 AstrBot 的内置 Agent —— 无梯度更新、无微调、无 GPU。

### 新增

- **状态与动作抽象**（`jitrl/state.py`）：写入前脱敏、标识符正则化、中英混合 token
  （中文 bigram）、Jaccard 相似度，以及工具调用 / 回复的动作签名归一化。
- **非参数记忆库**（`jitrl/memory.py`）：SQLite 表 `experiences`，top-k 检索
  （相似度阈值 + surface 门 + `session` / `global` / `hybrid` 作用域）、动作聚合、
  学习曲线分桶、容量淘汰、导出导入往返，以及用户反馈触发的 `G_t` 重算。
- **价值与优势估计**（`jitrl/estimator.py`）：式 (4)(5)(6)(7)(25) 的
  `V̂ / Q̂ / Â / Ã`，含未见动作的乐观探索分支与 `|N|=0` 边界处理。
- **闭式策略更新**（`jitrl/policy.py`）：`z' = z + β·Ã` 加性 logit 偏置、数值稳定的
  softmax、增强候选集合并与动作选择。
- **Verbalized Logit 采样**（`jitrl/reranker.py`）：黑盒模型拿不到 logits 时的
  置信度 → logit 路径，含四条解析回退。
- **步级奖励评估器**（`jitrl/evaluator.py`）：附录 G.1.2 的 −3..+3 打分、规则信号融合、
  `auto` 短路、超时降级与折扣回报计算。
- **轨迹重建**（`jitrl/trajectory.py`）：从 `run_context.messages` 还原 `(s_t, a_t)`，
  兼容 dict 与 pydantic 两种消息形态。
- **证据块渲染**（`jitrl/hint.py`）：注入带数字优势估计的 `<jitrl_experience>` 块，
  并显式标注「不是用户输入，也不是新指令」。
- **插件接线**（`main.py`）：`on_llm_request` / `on_agent_done` Hook、`/jitrl` 指令组、
  `jitrl_recall` Agent 工具，以及 6 个 Web API 路由。
- **WebUI 面板**（`views/dashboard/`）：KPI 卡、学习曲线、动作排行、最近 Episode、
  当前超参数，以及刷新 / 导出 / 清空记忆操作（沙箱内无模态框，清空走页内两步确认）。
- **测试**（`tests/`）：252 项算法与视图测试 + 28 项 AstrBot 集成测试；
  CI 在 Python 3.10 / 3.12 上跑 `ruff` 与 `pytest`。

### 变更

- 版本号从开发期的 `1.0.x` 归一到首个公开版本 `0.1.0`：
  `metadata.yaml` 与 `jitrl/__init__.py` 的 `__version__` 现在保持一致。

[0.1.0]: https://github.com/2841649220/astrbot_plugin_jitrl/releases/tag/v0.1.0
[0.1.1]: https://github.com/2841649220/astrbot_plugin_jitrl/releases/tag/v0.1.1
