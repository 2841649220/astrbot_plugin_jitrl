# 更新日志

本项目的所有重要变更都记录在这个文件里。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本规则：常规功能和修复递增末位补丁号；大规模架构改进才递增中间版本号。

## [0.1.1] - 2026-10-26

### 新增

- 独立 SQLite 检查点：手动命名、首次基线、按 Episode 数自动保存；快照包含已提交的 WAL 数据，校验后原子发布。
- 全库或指定会话事务回退；回退、清空和定向删除前自动备份，支持恢复备份以撤销操作。
- 自动历史和操作前备份分别轮换，默认各 20 份；手动检查点保留到主动删除。
- 管理员指令 `checkpoint`、`checkpoints`、`rollback`、`checkpoint_delete`，以及面板创建、回退和删除快照入口。
- 配置 `memory.checkpoint_interval`（默认 50，0 关闭定期保存/初始基线）和 `memory.keep_checkpoints`（默认 20）。
- 30 项恢复测试、23 项插件行为测试、7 项 Node.js 面板交互测试；CI 自动验证面板交互。

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
