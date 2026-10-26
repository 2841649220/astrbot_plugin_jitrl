# 更新日志

本项目的所有重要变更都记录在这个文件里。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
