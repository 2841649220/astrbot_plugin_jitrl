# V0.1.4 / V0.2.2 主线研究与优化

日期：2026-10-28。V0.1.3 主线的检索、学习与策略路径先行修正，再迁移到
`v0.2.0-neural-lens` 的 V0.2.2。沿用原有 SQLite 格式与配置键。

## 研究依据与范围

[JitRL 原论文 v3](https://arxiv.org/html/2601.18510v3) 的式 (4)–(7)、附录 C、F
把近邻状态、动作有效性和奖励归属放在价值估计之前。由此得到本次工程判断：
先保证取对经验、只给当前回合计分，再加速已有估计器，比增加探索强度更有依据。
论文的真实 logit 更新与本插件的临时文本提示仍有区别；这些修正不能当作远端模型的
logit 改进或真实任务成功率提升。

宿主依据为 [AstrBot v4.29.0 请求构建](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/astr_main_agent.py)
和 [Agent 完成钩子](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/astr_agent_hooks.py)：
请求会带入会话历史，完成钩子提供运行消息。只跳过历史中的 user/system 消息，不能隔离历史动作。
本次使用最后一条 user 消息的位置确定当前回合；调用方直接传入无 user 标记的单轮轨迹仍受支持。

## 发现与修正

| 路径 | 原问题 | 当前行为 |
| --- | --- | --- |
| 候选预算 | 先按重叠数截断，再按 Jaccard 排序，长状态会挤掉更准确的短状态 | 所有可能过阈值的候选参与精确排名；只保留赢家堆 |
| 新会话 | scope 尚未被索引时，session 未过滤其他会话，hybrid 未打折 | session 返回空；hybrid 对所有其他会话正常乘 0.85 |
| SQL 降级 | 仅扫描最新 N 行，较早的精确匹配可能永久不可见 | 分批扫描全部符合 scope/surface 的行，用同一排名键选赢家 |
| 零阈值 | 强制走有限 SQL 窗口 | 索引支持零重叠行；零阈值也能保持精确排名 |
| 学习轨迹 | 旧轮工具和回复被重新归属于当前任务，甚至占满步数预算 | 只重建当前轮；复用的 tool_call_id 不会串到旧工具结果 |
| 奖励信号 | 成功 JSON 的正文出现 error/timeout 也会被判失败；显式错误标记可能丢失 | 先读结构化成功/错误字段，保留 is_error/isError；无结构化信号时沿用原启发式 |
| 终止回复 | 后续还有工具调用时，较早的普通 assistant 文本仍可能成为最终动作 | 后续有效工具调用撤销较早的中间回复 |
| 工具门控 | 微小差异归一化后也会屏蔽工具；某个坏参数签名会连带屏蔽好的签名 | 原始负优势至少 0.05，且该工具所有已观察签名都需达到负优势和支持度要求；始终保留可用工具 |

举例：查询含 10 个 token，A 命中其中 9 个且没有额外 token，J=0.9；B 命中全部
10 个但还含 8 个额外 token，J=10/18。原来的重叠截断偏爱 B；现在 A 必须排在 B 前面。

`retrieval.prefilter_limit` 保留原键、默认值和最大返回条数的兼容限制。它现在约束 SQL
单批读取数量与返回的 `min(k, prefilter_limit)`，**不再是打分次数或耗时的硬上限**。
完整候选排名使低预算、低阈值或超出索引容量时也不会静默损失正确近邻。

## 加速方式与正确性边界

前缀过滤、长度过滤保持无损。未知 query token 仍计入 Jaccard 分母，只从需要扫描的
已知 token 前缀长度中扣除。若倒排列表的并集已覆盖全部位置，即可停止合并后续列表。

设查询 token 数为 Q、已知查询 token 数为 K、候选长度为 L，最大可能重叠
`u=min(K,L)`，因此 `J <= u/(Q+L-u)`。应用同一跨会话倍率后，将此上界与
当前第 k 名的 `(similarity, created_at, id)` 比较，不能超越者不必再求交。
比较包括时间和 ID，因而同分且更新的记录不会被误剪。

真正需要求交时使用整数 array 切片与集合求交，减少 Python 逐 token 循环。
[heapq](https://docs.python.org/3/library/heapq.html) 维护最多 k 个赢家，时效权重和
IndexHit 只为最终赢家生成。候选集合和位置排序仍使用 O(C) 临时空间，不能把整个
查询的内存写成 O(k)，也不保证所有数据分布下都有次线性耗时。无索引的完整 SQL
扫描仍随库大小增长，只将读取内存限制为批量大小加 k 条赢家。

## 同机测量

同一 Python 3.13.12 / Windows 机器，同一冻结负载。常规脚本每场景 40 次查询；
纯索引密集负载每场景 25 次查询、3 次预热。以下均为 p50，不包含远端模型耗时。
常规表的 N=20000 指 20000 个合成 Episode，本次实际存储 37777 条经验。

| 场景 | V0.1.3 | 优化后 | 比例 |
| --- | ---: | ---: | ---: |
| 常规检索，N=20000，阈值 0.8 | 0.543 ms | 0.409 ms | 1.33× |
| 宽松检索，N=20000，阈值 0.5 | 13.004 ms | 5.138 ms | 2.53× |
| 请求前段，N=20000，阈值 0.8 | 0.599 ms | 0.419 ms | 1.43× |
| 密集同分索引，20000 行、100 token、阈值 0.5 | 262.558 ms | 10.451 ms | 25.12× |
| 长状态干扰，5000 行、预算 32 | 101.245 ms | 2.786 ms | 36.34× |

长状态干扰场景的 recall@10 从 0 到 1，修正后六个纯索引场景的顺序与暴力 oracle
完全一致。新会话 hybrid 场景原来的 ID recall 已为 1，但相似度未打折，仍不算精确。
测量基线为 `ddeb046`，共享优化提交为 `9254f32`；随后补充了索引压缩后残留词表项的
边界修正与回归：没有倒排表的 token 按未知 token 处理，但仍计入查询分母。
这里的 recall 和时间均来自合成输入，不能代替线上任务成功率、用户满意度或因果收益。
常规脚本的请求前段是核心检索/估计/渲染近似，并非完整宿主钩子，也未计入完整多阶段
候选选择、缓存命中、网络与异步调度的全部成本；写入和状态抽象未作性能优化。

开发仓库中保留 [常规 before](../benchmarks/v014-before.json)、
[常规 after](../benchmarks/v014-after.json)、
[精确性 before](../benchmarks/v014-retrieval-before.json)、
[精确性 after](../benchmarks/v014-retrieval-after.json)。

```powershell
python scripts/bench_jitrl.py --no-checkpoints --queries 40 --n 1000,5000,20000 --json benchmarks/v014-after.json
python scripts/bench_retrieval_quality.py --json benchmarks/v014-retrieval-after.json
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
node --test tests/test_dashboard.cjs
```

Windows 的系统临时目录若不可访问，可先创建 `.tmp-pytest`，再给 pytest 指定一个新的
`--basetemp .tmp-pytest/<run-name>`。测试覆盖索引与强制 SQL 路径的跨会话、平台、零阈值、
小预算、排序与权重，另覆盖历史轨迹、结构化结果和工具门控。真实 AstrBot 集成测试在
未安装宿主时跳过；宿主适配器测试独立运行。

## 向 V0.2.2 迁移

迁移共享核心与必要的门控接线，不引入新的神经主干或配置默认值。新采集的轨迹采用
当前轮边界；已保存的历史检索快照、奖励结算与检查点不回写。神经训练仍为可选，
pending 奖励不成为监督目标，四层 MiTA 的 landmark 压缩、可复用 top-k KV experts 与
共同归一化路由保持原样。V0.1.4 的运行依赖不包含 PyTorch。
