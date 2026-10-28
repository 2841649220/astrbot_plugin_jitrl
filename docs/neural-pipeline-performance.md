# V0.2.3 神经流水线：性能与证据效果

日期：2026-10-28。基线为 `v0.2.2` / `0e0a57c`，改动只落在
`v0.2.0-neural-lens`；V0.1.4 主线保持原样。

## 先定位瓶颈

原实现的成本主要在 MiTA 之前：每次检索重新散列历史状态，SQLite 同时读取排名
根本用不到的历史快照和提示；`make_batch` 又对重复历史经验逐次编码，并产生大量
小张量与切片写入。默认四层 MiTA 的预编码 CPU 前向约 34 ms，而基线的 2048 条
记录检索与 16 样本编码各约 2 秒。

本次没有改动 `attention.py` / `mita_grouped.py`，四层 MiTA 的 landmark 压缩、
复用 top-k KV experts 与共享/原始分支的共同归一化继续保留。

## 实现

- 检索 SQL 只投影状态、动作与已结算奖励所需字段。`replay` 单独读取完整历史
  snapshot/rendered/checkpoint 字段，不把当前重检索结果当作历史训练输入。
- 每个 LensStore 有最多 4096 项的 LRU，以作用域、维度、完整状态 JSON 为键，
  只保存不可变状态、冻结向量和事件集合。它不缓存奖励、结算、候选决策或检索快照。
  一次检索内相同的特征文本再去重，重复状态不重复散列。
- reset 按作用域清理缓存，close 清理全部缓存；计算结束时核验 epoch，重置前启动的
  迟到 worker 不得回填缓存。空字符串 reset 被拒绝，`None` 才表示显式全量范围。
- 批量编码只在当前调用内部按作用域复用特征与 Engram 地址；每个张量字段一次转换。
  逐行的回报、可靠度、相似度、时效、PRE、mask 与原始记忆顺序都分别构造。
- 状态特征文本使用同一 `state_feature_text` 定义，保持原来的 embedding 版本与数值。
  没有新增运行依赖，没有在服务路径修改宿主线程数或下载权重。

## 证据效果修正

词面和结构两个半池经常选中同样的记忆，原实现没有补齐剩余预算。现在保留两个池
选出的全部成员，再按综合相似度补齐与排序。该过程仍在既定的 `replay_limit` 已结算
窗口内进行，不表示全库精确近邻或真实任务召回率。合成 2048 条记录场景中，旧版
只返回 98 条，当前能返回 128 条；完全不同文本的对照为 125 → 128 条。

PRE 只展示当前可用动作的经验，最多两条且来自不同 episode。`neural.min_support`
和 POST `CandidateScore.support` 按不同已结算 episode ID 计数，零相似度或零可靠度
不增加支持。同一轮重复调用四次，在门槛为 3 时仍只有一份回合支持，不能触发干预。
不同 episode ID 不保证统计独立；这些门控也不是概率置信界。

POST 的经验估计使用 `w = similarity × reliability`。每个 episode 累加
`mass=Σw`、`total=ΣwG`、`steps=n`，再以 `mass/n` 为该回合权重计算回合均值：
`value = Σ(total/n) / Σ(mass/n)`。状态基线 V 与按动作分组的经验 Q 使用同一公式。
同一回合重复相同调用不会增加该回合票重；低可靠来源仍受降权。原有的本地候选值
融合、残差代理门控、最大调整幅度与 provenance 检查保留。

新增 POST 审计元数据 `support_unit=distinct_episode`、`estimator=episode-kernel-v1`，
便于与旧版按行计数的事件区分。历史 POST 和训练快照不会重写。

宿主快时钟测试还发现：反馈刚提交时，后续 `time.time()` 可能得到同一时间戳。
隐式实时 `replay()` 现在包含等于本次读取时间的已提交结算；显式历史截止点、神经
检索与训练 cutoff 仍使用严格 `< cutoff`，未来结算和 pending 事实仍被排除。

## 同机测量

Windows / Python 3.13.12；每场景 5 次热查询取 p50，冷查询单独测一次；编码与模型
计时固定一个 CPU 线程。检索包括 SQLite、特征和快照构建；编码用两版相同的固定
128 条历史输入，避免把候选数量变化算成编码提速。

| 场景 | V0.2.2 | V0.2.3 | 解读 |
| --- | ---: | ---: | --- |
| 2048 条、重复状态，冷检索 | 1986.73 ms | 221.82 ms | 32 种重复特征文本，约 8.96× |
| 同上，热检索 p50 | 1990.73 ms | 132.60 ms | 约 15.01×，返回数 98 → 128 |
| 2048 条、不同文本，冷检索 | 2007.60 ms | 1942.35 ms | 仍需首次散列，无显著改善 |
| 同上，热检索 p50 | 2019.45 ms | 133.85 ms | 约 15.09× |
| 单样本、128 记忆编码 | 131.09 ms | 31.02 ms | 约 4.23× |
| 16 样本、128 记忆编码 | 2094.13 ms | 87.55 ms | 约 23.92× |
| 四层 MiTA、预编码 CPU 前向 | 33.68 ms | 34.94 ms | 同量级，不声称注意力内核加速 |

热检索的 Python 临时分配峰值由约 39.75 MiB 降至 3.41 MiB。此项在缓存预热后用
tracemalloc 单独测量，不包含常驻 LRU、SQLite、PyTorch 或 CUDA 分配，不能解释成
进程总内存下降同样比例。LRU 另有常驻成本，缓存未命中或超过容量时需要重新计算。

本负载包含重复文本和跨训练样本共用的历史记忆；去重收益随真实重复率变化。这里
没有网络等待、远端模型或真实任务成功率测量，不把这些延迟比或合成 MSE 写成线上
任务收益、远端 logits 或因果 uplift。

开发仓库报告：[before](../benchmarks/v023-pipeline-before.json)、
[after](../benchmarks/v023-pipeline-after.json)。报告保留源代码指纹与输入描述；基线
由同一脚本读取本地 `v0.2.2` 源码归档运行。

## 数值与兼容验证

[编码对照](../benchmarks/v023-encoder-equivalence.json) 覆盖 float32/float64 默认类型、
空记忆、128 记忆、不同事件长度、不同作用域和 ragged batch。相同历史输入的所有
张量字段逐字节一致；旧 state_dict 可严格加载。CPU 模型输出误差为 0，CUDA 最大
误差约 4.77e-7，四层 MiTA / 零层 GQA 的配置一致。

[CPU](../benchmarks/v023-neural-smoke-cpu.json) 与
[CUDA](../benchmarks/v023-neural-smoke-cuda.json) 均完成正式 384d、四层 grouped MiTA、
40 条合成事实、20 步训练、原子保存重载和后续候选评分闭环。演示门控刻意宽松，
只能证明机械链路，不代表真实任务收益或推荐部署阈值。

回归覆盖候选池填充、缓存上限与 reset 并发、奖励修订的历史视图、实时反馈同时间戳、
episode 重复、来源权重、不可用工具、逐行元数据和特征内存隔离。神经权重格式不变，
不要求为本次更新强制重训；旧模型对新增候选/门控的业务效果仍应在实际任务上观察。

```powershell
python scripts/bench_neural_pipeline.py --json benchmarks/v023-pipeline-after.json
python scripts/check_neural_encoding.py --reference v0.2.2 --json benchmarks/v023-encoder-equivalence.json
python scripts/neural_lens.py smoke --steps 20 --device cuda --json benchmarks/v023-neural-smoke-cuda.json
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
node --test tests/test_dashboard.cjs
```

复测基线时，可将 `git archive v0.2.2 jitrl` 解压到独立目录，并用同一基准脚本的
`--source-root <目录> --source-label v0.2.2` 运行。编码对照脚本会执行指定 Git 标签的
模型代码，只使用可信的本地版本。
