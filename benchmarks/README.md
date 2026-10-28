# JitRL 检索与估计基准（before / after）

## V0.1.4 / V0.2.2 主线优化

本次新增精确性与密集索引对照，完整方法和边界见
[主线研究记录](../docs/mainline-optimization.md)。`v014-before.json` / `v014-after.json`
是同一常规脚本的 40 次查询对照；`v014-retrieval-before.json` /
`v014-retrieval-after.json` 是 `scripts/bench_retrieval_quality.py` 的 25 次纯索引对照。
后者同时校验 ID、顺序和相似度，不能只凭 ID recall 判定检索正确。

- N=20000 的宽松检索 p50：13.004 → 5.138 ms（2.53×）。常规 N 是 Episode 数，实际为 37777 条经验。
- 密集 20000 行的纯索引 p50：262.558 → 10.451 ms（25.12×）。
- 小预算长状态干扰样本的 recall@10：0 → 1；新会话 session/hybrid 的错误结果同步修正。
- 数据为同机合成负载，不代表模型任务成功率或因果收益；下文保留 0.1.1 → 0.1.2 历史数据。

`scripts/bench_jitrl.py` 复现插件在每次 LLM 请求上真正执行的那条路径：状态抽象 → top-k
检索（式 24）→ 动作过滤 → 候选集 → 价值/优势估计（式 4-7、25）→ 动作选择 → 证据块渲染。
它只依赖标准库与仓库自身模块，因此在 0.1.1（基线）与 0.1.2（索引 + 两段式估计）两棵树上
可以跑**同一份脚本**，数字才具有可比性。

## 方法

```bash
# 当前树（0.1.2），约 1 分钟
python scripts/bench_jitrl.py --no-checkpoints --json benchmarks/after.json

# 基线树（0.1.1）：从 HEAD 导出一份干净副本后运行同一脚本
git archive HEAD | tar -x -C .tmp-baseline/src
python .tmp-baseline/src/scripts/bench_jitrl.py --no-checkpoints \
    --n 200,1000,5000 --json benchmarks/baseline.json
```

- 负载：确定性随机种子；Zipf 偏斜词表（少量高频词 + 大量长尾词）、中英混合状态、
  30–120 个 token、多个会话/平台、180 天时间戳、约 1.45 步/Episode。
- 场景参数（k=10、r=0.8、hybrid、30 天半衰期、prefilter_limit=4000）是脚本里的**冻结常量**，
  不随插件默认值漂移；每种场景预热后取 30 次查询的 p50/p95。
- 计时包含 `asyncio.to_thread` 与 SQLite 往返——也就是用户真实等待的时间。
- 机器：Windows / Python 3.13 / 本机 SSD；单进程独占运行，无并发负载。
- 0.1.1 的基线只跑 N ≤ 5000：它在 N=20000 时写入单条 Episode 需要数十毫秒
  （每次写入都做一次全表 DELETE + `COUNT(*)`），整个场景超过 20 分钟仍无法完成。
  这不是为了"跑赢"而省略——这个写入成本本身就是 0.1.2 修掉的问题之一。

## 结果（p50 延迟，30 次查询）

| 场景 | 0.1.1 | 0.1.2 | 加速 |
| --- | ---: | ---: | ---: |
| topk k=10, r=0.8, N=200 | 10.83 ms | 0.28 ms | ~38× |
| topk k=10, r=0.8, N=1000 | 57.64 ms | 0.31 ms | ~184× |
| topk k=10, r=0.8, N=5000 | 124.04 ms | 0.39 ms | ~319× |
| topk k=10, r=0.8, N=20000 | —（不可行） | 0.69 ms | >300× |
| topk k=10, r=0.5, N=5000 | 118.94 ms | 4.54 ms | ~26× |
| topk k=10, r=0.5, N=20000 | —（不可行） | 13.99 ms | — |
| 整个请求前段（含抽象/估计/渲染），N=5000 | 118.74 ms | 0.47 ms | ~253× |
| 整个请求前段，N=20000 | —（不可行） | 0.79 ms | — |
| 写入一个 Episode，N=5000 库 | 22.43 ms | 0.66 ms | ~34× |
| 估计 + 选动作（10 邻居） | 25.3 µs | 30.6 µs | +5 µs，可忽略 |

解读：

- 0.1.1 的检索耗时与**库大小线性相关**（每条记忆都要 JSON 解码、重建 `frozenset`、算一次
  Jaccard），N=5000 时已超过 120 ms——这还是在 4000 行预筛上限之内。
- 0.1.2 的耗时基本与库大小无关：倒排索引把候选集缩到"能过阈值的那几百行"，并且常驻内存
  的 token id 数组省掉了逐请求的 JSON 解码与字符串集合重建。
- 阈值降到 r=0.5 时候选集必然变大（前缀过滤只能剪掉"不可能达到 0.5"的行），
  N=20000 时约 14 ms p50；对应基线在同样 4000 行预筛下约 120 ms，仍然快一个数量级。
- 写入路径的 35× 来自两处：`experiences(episode_id)` 索引（此前每次写 Episode 的两条
  DELETE 都是全表扫描）与行数缓存（此前每次写入都 `COUNT(*)` 全表）。
- 估计/选动作的微基准从 25.3 µs 变成 30.6 µs（+5 µs）：这部分本来就不是瓶颈，0.1.2 改的是
  **语义正确性**（V/Q 同权、低样本收缩、探索项取期望），多出来的 5 µs 是收缩与期望分支的
  算术成本，相对 0.4 ms 的检索完全不值一提。

## 数字怎么读

- 这是单机合成负载，绝对值随 CPU/磁盘/负载变化；可信的是**同一台机器上的相对比例**与
  "是否随 N 增长"这一形状。
- r=0.8 是插件默认值，也是论文 WebArena 的取值；r=0.5 属于放宽检索，成本更高是预期行为。
- 复现前请保证没有其它重进程在跑；`benchmarks/baseline.json` 与 `after.json` 都保留了
  完整的分场景数据（含 p95、命中率、每次查询的邻居数）。
# Neural Lens / MiTA 实验（V0.2.0）

本目录同时保存原有 V0.1 检索基准和新增神经研究记录。新增结果均为本地合成工作负载，不能当作真实 Agent 成功率。

| 记录 | 内容与复现 |
|---|---|
| `neural-smoke-grouped-cuda.json` | 当前 384d、四层 MiTA grouped 执行的 CUDA 机械 smoke；无真实 Agent 效果声明。旧 `neural-smoke*.json` 保留为历史记录 |
| `mita-optimized-cpu.json` | 当前 grouped/reference/GQA 同条件 CPU fp32；`python scripts/bench_mita.py --device cpu --dtype float32 --lengths 161,1024 --mita-execution grouped --compare-reference --warmup 3 --repeats 7 --model --json benchmarks/mita-optimized-cpu.json` |
| `mita-optimized-cuda.json` | RTX 3050 Ti CUDA fp32，同一输入、权重、投影和同步协议；将上命令改为 `--device cuda --lengths 161,1024,8192` |
| `mita-optimized-cuda-fp16.json` | RTX 3050 Ti CUDA fp16；在 CUDA 命令上改为 `--dtype float16` |
| `mita-profile-optimized.json` | 当前 production kernel 的 Kineto/CUDA 归因；profiler 开销不用于正常延迟结论。旧 `mita-profile.json` 是 reference 历史证据 |
| `mita-scaling-cuda-fp16.json` | 优化前 eager/reference 的历史长度扫描，保留用于解释为什么需要 grouped；不与当前结果混作同一分母 |
| `mita-memory-ablation.json` | 两个种子、三种等参数主干、独立规则绑定留出集，保留逐样本预测；`python scripts/ablate_mita.py --json benchmarks/mita-memory-ablation.json` |

当前结果：默认 grouped MiTA 已替换四个 token-attention block，并通过输出、梯度、padding、静态/动态 packing 和 FP16 重复性回归。短序列仍可能慢于 fused GQA：同一 RTX 3050 Ti 上，N=161 fp16 forward 为 GQA 0.702 ms、grouped 2.633 ms；N=8192 则为 15.448 ms、6.093 ms，forward+backward 为 63.630 ms、12.593 ms。N=8192 的 grouped CUDA allocator 峰值仍高于 GQA，不能把速度交叉点写成普遍显存优势。历史基准按其源码哈希复现，完整算量、profiler 和边界见 [MiTA 性能分析](../docs/mita-performance-analysis.md)。
