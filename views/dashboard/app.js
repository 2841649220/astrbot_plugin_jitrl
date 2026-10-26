// JitRL dashboard view. Runs inside AstrBot's restricted plugin iframe and
// talks to the plugin backend only through the AstrBotPluginView bridge.

const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
const hint = document.getElementById("hint");

function say(message, isError) {
  hint.textContent = message || "";
  hint.classList.toggle("err", Boolean(isError));
}

function num(value, digits) {
  const d = typeof digits === "number" ? digits : 2;
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(d) : "--";
}

function signClass(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "";
  return value > 0 ? "pos" : "neg";
}

async function apiGet(endpoint, params, fallback) {
  try {
    return await bridge.apiGet(endpoint, params);
  } catch (error) {
    say(String((error && error.message) || error), true);
    return fallback;
  }
}

function renderOverview(data) {
  document.getElementById("kpi-total").textContent = data.total || 0;
  const stats = data.stats || {};
  document.getElementById("kpi-episodes").textContent = stats.episodes || 0;
  document.getElementById("kpi-biased").textContent = stats.biased || 0;
  document.getElementById("kpi-failures").textContent = stats.failures || 0;

  const modeLabels = { off: "仅记录", hint: "偏好提示", logit: "闭式 Logit 重排" };
  document.getElementById("chip-mode").textContent =
    "模式：" + (modeLabels[data.apply_mode] || data.apply_mode || "--");
  const state = document.getElementById("chip-state");
  state.textContent = data.enabled ? "运行中" : "已停用";
  state.classList.toggle("chip-muted", !data.enabled);

  const list = document.getElementById("hyper");
  list.textContent = "";
  const hyper = data.hyper || {};
  const rows = [
    ["近邻数 k", hyper.k],
    ["相似度阈值 r", hyper.threshold],
    ["折扣因子 gamma", hyper.gamma],
    ["温度 beta", hyper.beta],
    ["探索率 lambda", hyper.exploration_rate],
    ["乐观奖励 alpha", hyper.ucb_alpha],
    ["记忆作用域", hyper.scope],
    ["评估器", hyper.evaluator],
  ];
  for (const row of rows) {
    const dt = document.createElement("dt");
    dt.textContent = row[0];
    const dd = document.createElement("dd");
    dd.textContent = row[1] === undefined || row[1] === null ? "--" : String(row[1]);
    list.append(dt, dd);
  }

  const box = document.getElementById("last-decision");
  const last = data.last_decision;
  box.textContent = last
    ? "最近决策：" + (last.reason || "（无）")
    : "本次进程还没有产生策略决策。";
}

function renderCurve(payload) {
  const host = document.getElementById("curve");
  host.textContent = "";
  const buckets = (payload && payload.buckets) || [];
  if (!buckets.length) {
    const empty = document.createElement("div");
    empty.className = "curve-empty";
    empty.textContent = "还没有足够的经验绘制学习曲线。";
    host.append(empty);
    return;
  }
  const peak = Math.max.apply(
    null,
    buckets.map(function (b) { return Math.abs(b.mean_return || 0); }).concat([0.001])
  );
  for (const bucket of buckets) {
    const value = bucket.mean_return || 0;
    const bar = document.createElement("div");
    bar.className = "curve-bar " + signClass(value);
    const height = Math.max(3, Math.round((Math.abs(value) / peak) * 120));
    bar.style.height = height + "px";
    bar.title =
      "bucket " + bucket.bucket +
      ": mean_return=" + num(value) +
      ", mean_reward=" + num(bucket.mean_reward) +
      ", n=" + (bucket.count || 0);
    host.append(bar);
  }
}

function renderActions(payload) {
  const host = document.getElementById("actions");
  host.textContent = "";
  const rows = (payload && payload.actions) || [];
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "还没有动作统计。先用几轮对话积累经验。";
    host.append(empty);
    return;
  }
  const peak = Math.max.apply(
    null,
    rows.map(function (r) { return Math.abs(r.mean_return || 0); }).concat([0.001])
  );
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "action-row";

    const name = document.createElement("div");
    name.className = "action-name";
    name.textContent = row.action_text || row.action_key || "?";
    name.title = row.action_key || "";

    const bar = document.createElement("div");
    bar.className = "action-bar";
    const fill = document.createElement("i");
    fill.className = signClass(row.mean_return);
    const ratio = Math.abs(row.mean_return || 0) / peak;
    fill.style.width = Math.max(4, Math.round(ratio * 100)) + "%";
    bar.append(fill);

    const mean = document.createElement("div");
    mean.className = "action-num";
    mean.textContent = num(row.mean_return);

    const positive = document.createElement("div");
    positive.className = "action-num";
    positive.textContent =
      (typeof row.positive_rate === "number"
        ? Math.round(row.positive_rate * 100) + "%"
        : "--") + " · n=" + (row.count || 0);

    line.append(name, bar, mean, positive);
    host.append(line);
  }
}

function renderEpisodes(payload) {
  const host = document.getElementById("episodes");
  host.textContent = "";
  const rows = (payload && payload.episodes) || [];
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "还没有 episode。";
    host.append(empty);
    return;
  }
  for (const row of rows) {
    const item = document.createElement("li");
    item.className = "episode";
    const top = document.createElement("div");
    top.className = "episode-top";
    const when = document.createElement("span");
    when.textContent = new Date((row.created_at || 0) * 1000).toLocaleString();
    const score = document.createElement("span");
    score.textContent =
      "steps=" + (row.step_count || 0) + " · G=" + num(row.mean_return);
    top.append(when, score);
    const text = document.createElement("div");
    text.className = "episode-text";
    text.textContent = row.user_text || "（无用户文本）";
    item.append(top, text);
    host.append(item);
  }
}

async function refresh() {
  say("正在加载…");
  const overview = await apiGet("overview", {}, null);
  if (!overview) {
    say("无法读取 JitRL 状态，请确认插件已加载且后端可用。", true);
    return;
  }
  renderOverview(overview);
  const curve = await apiGet("curve", { buckets: 14 }, null);
  const actions = await apiGet("actions", { limit: 20 }, null);
  const episodes = await apiGet("episodes", { limit: 8 }, null);
  renderCurve(curve);
  renderActions(actions);
  renderEpisodes(episodes);
  say("数据库：" + (overview.db_path || "--"));
}

async function downloadExport() {
  const button = document.getElementById("btn-export");
  button.disabled = true;
  try {
    await bridge.download("export", "jitrl-export.json");
    say("已开始下载导出文件。");
  } catch (error) {
    say("导出失败：" + String((error && error.message) || error), true);
  } finally {
    button.disabled = false;
  }
}

async function forgetAll() {
  if (!window.confirm("确定清空全部 JitRL 记忆？此操作不可撤销。")) return;
  const button = document.getElementById("btn-forget");
  button.disabled = true;
  try {
    const result = await bridge.apiPost("forget", { scope: "all" });
    say("已删除 " + (result && result.removed ? result.removed : 0) + " 条经验。");
    await refresh();
  } catch (error) {
    say("清空失败：" + String((error && error.message) || error), true);
  } finally {
    button.disabled = false;
  }
}

async function main() {
  if (!bridge) {
    say("未检测到 AstrBot 插件视图桥接，无法加载数据。", true);
    return;
  }
  await bridge.ready();
  document.getElementById("btn-refresh").addEventListener("click", refresh);
  document.getElementById("btn-export").addEventListener("click", downloadExport);
  document.getElementById("btn-forget").addEventListener("click", forgetAll);
  await refresh();
}

main();