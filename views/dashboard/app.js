// JitRL dashboard view. Runs inside AstrBot's restricted plugin iframe and
// talks to the plugin backend only through the AstrBotPluginView bridge.
//
// The host frame is sandboxed with `allow-scripts allow-forms allow-downloads`
// and *without* `allow-modals`, so window.alert/confirm/prompt are silently
// suppressed there (confirm() returns false). Destructive actions therefore use
// an in-page two-step confirmation instead of a modal.

const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
const hint = document.getElementById("hint");
const actionHint = document.getElementById("action-hint");
const forgetButton = document.getElementById("btn-forget");
const robotSelect = document.getElementById("robot-select");
const robotHint = document.getElementById("robot-hint");

const FORGET_ARM_MS = 6000;
let forgetArmed = false;
let forgetTimer = null;
let checkpointArmed = null;
let mutationBusy = false;
let refreshVersion = 0;
let selectedRobotId = "";

function setMutationBusy(busy) {
  mutationBusy = busy;
  for (const button of document.querySelectorAll("[data-memory-mutation]")) {
    button.disabled = busy;
  }
  document.getElementById("checkpoint-name").disabled = busy;
  document.getElementById("btn-export").disabled = busy || !selectedRobotId;
}

function disarmCheckpoint() {
  if (!checkpointArmed) return;
  clearTimeout(checkpointArmed.timer);
  checkpointArmed.button.textContent = checkpointArmed.label;
  checkpointArmed.button.classList.remove("btn-armed");
  checkpointArmed = null;
}

function setHint(message, isError) {
  hint.textContent = message || "";
  hint.classList.toggle("err", Boolean(isError));
}

function sayAction(message, isError) {
  actionHint.textContent = message || "";
  actionHint.classList.toggle("err", Boolean(isError));
}

function num(value, digits) {
  const d = typeof digits === "number" ? digits : 2;
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(d) : "--";
}

function signClass(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  if (value === 0) return "";
  return value > 0 ? "pos" : "neg";
}

async function apiGet(endpoint, params, fallback) {
  try {
    return await bridge.apiGet(endpoint, { ...params, robot: selectedRobotId });
  } catch (error) {
    setHint(String((error && error.message) || error), true);
    return fallback;
  }
}

function renderRobotSelector(robots, selected) {
  robotSelect.textContent = "";
  for (const robot of robots || []) {
    const option = document.createElement("option");
    option.value = robot.id;
    option.textContent = robot.label || (robot.platform_name + " · " + robot.self_id);
    robotSelect.append(option);
  }
  selectedRobotId = selected || "";
  robotSelect.value = selectedRobotId;
  robotSelect.disabled = !selectedRobotId;
  for (const button of document.querySelectorAll("[data-memory-mutation]")) {
    button.disabled = !selectedRobotId || mutationBusy;
  }
  document.getElementById("checkpoint-name").disabled = !selectedRobotId || mutationBusy;
  document.getElementById("btn-export").disabled = !selectedRobotId || mutationBusy;
  robotHint.textContent = selectedRobotId
    ? "当前面板只查看和管理该机器人实例的独立记忆。"
    : "还没有机器人实例完成学习；开始一轮对话后即可在此选择。";
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

function renderCheckpoints(payload) {
  disarmCheckpoint();
  const host = document.getElementById("checkpoints");
  host.textContent = "";
  const policy = document.getElementById("checkpoint-policy");
  if (!payload) {
    policy.textContent = "检查点列表加载失败，请刷新重试。";
    return;
  }
  policy.textContent =
    (payload.interval > 0
      ? "每 " + payload.interval + " 个 Episode 自动保存（重载后重新计数）"
      : "定期自动保存已关闭") +
    "；自动快照和操作前备份各保留 " + payload.keep + " 份。手动检查点保留到主动删除。";
  const rows = payload.checkpoints || [];
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "还没有检查点。在确认记忆可信时创建一个，即可随时回退。";
    host.append(empty);
    return;
  }
  const kinds = {
    manual: "手动 · 持久保留", auto: "定期自动", startup: "初始基线",
    before_forget: "删除前备份", before_restore: "回退前备份",
  };
  for (const row of rows) {
    const item = document.createElement("li");
    item.className = "checkpoint";
    const details = document.createElement("div");
    details.className = "checkpoint-details";
    const title = document.createElement("div");
    title.textContent = row.name || kinds[row.kind] || row.kind;
    const meta = document.createElement("div");
    meta.className = "checkpoint-meta";
    meta.textContent = new Date(row.created_at * 1000).toLocaleString() +
      " · " + (kinds[row.kind] || row.kind) +
      " · " + row.experiences + " 条经验 / " + row.episodes + " 个 Episode" +
      " · " + num(row.size_bytes / 1024, 1) + " KB";
    const id = document.createElement("code");
    id.className = "checkpoint-id";
    id.textContent = row.id;
    details.append(title, meta, id);
    const controls = document.createElement("div");
    controls.className = "checkpoint-controls";
    for (const action of ["rollback", "checkpoint_delete"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-danger";
      button.textContent = action === "rollback" ? "回退机器人记忆" : "删除快照";
      button.setAttribute("data-memory-mutation", "");
      button.disabled = mutationBusy;
      button.addEventListener("click", function () { checkpointAction(row, action, button); });
      controls.append(button);
    }
    item.append(details, controls);
    host.append(item);
  }
}

async function createCheckpoint(event) {
  event.preventDefault();
  if (mutationBusy) return;
  disarmCheckpoint();
  disarmForget();
  setMutationBusy(true);
  const input = document.getElementById("checkpoint-name");
  try {
    const result = await bridge.apiPost("checkpoint", { name: input.value.trim(), robot: selectedRobotId });
    const row = result.checkpoint;
    sayAction("已保存检查点 " + row.id + "（" + row.experiences + " 条经验）。");
    input.value = "";
    await refresh();
  } catch (error) {
    sayAction("创建失败：" + String((error && error.message) || error), true);
  } finally {
    setMutationBusy(false);
  }
}

async function checkpointAction(row, action, button) {
  if (mutationBusy) return;
  if (!checkpointArmed || checkpointArmed.id !== row.id || checkpointArmed.action !== action) {
    disarmCheckpoint();
    disarmForget();
    const label = button.textContent;
    button.textContent = action === "rollback" ? "再次点击确认回退" : "再次点击确认删除";
    button.classList.add("btn-armed");
    checkpointArmed = { id: row.id, action, button, label, timer: setTimeout(disarmCheckpoint, FORGET_ARM_MS) };
    sayAction(action === "rollback"
      ? "将用此检查点替换当前机器人的全部会话记忆，当前数据会自动备份。请在 6 秒内再次点击确认。"
      : "删除此快照不可撤销。请在 6 秒内再次点击确认。");
    return;
  }
  disarmCheckpoint();
  setMutationBusy(true);
  try {
    const result = await bridge.apiPost(action, { id: row.id, scope: "all", confirm: true, robot: selectedRobotId });
    sayAction(action === "rollback"
      ? "已回退 " + result.experiences + " 条经验。撤销本次回退可恢复备份 " + result.safety_checkpoint.id + "。"
      : "已删除指定快照。");
    await refresh();
  } catch (error) {
    sayAction("操作失败：" + String((error && error.message) || error), true);
  } finally {
    setMutationBusy(false);
  }
}

async function refresh() {
  const version = ++refreshVersion;
  setHint("正在加载…");
  const robotPayload = await apiGet("robots", {}, null);
  if (version !== refreshVersion) return;
  if (!robotPayload) {
    setHint("无法读取机器人实例列表，请确认插件已加载且后端可用。", true);
    return;
  }
  const robots = robotPayload.robots || [];
  if (!robots.length) {
    renderRobotSelector([], "");
    renderOverview({ enabled: false, apply_mode: "off", total: 0, stats: {}, hyper: {} });
    renderCurve({ buckets: [] });
    renderActions({ actions: [] });
    renderEpisodes({ episodes: [] });
    renderCheckpoints({ checkpoints: [], interval: 0, keep: 0 });
    setHint("数据库：等待机器人开始学习");
    return;
  }
  if (!robots.some((robot) => robot.id === selectedRobotId)) {
    selectedRobotId = robots[0].id;
  }
  renderRobotSelector(robots, selectedRobotId);
  const overview = await apiGet("overview", {}, null);
  if (version !== refreshVersion) return;
  if (!overview) {
    setHint("无法读取 JitRL 状态，请确认插件已加载且后端可用。", true);
    return;
  }
  renderRobotSelector(overview.robots, overview.robot_id);
  renderOverview(overview);
  const [curve, actions, episodes, checkpoints] = await Promise.all([
    apiGet("curve", { buckets: 14 }, null),
    apiGet("actions", { limit: 20 }, null),
    apiGet("episodes", { limit: 8 }, null),
    apiGet("checkpoints", {}, null),
  ]);
  if (version !== refreshVersion) return;
  renderCurve(curve);
  renderActions(actions);
  renderEpisodes(episodes);
  renderCheckpoints(checkpoints);
  setHint("数据库：" + (overview.db_path || "--"));
}

async function downloadExport() {
  const button = document.getElementById("btn-export");
  button.disabled = true;
  try {
    await bridge.download("export", { robot: selectedRobotId }, "jitrl-export.json");
    sayAction("已开始下载导出文件。");
  } catch (error) {
    sayAction("导出失败：" + String((error && error.message) || error), true);
  } finally {
    button.disabled = false;
  }
}

function disarmForget() {
  forgetArmed = false;
  if (forgetTimer) clearTimeout(forgetTimer);
  forgetTimer = null;
  forgetButton.textContent = "清空记忆";
  forgetButton.classList.remove("btn-armed");
}

async function forgetAll() {
  if (mutationBusy) return;
  if (!forgetArmed) {
    disarmCheckpoint();
    // Two-step confirmation: the iframe sandbox blocks modal dialogs, so the
    // destructive action is guarded in-page and auto-disarms after a timeout.
    forgetArmed = true;
    forgetButton.textContent = "再次点击确认清空";
    forgetButton.classList.add("btn-armed");
    sayAction("将清空当前机器人的全部记忆，清空前会自动保存检查点。请再点一次确认（6 秒后自动取消）。");
    if (forgetTimer) clearTimeout(forgetTimer);
    forgetTimer = setTimeout(disarmForget, FORGET_ARM_MS);
    return;
  }
  disarmForget();
  setMutationBusy(true);
  try {
    const result = await bridge.apiPost("forget", { scope: "all", confirm: true, robot: selectedRobotId });
    const removed = result && typeof result.removed === "number" ? result.removed : 0;
    sayAction("已删除 " + removed + " 条经验，可从删除前备份回退。");
    await refresh();
  } catch (error) {
    sayAction("清空失败：" + String((error && error.message) || error), true);
  } finally {
    setMutationBusy(false);
  }
}

async function main() {
  if (!bridge) {
    setHint("未检测到 AstrBot 插件视图桥接，无法加载数据。", true);
    return;
  }
  await bridge.ready();
  document.getElementById("btn-refresh").addEventListener("click", refresh);
  robotSelect.addEventListener("change", () => {
    selectedRobotId = robotSelect.value;
    refresh();
  });
  document.getElementById("btn-export").addEventListener("click", downloadExport);
  forgetButton.addEventListener("click", forgetAll);
  document.getElementById("checkpoint-form").addEventListener("submit", createCheckpoint);
  await refresh();
}

main();
