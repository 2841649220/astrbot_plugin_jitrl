// User-visible dashboard interactions, using only Node's built-in test runner.
// The small DOM/bridge adapter also preserves the iframe's no-modal contract.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const view = path.join(__dirname, "..", "views", "dashboard");
const html = fs.readFileSync(path.join(view, "index.html"), "utf8");
const source = fs.readFileSync(path.join(view, "app.js"), "utf8");
const snapshot = {
  id: "a".repeat(32), name: "可信版本", kind: "manual", created_at: 1000,
  episodes: 1, experiences: 2, size_bytes: 8192,
};
const robot = { id: "robot-one", platform_name: "test", self_id: "bot-1", label: "test · bot-1" };

class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.events = {};
    this.attributes = {};
    this.classes = new Set();
    this.value = "";
    this.disabled = false;
    this.style = {};
    this.classList = {
      add: (c) => this.classes.add(c),
      remove: (c) => this.classes.delete(c),
      toggle: (c, enabled) => enabled ? this.classes.add(c) : this.classes.delete(c),
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text || this.children.map((c) => c.textContent).join(""); }
  set innerHTML(value) { throw new Error("Untrusted data must use textContent"); }
  append(...items) { this.children.push(...items); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, fn) { this.events[name] = fn; }
  async emit(name) {
    if (this.disabled) return;
    await this.events[name]({ preventDefault() {} });
    // Browser event listeners need not return the promise they schedule.
    await new Promise(setImmediate);
  }
}

async function dashboard(robots = [robot]) {
  const ids = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map((m) => [m[1], new Element()]));
  ids["btn-forget"].setAttribute("data-memory-mutation", "");
  ids["btn-checkpoint"].setAttribute("data-memory-mutation", "");
  const all = () => {
    const result = new Set();
    const visit = (el) => { if (result.has(el)) return; result.add(el); el.children.forEach(visit); };
    Object.values(ids).forEach(visit);
    return [...result];
  };
  const timers = new Map();
  let nextTimer = 0;
  const posts = [];
  const gets = [];
  const bridge = {
    inventory: [{ ...snapshot }],
    ready: async () => {},
    apiGet: async (endpoint, params) => {
      gets.push({ endpoint, params: { ...params } });
      return ({
      robots: { robots },
      overview: { total: 2, enabled: true, apply_mode: "hint", stats: {}, robot_id: params.robot, robots },
      curve: { buckets: [] }, actions: { actions: [] }, episodes: { episodes: [] },
      checkpoints: { checkpoints: bridge.inventory, interval: 50, keep: 20 },
      })[endpoint];
    },
    apiPost: async (endpoint, payload) => {
      posts.push({ endpoint, payload: JSON.parse(JSON.stringify(payload)) });
      if (bridge.postHandler) return bridge.postHandler(endpoint, payload);
      if (endpoint === "checkpoint") return { checkpoint: { ...snapshot, name: payload.name } };
      if (endpoint === "rollback") return { experiences: 2, safety_checkpoint: { id: "b".repeat(32) } };
      if (endpoint === "checkpoint_delete") { bridge.inventory = []; return { deleted: payload.id }; }
      if (endpoint === "forget") return { removed: 2 };
      throw new Error("Unexpected endpoint " + endpoint);
    },
  };
  const document = {
    getElementById: (id) => { assert.ok(ids[id], "Missing element " + id); return ids[id]; },
    createElement: (tag) => new Element(tag),
    querySelectorAll: (selector) => {
      assert.equal(selector, "[data-memory-mutation]");
      return all().filter((el) => "data-memory-mutation" in el.attributes);
    },
  };
  const context = vm.createContext({
    window: { AstrBotPluginView: bridge }, document,
    setTimeout: (fn) => { timers.set(++nextTimer, fn); return nextTimer; },
    clearTimeout: (id) => timers.delete(id),
  });
  vm.runInContext(source.replace(/\nmain\(\);\s*$/, "\nglobalThis.boot = main();"), context);
  await context.boot;
  return { ids, bridge, posts, gets, timers, document,
    controls: () => ids.checkpoints.children[0].children[1].children,
  };
}

test("snapshot inventory renders its ID, counts and retention policy", async () => {
  const app = await dashboard();
  assert.match(app.ids.checkpoints.textContent, /可信版本/);
  assert.match(app.ids.checkpoints.textContent, /2 条经验/);
  assert.match(app.ids.checkpoints.textContent, new RegExp(snapshot.id));
  assert.match(app.ids["checkpoint-policy"].textContent, /手动检查点保留到主动删除/);
  assert.equal(app.controls()[0].textContent, "回退全库");
});

test("form creates a named snapshot and reports the saved ID", async () => {
  const app = await dashboard();
  app.ids["checkpoint-name"].value = "  已验证  ";
  await app.ids["checkpoint-form"].emit("submit");
  assert.deepEqual(app.posts, [{ endpoint: "checkpoint", payload: { name: "已验证", robot: robot.id } }]);
  assert.equal(app.ids["checkpoint-name"].value, "");
  assert.match(app.ids["action-hint"].textContent, new RegExp(snapshot.id));
});

test("rollback requires two clicks and includes explicit global confirmation", async () => {
  const app = await dashboard();
  const [button] = app.controls();
  await button.emit("click");
  assert.equal(app.posts.length, 0);
  assert.equal(button.textContent, "再次点击确认回退");
  await button.emit("click");
  assert.deepEqual(app.posts, [{ endpoint: "rollback", payload: { id: snapshot.id, scope: "all", confirm: true, robot: robot.id } }]);
  assert.match(app.ids["action-hint"].textContent, new RegExp("b".repeat(32)));
});

test("switching action and expiring confirmation cannot authorize deletion", async () => {
  const app = await dashboard();
  const [rollback, remove] = app.controls();
  await rollback.emit("click");
  await remove.emit("click");
  assert.equal(rollback.textContent, "回退全库");
  assert.equal(app.posts.length, 0);
  for (const fn of [...app.timers.values()]) fn();
  assert.equal(remove.textContent, "删除快照");
  await remove.emit("click");
  assert.equal(app.posts.length, 0);
  await remove.emit("click");
  assert.equal(app.posts[0].endpoint, "checkpoint_delete");
  assert.equal(app.posts[0].payload.robot, robot.id);
  assert.equal(app.ids.checkpoints.children[0].textContent.includes("还没有检查点"), true);
});

test("pending operations disable every memory mutation and prevent duplicates", async () => {
  const app = await dashboard();
  let release;
  app.bridge.postHandler = () => new Promise((resolve) => { release = resolve; });
  const [button] = app.controls();
  await button.emit("click");
  const pending = button.emit("click");
  await new Promise(setImmediate);
  assert.ok(app.document.querySelectorAll("[data-memory-mutation]").every((el) => el.disabled));
  await app.ids["btn-forget"].emit("click");
  await button.emit("click");
  assert.equal(app.posts.length, 1);
  release({ experiences: 2, safety_checkpoint: { id: "backup" } });
  await pending;
  await new Promise(setImmediate);
  assert.ok(app.document.querySelectorAll("[data-memory-mutation]").every((el) => !el.disabled));
});

test("failed rollback shows the error and re-enables controls", async () => {
  const app = await dashboard();
  app.bridge.postHandler = async () => { throw new Error("安全备份失败"); };
  const [button] = app.controls();
  await button.emit("click");
  await button.emit("click");
  assert.match(app.ids["action-hint"].textContent, /安全备份失败/);
  assert.ok(app.ids["action-hint"].classes.has("err"));
  assert.equal(button.disabled, false);
});

test("clear retains two-step confirmation and explains recovery", async () => {
  const app = await dashboard();
  await app.ids["btn-forget"].emit("click");
  assert.equal(app.posts.length, 0);
  assert.match(app.ids["action-hint"].textContent, /自动保存检查点/);
  await app.ids["btn-forget"].emit("click");
  assert.equal(app.posts[0].endpoint, "forget");
  assert.equal(app.posts[0].payload.robot, robot.id);
  assert.match(app.ids["action-hint"].textContent, /删除前备份回退/);
});

test("every dashboard read is scoped to the selected robot", async () => {
  const app = await dashboard();
  assert.ok(app.gets.length >= 6);
  assert.ok(app.gets.filter(({ endpoint }) => endpoint !== "robots").every(({ params }) => params.robot === robot.id));
  assert.equal(app.ids["robot-select"].value, robot.id);
});

test("switching robots refreshes only the selected instance", async () => {
  const second = { id: "robot-two", platform_name: "test", self_id: "bot-2", label: "test · bot-2" };
  const app = await dashboard([robot, second]);
  app.ids["robot-select"].value = second.id;
  await app.ids["robot-select"].emit("change");
  assert.equal(app.ids["robot-select"].value, second.id);
  assert.ok(app.gets.filter(({ endpoint }) => endpoint !== "robots").slice(-5).every(({ params }) => params.robot === second.id));
});

test("unlabeled legacy robot falls back to the platform instance id", async () => {
  const legacy = {
    id: "robot-legacy",
    platform_name: "qq_official",
    platform_id: "爱弥斯",
    self_id: "unknown_selfid",
  };
  const app = await dashboard([legacy]);
  const options = app.ids["robot-select"].children;
  assert.equal(options.length, 1);
  assert.equal(options[0].textContent, "qq_official · 爱弥斯");
});
