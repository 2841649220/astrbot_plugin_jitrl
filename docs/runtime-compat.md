# AstrBot runtime compatibility audit

Every host API this plugin touches, checked against the AstrBot source for the
version range declared in `metadata.yaml` (`astrbot_version: ">=4.29,<5"`).

**Verdict: 34/34 APIs OK, no MISMATCH, no UNVERIFIED item, and the declared
version range is accurate.** Seven non-blocking follow-ups are listed in §5.

## 0. What was audited

| Ref | Upstream tree | Commit | Version |
| --- | --- | --- | --- |
| **v4.29.0** | `refs/tags/v4.29.0` | `db4ce0c4b0ae94f443c641f0f78cc9228ddabbeb` | `4.29.0` — lower bound of our range |
| **v4.29.3** | `refs/tags/v4.29.3` | `f61a7f42696da6df25dc7a249ec9b8d54ed9da18` | `4.29.3` |
| **master** | `refs/heads/master` | `cd405f2a77f3e3eae5a8c747596daaa7655c85b2` | `4.29.3` in `pyproject.toml` |

Repository: <https://github.com/AstrBotDevs/AstrBot>. All line links below point
at the **v4.29.0** tree (the oldest host a user can run us on). Files marked
*(identical on master)* were byte-compared against `cd405f2` with
`git diff --no-index`; the few that differ only gained new members, listed in
§3.

Method: the two release trees were downloaded from
`codeload.github.com/AstrBotDevs/AstrBot/tar.gz/...` and read directly; every
`from astrbot…` import was then re-checked mechanically with an AST pass over
the upstream tree, and every host class member the plugin touches was verified
to exist on the upstream class body (`Context`, `AstrMessageEvent`,
`ProviderRequest`, `ToolSet`, `LLMResponse`, `ContextWrapper`, `Star`,
`ProviderManager`, `Provider`). There is **no AstrBot installation on this
machine** (`import astrbot` → `ModuleNotFoundError`), so runtime semantics were
verified by reading the host dispatch code, not by executing it — see §7.

**About the plugin line numbers.** `main.py` was rewritten in parallel with this
audit (it grew from 1483 to 1728 lines during the review). Every citation below
names the *symbol* as well as the line number, and the numbers are from the tree
as read on 2026-10-27 00:52 (+08:00). If `main.py` keeps moving, treat the symbol
names as authoritative and re-derive the numbers; the upstream half of each row
is pinned to an immutable commit and will not drift.

## 1. API-by-API verdicts

Verdict key: **OK** = plugin usage matches the audited host contract;
**MISMATCH** = would break or behave differently on the host; **UNVERIFIED** =
cannot be settled from the source alone.

| # | Host API (exact upstream signature, v4.29.0) | Plugin usage | Verdict | Fix |
| --- | --- | --- | --- | --- |
| 1 | `astrbot.api.star` re-exports `Context`, `Star`, `StarTools`, `register` — [api/star/__init__.py:1](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/star/__init__.py#L1) *(identical on master)* | `main.py:46` `from astrbot.api.star import Context, Star` | **OK** | — |
| 2 | `Star.__init__(self, context: Context, config: dict \| None = None)` — [core/star/base.py:27](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/base.py#L27) | `main.py:163` `def __init__(self, context, config=None)`; `main.py:170` `super().__init__(context)` | **OK** | — |
| 3 | Loader instantiates with keyword args `config=plugin_config`, falling back to `context=` only on `TypeError`; `plugin_config` is an `AstrBotConfig(dict)` — [core/star/star_manager.py:1230](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L1230) | `main.py:163-171` (param names literally `context`/`config`; `JitRLConfig.from_raw` accepts `dict`) | **OK** | — (empty config → falsy `AstrBotConfig` → loader passes `context=` only; `config=None` is already handled) |
| 4 | `async def terminate(self) -> None` on `Star`; host calls `await star_cls.terminate()` on unload — [core/star/base.py:144](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/base.py#L144), [star_manager.py:1966](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L1966) | `main.py:229` `async def terminate` | **OK** | — |
| 5 | `register_on_llm_request(**kwargs)`; handler called as `handler.handler(event, req)` — [star_handler.py:433](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/register/star_handler.py#L433) *(identical on master)*; dispatch at [internal.py:307](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py#L307) | `main.py:692-697` `async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None` | **OK** | — |
| 6 | `register_on_agent_done(**kwargs)`; three hook args `event, run_context, response` — [star_handler.py:507](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/register/star_handler.py#L507); dispatch at [astr_agent_hooks.py:36](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/astr_agent_hooks.py#L36) | `main.py:812-818` `async def on_agent_done(self, event, run_context: Any, response: Any)` | **OK** | — |
| 7 | `ContextWrapper.messages: list[Message]` — [core/agent/run_context.py:17](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/run_context.py#L17) | `main.py:843` `getattr(run_context, "messages", None) or []` | **OK** | — (defensive `getattr`; field exists) |
| 8 | `register_llm_tool(name=None, **kwargs)`; JSON schema is built from the docstring via `docstring_parser` and **raises `ValueError` at import time** for a documented arg without a type — [star_handler.py:586](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/register/star_handler.py#L586) *(identical on master)* | `main.py:975-987` `@filter.llm_tool(name="jitrl_recall")`; docstring declares `query(string): …` (method at `main.py:976`) | **OK** | — |
| 9 | `register_command_group(command_group_name, sub_command=None, alias=None)` → `RegisteringCommandable` with `.command`/`.group`/`.custom_filter` — [star_handler.py:208](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/register/star_handler.py#L208) | `main.py:1043-1046` `@filter.command_group("jitrl")` + `main.py:1048` … `main.py:1421` `@jitrl_group.command(…)` (12 sub-commands) | **OK** | — |
| 10 | Command params come from the signature (first two params skipped) and are coerced by type/default — [core/star/filter/command.py:66](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/filter/command.py#L66), [command.py:93](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/filter/command.py#L93) | `main.py:1081` `limit: int = 10`, `main.py:1142` `query: str = ""`, `main.py:1378` `def jitrl_rollback(… checkpoint_id: str = "" …)`, … — every sub-command param has a default | **OK** | — |
| 11 | Async-generator command handlers are supported (`async for ret in call_handler(...)`) — [star_request.py:74](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/pipeline/process_stage/method/star_request.py#L74) | every `/jitrl …` handler is `async def` + `yield event.plain_result(...)` (`main.py:1048-1446`) | **OK** | — |
| 12 | `AstrMessageEvent.plain_result(text) -> MessageEventResult` — [astr_message_event.py:404](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L404) | `main.py:1053`, `main.py:1089`, `main.py:1119`, `main.py:1150`, … (47 call sites) | **OK** | — |
| 13 | `AstrMessageEvent.is_admin() -> bool` (method, not property) — [astr_message_event.py:268](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L268) | `main.py:1039`, `main.py:1332` | **OK** | — |
| 14 | `set_extra(key, value)` / `get_extra(key=None, default=None)` on a per-event `_extras` dict — [astr_message_event.py:222](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L222) | `main.py:716`, `main.py:727`, `main.py:775-776`, `main.py:789`, `main.py:839`, `main.py:852` | **OK** | — (`clear_extra()` is never called by the host pipeline, so extras survive from `on_llm_request` to `on_agent_done`) |
| 15 | `unified_msg_origin` property (`platform:type:session_id`) — [astr_message_event.py:106](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L106) | `main.py:725`, `main.py:840`, `main.py:992`, `main.py:1055`, `main.py:1404` | **OK** | — |
| 16 | `get_message_str() -> str` — [astr_message_event.py:140](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L140) | `main.py:728`, `main.py:841`, `main.py:994` | **OK** | — |
| 17 | `get_platform_name()` — [astr_message_event.py:126](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L126) | `main.py:323` (wrapped in try/except) | **OK** | — |
| 18 | `ProviderRequest` fields `prompt`, `system_prompt`, `extra_user_content_parts: list[ContentPart]`, `func_tool: ToolSet \| None` — [core/provider/entities.py:96](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L96), [:104](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L104), [:106](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L106), [:113](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L113) *(identical on master)* | `main.py:728` `req.prompt`; `main.py:741`, `main.py:385-424` `req.func_tool`; `main.py:775` `req.extra_user_content_parts` | **OK** | — |
| 19 | `ToolSet.names() -> list[str]` and `remove_tool(name) -> None` — [core/agent/tool.py:358](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/tool.py#L358), [:110](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/tool.py#L110) | `main.py:390`, `main.py:423`, `main.py:741` | **OK** | — (the `ToolSet` on the request is a fresh container from `get_full_tool_set()`, so `remove_tool` cannot corrupt the global tool registry; see §2.5) |
| 20 | `ContentPart.mark_as_temp()` (sets private `_no_save`) and `TextPart(text: str)` — [core/agent/message.py:68](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/message.py#L68), [:80](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/message.py#L80) *(identical on master)* | `main.py:120` lazy `from astrbot.core.agent.message import TextPart` inside `_make_temporary_part` (`main.py:106-127`); `main.py:122` `TextPart(text=text).mark_as_temp()`; appended at `main.py:773-776` | **OK** | — (this is the import the official guide prescribes: [docs/en/dev/star/guides/listen-message-event.md:312](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/en/dev/star/guides/listen-message-event.md#L312)) |
| 21 | `get_astrbot_plugin_data_path() -> str` — **no arguments**, returns `<root>/data/plugin_data`, the plugin name is *not* appended — [core/utils/astrbot_path.py:60](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/utils/astrbot_path.py#L60) | `main.py:172` `Path(get_astrbot_plugin_data_path()) / _PLUGIN_DATA_DIR` | **OK** | — (but see R5: the literal is a hard-coded duplicate of the plugin name) |
| 22 | `await Context.get_using_provider_async(umo: str \| None = None, *, locale="zh-CN") -> Provider \| None` — [core/star/context.py:514](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/context.py#L514) | `main.py:300` `await self.context.get_using_provider_async(umo)` | **OK** | — (the sync `get_using_provider()` *is* `@deprecated`; we use the async form) |
| 23 | `async ProviderManager.get_provider_by_id(provider_id) -> Providers \| None` — [core/provider/manager.py:221](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/manager.py#L221) | `main.py:290-292` `await self.context.provider_manager.get_provider_by_id(provider_id)` | **OK** | see R3 (documented accessor is `context.get_provider_by_id`) |
| 24 | `async Provider.text_chat(prompt=None, …, system_prompt=None, …, **kwargs) -> LLMResponse`; `LLMResponse.completion_text` property — [core/provider/provider.py:100](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/provider.py#L100) *(identical on master)*, [entities.py:447](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L447) | `main.py:304-308` `provider.text_chat(prompt=…, system_prompt=…)` then `(response.completion_text or "").strip()`; `main.py:851` same on the hook `response` | **OK** | see R4 (docstring calls `completion_text` outdated; it is not `@deprecated` and derives from `result_chain`) |
| 25 | `Context.register_web_api(self, route: str, view_handler, methods: list[str], desc: str) -> None` — [core/star/context.py:741](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/context.py#L741) | `main.py:1486-1492` `register(route, handler, methods, desc)` positionally, guarded by `getattr(self.context, "register_web_api", None)` | **OK** | — (param order and names match; the guard is dead code on ≥4.29 but harmless) |
| 26 | Plugin routes are served from `/api/v1/plugins/extensions/{plugin_path}` and matched by regex against the **registered route** — [dashboard/api/plugins.py:398](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/api/plugins.py#L398), [:209](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/api/plugins.py#L209) *(identical on master)* | routes `/astrbot_plugin_jitrl/<name>` at `main.py:1454-1485`; handlers take no arguments (no `<param>` segments) | **OK** | — |
| 27 | `astrbot.api.web` helpers `json_response(data=None, *, status_code=200, …)`, `error_response(message, *, status_code=400, …)`, `file_response(path, *, filename=None, content_type=None, …)`, `request.query.get(k, default, type=…)`, `await request.json(default=None)` — [api/web.py:343](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/web.py#L343), [:366](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/web.py#L366), [:391](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/web.py#L391), [:304](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/web.py#L304), [:310](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/web.py#L310) *(identical on master)* | `main.py:1503-1728` (10 handlers, routes at `main.py:1454-1485`; `_web_overview` at `main.py:1496`, `_web_export` at `main.py:1703`) | **OK** | — |
| 28 | Plugin pages: first-level dirs of `views/` (fallback `pages/`) containing `index.html` are discovered on the filesystem — [dashboard/services/plugin_page_service.py:474](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/services/plugin_page_service.py#L474), constant at [:29](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/services/plugin_page_service.py#L29) *(identical on master)* | `views/dashboard/index.html`, `views/dashboard/app.js`, `views/dashboard/style.css`; view name = `dashboard` | **OK** | — (no `views:` key needed in `metadata.yaml`; see §3.2) |
| 29 | Bridge SDK is auto-injected into the served HTML and exposes `window.AstrBotPluginView` (alias `window.AstrBotPluginPage`) with `ready/getContext/getLocale/getI18n/t/onContext/apiGet/apiPost/upload/download/subscribeSSE/unsubscribeSSE` — [dashboard/plugin_page_bridge.js:207](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/plugin_page_bridge.js#L207) *(identical on master)*, injection at [plugin_page_service.py:611](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/dashboard/services/plugin_page_service.py#L611) | `views/dashboard/app.js:9` `window.AstrBotPluginView \|\| window.AstrBotPluginPage`; `ready()` at `app.js:404`; `apiGet` at `app.js:58-65`; `apiPost` at `app.js:285`, `:314`, `:388`; `download` at `app.js:354` | **OK** | — |
| 30 | Bridge endpoint → URL mapping `/api/v1/plugins/extensions/{pluginName}/{endpoint}` and envelope handling (`{status:"error"}` ⇒ reject, otherwise `data?.data ?? data`) — [dashboard/src/views/PluginViewPage.vue:122](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/dashboard/src/views/PluginViewPage.vue#L122), [:382](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/dashboard/src/views/PluginViewPage.vue#L382) *(identical on master)* | `app.js:329` `apiGet("overview")`, `:337` `curve`, `:338` `actions`, `:339` `episodes`, `:340` `checkpoints`, `:285` `checkpoint`, `:314` `rollback`/`checkpoint_delete`, `:388` `forget`, `:354` `export` | **OK** | — (endpoints are plugin-relative and match the registered `/astrbot_plugin_jitrl/*` routes; `json_response` payloads arrive verbatim because they carry no `data` key) |
| 31 | `from astrbot.api import logger` — `_PluginContextLogger` routes to the calling plugin's logger — [api/__init__.py:1](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/api/__init__.py#L1) *(identical on master)* | `main.py:43`, `jitrl/memory.py:27`, `jitrl/evaluator.py:45`, `jitrl/checkpoints.py:19`, `jitrl/reranker.py:25`; no stdlib `logging` anywhere in shipped sources (enforced by `tests/test_manifest.py:170`) | **OK** | — |
| 32 | `astrbot_version` is parsed with `packaging.SpecifierSet` and must contain the running `VERSION` — [star_manager.py:674](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L674) | `metadata.yaml:6` `astrbot_version: ">=4.29,<5"` | **OK** | — (verified `4.29.0 ≤ v < 5`; `<5` is not vacuous because the dashboard/plugin-page contract is still evolving) |
| 33 | `AstrMessageEvent.get_platform_id()` returns the platform-instance ID (`self.platform_meta.id`) — [astr_message_event.py:133](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L133), [master:184](https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/platform/astr_message_event.py#L184) | `main.py:_robot_identity` keys the robot on this ID alone; `_robot_key` hashes it into `robots/<hash>` | **OK** | — (docstring: several adapters of the same type may run at once and "ID 是唯一的"; the value is the platform entry's unique `id` — every bundled adapter's `meta().id` is `config["id"]` — [platform_metadata.py:10](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/platform_metadata.py#L10)) |
| 34 | `AstrMessageEvent.get_self_id()` returns `getattr(self.message_obj, "self_id", "")` — [astr_message_event.py:199](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L199), [master:250](https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/platform/astr_message_event.py#L250) | `main.py` records it in the robot's `robot.json` as display/diagnostic metadata; it no longer contributes to the robot key | **OK** | — (the accessor exists and returns `str`, but the value is adapter-defined and is **not** a stable bot identity — `qq_official` derives it from message content; see the V0.2.1 delta note) |

**No MISMATCH and no UNVERIFIED row.**

V0.2.1 delta (2026-10-27 re-release): rows 33–34 were revised when the robot-isolation key
changed from `platform_id:self_id` (the 0.1.3 and original 0.2.1 behavior) to the AstrBot
platform instance ID alone. The pinned v4.29.0
API and current upstream `master` both expose the two accessors, but the earlier
`#L1482`/`#L1588`/`#L1696`/`#L1802` anchors matched neither tree (the file is 535
lines in v4.29.0 and 599 on `master`); the anchors above are the real ones. The same
code was confirmed in the AstrBot 4.29.3 checkout installed on this machine
(`D:\AstrBot\astrbot\__init__.py` → `4.29.3`; the "no installation on this machine"
note in §0 predates that install), but the tree does not sit at the same line for every
file (`tg_adapter.py`, for example, is at `:571` there and `:561` in v4.29.0), so the
links here stay pinned to the tag.

**`get_self_id()` is not a bot identity.** `AstrBotMessage.self_id` is only declared, never
initialised by the host ([astrbot_message.py:57](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astrbot_message.py#L57); `__init__` sets no default), so each adapter
decides what to put there and `get_self_id()` is a bare `getattr`
([astr_message_event.py:199](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L199)). `qq_official` derives the value from message
content, so one bot instance presents as several robots: the first bot mention's id, or the
literal `qq_official`, for group messages
([:876](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L876)),
the literal `unknown_selfid` for C2C messages
([:890](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L890)),
`str(message.mentions[0].id)` for guild messages — the *first* mention, which is not
necessarily the bot
([:903](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L903)),
and the `qq_official` fallback
([:937](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L937)).
`qq_official_webhook` reuses the same `_parse_from_qqofficial` parser
([qo_webhook_adapter.py:37](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial_webhook/qo_webhook_adapter.py#L37))
and inherits all of it. Hashing `platform_id:self_id` therefore opened a new
`robots/<hash>` directory for every message shape of a single bot; keying on the platform
instance ID collapses those back into one robot.

**The platform instance ID is the stable key.** It is `platform_meta.id`
([:133](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L133)), i.e. the `id` field of the platform entry in the
AstrBot config, documented as "平台的唯一标识符，用于配置中识别特定平台"
([platform_metadata.py:10](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/platform_metadata.py#L10)).
`PlatformManager` refuses to load a platform whose `id` is empty and rewrites `:`/`!`
([manager.py:125–142](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/manager.py#L125)); the ID is persisted in the config, so it survives restarts,
and the host itself keys conversation state on it
([session_status_check/stage.py:37](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/pipeline/session_status_check/stage.py#L37)).
Uniqueness is maintained by the config and the WebUI (a conflicting ID requires explicit
confirmation) rather than generated cryptographically, and `weixin_oc` even uses
`self.meta().id` as its `self_id`
([weixin_oc_adapter.py:1558](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/weixin_oc/weixin_oc_adapter.py#L1558)) — the host's own notion of
"which bot" is the platform instance, not `self_id`.

**Adapters disagree on what `self_id` means.** Some return a real bot account ID
(`aiocqhttp` QQ number,
[aiocqhttp_platform_adapter.py:146](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/aiocqhttp/aiocqhttp_platform_adapter.py#L146);
`discord` bot user id, `kook` `client.bot_id`, `mattermost`/`misskey`/`slack` self user
id, `lark` bot `open_id`), some a human-readable name (`telegram` bot username,
[tg_adapter.py:561](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/telegram/tg_adapter.py#L561)
— which changes if the account is renamed — and `wecom_ai_bot` `wecom_ai_bot_name`,
[wecomai_adapter.py:529](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/wecom_ai_bot/wecomai_adapter.py#L529)),
and `webchat` the constant `"webchat"`. No adapter contract promises stability, which is
why the plugin does not use it as a key.

**Residual risk: adapters that multiplex bot accounts.** `aiocqhttp` accepts several OneBot
reverse-WebSocket clients on one endpoint: the pinned `aiocqhttp` 1.4.4 registers every
client under its `X-Self-ID` (`aiocqhttp/__init__.py:569–570`) and routes each action to
the matching socket (`api_impl.py:131–139`), and AstrBot passes the event's `self_id` in
the action payload ([aiocqhttp_message_event.py:103–105](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/aiocqhttp/aiocqhttp_message_event.py#L103)). Likewise `satori` keeps the
whole `logins` list from READY and takes `self_id` from each event's own `login`
([satori_adapter.py:263](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/satori/satori_adapter.py#L263),
[:309](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/satori/satori_adapter.py#L309),
[:366](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/satori/satori_adapter.py#L366)).
Two bot accounts behind one platform instance therefore share one `robots/<hash>`
directory. That is the deliberate trade: the alternative splits **every** `qq_official`
instance into several directories, which is the bug this delta fixes. The supported
deployment is one platform entry per bot account, so each robot keeps its own directory;
operators who multiplex accounts on one adapter instance should split the platform entries
(AstrBot keys its own conversations by platform ID for the same reason).

**Upgrade path for legacy directories (0.1.3/0.2.1).** Because those releases keyed directories as
`sha256(platform_id:self_id)`, start-up consolidation groups every `robots/*/robot.json`
by its recorded `platform_id`: the richest directory becomes the canonical
platform-instance directory, the others are moved to
`robots/.legacy/<canonical-key>/`, and their V0.1 episodes are merged into the canonical
bank on first use. `MemoryBank.merge_json` deduplicates by episode ID, so an interrupted
merge is retried safely; archived directories and their checkpoints, exports and Neural
Lens weights are preserved, never deleted. If the canonical robot has no neural store,
the archived store with the most data and newest write is adopted so a lens trained
in a non-richest directory is not silently left behind.

## 2. Dispatch semantics verified in the host source (not just signatures)

1. **Hook argument order/count.** `call_event_hook(event, hook_type, *args)` awaits
   `handler.handler(event, *args)`, and plugin handlers are bound
   `functools.partial(raw_handler, plugin_instance)` at load time
   ([star_manager.py:1268-1279](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L1268)).
   So `on_llm_request(self, event, req)` and `on_agent_done(self, event, run_context, response)`
   are exactly right, and the hook return value is ignored unless the event is stopped.
2. **LLM-tool binding.** `register_llm_tool` stores the *unbound* function
   (`llm_tools.add_func(..., md.handler)`), and the same `partial(…, star_cls)`
   pass rebinds it ([star_manager.py:1282-1306](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L1282)).
   `jitrl_recall(self, event, query)` therefore receives the instance, the event, and
   the JSON-decoded `query` keyword.
3. **`mark_as_temp` really is transient.** The part reaches the provider
   (`extra_user_content_parts` is serialised into the user message —
   [entities.py:213-248](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/entities.py#L213),
   OpenAI path [openai_source.py:1409-1424](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/sources/openai_source.py#L1409)),
   and is dropped from persisted history both per part
   ([message.py:350-355](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/message.py#L350))
   and per message ([internal.py:521](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py#L521)).
   `prepare_request_images` skips non-image parts, so it cannot strip the block.
4. **`req.func_tool` is a per-request object.** `get_full_tool_set()` builds a new
   `ToolSet` on every call
   ([func_tool_manager.py:496-513](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/provider/func_tool_manager.py#L496)),
   so `_apply_tool_gate`'s `remove_tool` cannot leak into other requests. The
   tool-loop runner later replaces `req.func_tool` with a light set
   ([tool_loop_agent_runner.py:312](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/runners/tool_loop_agent_runner.py#L312)),
   so the gate applies to the first LLM call of an event only — behavioural, not an API break.
5. **Page sandbox matches the comment in `app.js:4-7`:** the iframe is served with
   `sandbox="allow-scripts allow-forms allow-downloads"`
   ([PluginViewPage.vue:651](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/dashboard/src/views/PluginViewPage.vue#L651)),
   i.e. no `allow-modals`, so the in-page two-step confirmation is required and correct.

## 3. Version-range analysis

### 3.1 Is `>=4.29,<5` honest?

Yes. Everything used above exists in **v4.29.0**; the files that define the
contracts are byte-identical to **master (4.29.3)** for
`astrbot/api/__init__.py`, `astrbot/api/web.py`, `astrbot/api/star/__init__.py`,
`astrbot/api/event/filter/__init__.py`, `astrbot/core/star/base.py`,
`astrbot/core/star/register/star_handler.py`, `astrbot/core/agent/message.py`,
`astrbot/core/provider/entities.py`, `astrbot/core/provider/provider.py`,
`astrbot/core/star/star_manager.py`, `astrbot/core/provider/func_tool_manager.py`,
`astrbot/dashboard/api/plugins.py`, `astrbot/dashboard/services/plugin_page_service.py`,
`astrbot/dashboard/plugin_page_bridge.js` and `dashboard/src/views/PluginViewPage.vue`.

The four files that did change add members only and touch nothing we call:
`core/star/context.py` (+24/−1: new `locale` kwarg on `get_using_provider_async`,
new `activate_llm_tool_async`), `core/platform/astr_message_event.py` (+65/−1: new
helpers), `core/agent/tool.py` (+20: Gemini enum handling), `core/utils/astrbot_path.py`
(+17/−2: only `get_astrbot_system_tmp_path` was rewritten).

Do **not** widen the lower bound below 4.29: the `views/` plugin-page feature,
the `AstrBotPluginView` bridge, `astrbot.api.web` and `mark_as_temp()` are all
required, and `mark_as_temp` only landed in 4.24.x — the guide annotates it
`>= v4.24.0` ([listen-message-event.md:332](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/zh/dev/star/guides/listen-message-event.md#L332))
and the release note is [changelogs/v4.24.2.md:13](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/changelogs/v4.24.2.md#L13).

### 3.2 `views:` in `metadata.yaml` is not needed

`StarMetadata.views` is parsed ([star_manager.py:556-560](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/star_manager.py#L556))
but nothing in the dashboard consumes it for listing or serving; pages are
discovered from the filesystem. Leaving the key out is correct, not an omission.

## 4. Deprecated / private-looking API usage

| Usage | Assessment |
| --- | --- |
| `astrbot.core.agent.message.TextPart` (`main.py:106-127`, lazy import at `main.py:120`) | `astrbot.core.*` looks internal, but the official plugin guide prescribes exactly this import for dynamic per-request context ([listen-message-event.md:312](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/en/dev/star/guides/listen-message-event.md#L312)). **Fine.** |
| `astrbot.core.utils.astrbot_path.get_astrbot_plugin_data_path` (`main.py:47`) | Same: prescribed by the plugin-page guide ([plugin-pages.md:210](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/zh/dev/star/guides/plugin-pages.md#L210)). **Fine.** |
| `self.context.provider_manager` (`main.py:290`) | Public instance attribute ([context.py:157](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/context.py#L157)) and `provider_manager.llm_tools` is documented for tool management, but the guide names `context.get_provider_by_id()` for provider lookup. Not deprecated — see R3. |
| `Provider.text_chat` (`main.py:304`) | Not deprecated — `astrbot/core/provider/provider.py` carries no `@deprecated` marker at all in v4.29.0. The guide's higher-level entry point is `context.llm_generate` — see R4. |
| `LLMResponse.completion_text` | Not decorated `@deprecated`; its docstring notes it is "outdated, prefer `result_chain`". The property derives from `result_chain.get_plain_text()`, so it stays correct. **Fine.** |
| No stdlib `logging`, no `req.system_prompt` rewriting | Both market rules hold: `main.py` never writes `req.system_prompt`, and injected context goes to `req.extra_user_content_parts` with `mark_as_temp()`. |

## 5. Prioritised remediation list

No blocking (P0) item: nothing has to change for the plugin to load and run on
AstrBot 4.29.0 – 4.29.3.

| ID | Priority | Item | Evidence | Suggested change |
| --- | --- | --- | --- | --- |
| R1 | **P1** | `README.md:355` claims "`jitrl/*` 完全**不 import astrbot**", which is no longer true (the modules import `astrbot.api.logger`). A market reviewer reading the README will see a false statement. | `jitrl/memory.py:27`, `jitrl/evaluator.py:45`, `jitrl/checkpoints.py:19`, `jitrl/reranker.py:25` | Reword to "only depends on `astrbot.api.logger`, so the algorithm can be unit-tested with a stub logger". |
| R2 | **P1** | The host contract is not exercised in CI: `tests/test_integration.py:29-34` skips the whole module when AstrBot is absent, which is the case here (baseline: 316 passed, 1 skipped). A future AstrBot change could break the plugin silently. | `tests/test_integration.py:29` | Add a CI job (or a documented manual gate) that installs AstrBot at the pinned tag `v4.29.0` — or vendors `astrbot/api/web.py`, `core/star/register/star_handler.py`, `core/agent/message.py` at `db4ce0c` — so this suite runs at least once per release. |
| R3 | **P2** | Provider lookup goes through `context.provider_manager` instead of the documented `context.get_provider_by_id(provider_id=…)`. Works on 4.29.x, but the documented accessor also warns on a stale provider id. | `main.py:290`; guide [plugin.md:1241](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/zh/dev/star/plugin.md#L1241) | Either keep the async call (fine) or switch to `self.context.get_provider_by_id(provider_id)` and keep the existing try/except. |
| R4 | **P2** | Auxiliary evaluator calls use `provider.text_chat(...)` directly; the documented high-level path is `await self.context.llm_generate(chat_provider_id=…, prompt=…, system_prompt=…)`, which validates the provider type and raises `ProviderNotFoundError`. | `main.py:264-308` (`_llm_text`); guide [ai.md:26](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/docs/en/dev/star/guides/ai.md#L26) | Optional: route `_llm_text` through `llm_generate` when a provider id is configured and keep `text_chat` for the session-provider fallback. |
| R5 | **P2** | `_PLUGIN_DATA_DIR = "astrbot_plugin_jitrl"` duplicates the metadata name as a literal; installing the folder under a different name would split the data directory. | `main.py:60`, `main.py:172`; `metadata.yaml:1` | Derive it from `self.name` / `StarMetadata.name` with the literal as fallback. |
| R6 | **P3** | `_apply_tool_gate` only influences the first LLM call of an event because the runner swaps in a light tool set; the docstring implies it constrains the whole run. | `main.py:385-424`; [tool_loop_agent_runner.py:312](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/agent/runners/tool_loop_agent_runner.py#L312) | Document the limitation, or also filter inside `on_using_llm_tool` if hard gating is wanted. |
| R7 | **P3** | `_register_web_apis` relies on `getattr(self.context, "register_web_api", None)` for "older builds"; on the declared range the attribute always exists. Harmless, but the comment ("an older AstrBot build without plugin web APIs") is misleading given `>=4.29`. | `main.py:1486-1489` | Keep the guard (cheap) and fix the comment. |

## 6. Reproduction

```powershell
# upstream trees used for this audit
Invoke-WebRequest https://codeload.github.com/AstrBotDevs/AstrBot/tar.gz/refs/tags/v4.29.0 `
  -OutFile v4.29.0.tar.gz
Invoke-WebRequest https://codeload.github.com/AstrBotDevs/AstrBot/tar.gz/refs/heads/master `
  -OutFile master.tar.gz
git ls-remote https://github.com/AstrBotDevs/AstrBot.git `
  refs/heads/master refs/tags/v4.29.0 refs/tags/v4.29.3
```

Then, for every import and host member used by `main.py`, cross-check against
the extracted tree (`ast.parse` over `astrbot/**`): every `from astrbot…`
import resolves (the only name that is not a top-level symbol is `filter`, which
is the `astrbot.api.event.filter` sub-package), and 23 of the 24 host members the plugin touches
were found on the upstream class bodies — the 24th, `Context.provider_manager`, is
an instance attribute set in `Context.__init__` ([context.py:157](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/star/context.py#L157))
rather than a class-body member.
`filter` resolves as the sub-package `astrbot/api/event/filter/`, which is why a
naive "is it a top-level name in `api/event/__init__.py`" check reports a false
negative.

## 7. What this audit does not prove

- No live host was executed (AstrBot is not installed here and `pip install` is
  out of scope), so this is source-level verification, not an end-to-end run.
  R2 closes that gap.
- Behaviour of individual LLM providers (OpenAI-compatible vs Anthropic vs
  Gemini) beyond `extra_user_content_parts` handling was not exercised; the three
  sources were read and all serialise `TextPart` the same way.
- The dashboard page was not rendered in a browser; the bridge contract, endpoint
  mapping, JSON envelope and iframe sandbox were verified in source.
