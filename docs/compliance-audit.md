# JitRL 插件 · AstrBot 官方插件市场合规审计

## V0.2.3 神经流水线复核（2026-10-28）

- 四层 MiTA 的注意力模块与 grouped 执行内核相对 V0.2.2 没有变化；相同历史输入的
  编码逐字节等价，旧权重可严格加载。CPU/CUDA 的正式尺寸训练与重载评分闭环通过。
- 冻结特征缓存限制在单个神经库、带作用域键，不缓存奖励或历史快照；reset/close
  清理缓存，epoch 阻止迟到计算回填。显式历史 cutoff 仍严格，实时回放单独处理同时间戳。
- Python 3.13 完整测试 1036 项通过；本机 AstrBot / Python 3.12 下的集成与插件适配
  测试 74 项通过，面板测试 10 项通过。没有增加运行依赖、远端调用或逐请求 system prompt 改写。
- 性能来自本机合成负载，业务收益未经实测；候选支持度指不同 episode ID，不能当作
  统计独立性保证。完整方法、数据、数值等价性和适用边界见
  [V0.2.3 验证记录](neural-pipeline-performance.md)。

## V0.2.2 主线迁移复核（2026-10-28）

- V0.1.4 的精确检索、会话过滤、当前轮轨迹和保守工具门控已迁入本分支；研究依据与
  性能边界见[主线优化记录](mainline-optimization.md)。V0.1 主线未引入神经依赖。
- 本次未修改 `jitrl/neural/`，四层 MiTA、历史检索快照、pending 奖励排除与作用域合同
  由原有神经测试继续覆盖；正式 384d / 四层 / grouped MiTA 的 CUDA 采集、20 步训练、
  原子保存重载和候选评分闭环通过，报告在开发仓库的
  [v022-neural-smoke-cuda.json](../benchmarks/v022-neural-smoke-cuda.json)。它只使用合成数据。
- `astrbot.api.logger` 和临时 `extra_user_content_parts` / `mark_as_temp()` 注入方式保留，
  没有新增标准库日志或逐请求 system prompt 改写。版本声明与发布 ZIP 由构建脚本校验。
- Python 3.13 的完整神经分支测试通过；另用本机 AstrBot / Python 3.12 执行
  `tests/test_integration.py`，29 项通过。测试使用独立的 `ASTRBOT_ROOT` 临时目录。

以下章节保留历史审计快照及其当时的版本、行号与测量。

审计对象：`astrbot_plugin_jitrl`（仓库根 `C:/Users/ASUS/Desktop/astrbot_plugin_jitrl`，版本 0.1.1）  
审计日期：2026-10-27  
规范基线：docs.astrbot.app（2026-10-27 抓取）、AstrBot master 源码（`astrbot/__init__.py` 声明 `__version__ = "4.29.3"`）、插件市场 JSON 规范 2026-06-27。  
审计者：compliance-auditor（只读审计；本文件是本次唯一写产物，写入 `docs/`）  
审计快照：`main.py` 1483 行、`_conf_schema.json` 453 行、`views/dashboard/app.js` 412 行。审计期间 Lead 正在并行修改 `jitrl/*.py` 与 `main.py`；**引用到的行号以本快照为准**，打补丁前请重新定位。

**结论摘要**：仓库**没有发现明确的「违规」项**。`metadata.yaml` / `_conf_schema.json` / 目录布局 / 发布 ZIP / 视图桥接 / 认证入口基本全部符合官方规范，且 `tests/test_manifest.py` 已把大部分契约变成可执行断言。发现 **6 项「风险」**（其中 1 项直接影响市场可安装性：`astrbot_version` 下限）与 5 项可选增强，逐项见第 2 节。

## V0.2.1 隔离功能修正与发布包复核（2026-10-27 重发）

本节补充 0.2.1 重发的机器人隔离修正审核证据；0.2.1 初版曾把 `self_id` 纳入实例键，现改为仅使用平台实例 ID，启动时合并旧版遗留目录。上文的静态行号和仓库版本描述属于早期快照，若与此节冲突，以当前源码、`docs/runtime-compat.md` 和发布构建输出为准。

- 机器人身份只使用 AstrBot 平台实例 ID（`AstrMessageEvent.get_platform_id()` → `self.platform_meta.id`）：插件 `main.py:_robot_identity` 只取该 ID 作 `robots/<hash>` 的隔离键，`get_self_id()` **不再参与**哈希，只写入 `robots/<hash>/robot.json` 供展示与排查。依据：平台实例 ID 就是配置中平台条目的 `id` 字段（各官方适配器的 `meta().id` 均取 `config["id"]`），官方注释为「平台的唯一标识符，用于配置中识别特定平台」（[v4.29.0 platform_metadata.py:10](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/platform_metadata.py#L10)）；`PlatformManager` 对空 ID 直接跳过加载、对 `:`/`!` 自动改写（[v4.29.0 manager.py:125](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/manager.py#L125)）；宿主自身也用它标识会话（[v4.29.0 session_status_check/stage.py:37](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/pipeline/session_status_check/stage.py#L37)）；事件 API 见 [v4.29.0:133](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L133) 与 [master:184](https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/platform/astr_message_event.py#L184)。
- `get_self_id()` 不能作为机器人身份：它只是 `getattr(self.message_obj, "self_id", "")`（[v4.29.0:199](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astr_message_event.py#L199)，[master:250](https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/platform/astr_message_event.py#L250)），而 `AstrBotMessage.self_id` 由各适配器自行赋值（[astrbot_message.py:57](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/astrbot_message.py#L57)）。`qq_official` 按消息内容派生：群聊取首个机器人 mention id 或字面量 `qq_official`（[v4.29.0:876](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L876)）、C2C 固定 `unknown_selfid`（[:890](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L890)）、频道取 `mentions[0].id`（[:903](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L903)），兜底 `qq_official`（[:937](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial/qqofficial_platform_adapter.py#L937)）；`qq_official_webhook` 复用同一解析器（[qo_webhook_adapter.py:37](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.0/astrbot/core/platform/sources/qqofficial_webhook/qo_webhook_adapter.py#L37)）。同一机器人实例因此会分裂出多个 self_id 与多个 `robots/<hash>` 目录，本版改以平台实例 ID 隔离后合并为一个。以上 v4.29.0 行号均按标签源码核对，并在本机安装的 AstrBot 4.29.3（`D:\AstrBot`）中确认对应代码存在（该树中 `get_platform_id()`/`get_self_id()` 位于 `:184`/`:250`）。
- 已知边界（不构成合规问题）：`aiocqhttp` 反向 WebSocket 可同时接入多个 OneBot 账号（`aiocqhttp` 1.4.4 以 `X-Self-ID` 路由，`aiocqhttp/__init__.py:569-570`、`api_impl.py:131-139`；AstrBot 发送动作时按事件 `self_id` 路由，`aiocqhttp_message_event.py:103-105`、`:263-264`），`satori` 的 READY 返回 `logins` 列表、每个事件自带 `login`（`satori_adapter.py:263`、`:309`、`:366`）。这类部署下按平台实例隔离会把多个账号的数据合并；约定部署形态为「一个机器人账号 = 一个平台实例」，与宿主 `platform_id` 语义一致。
- `metadata.yaml` 使用语义化版本字段并保留规范要求的名称、作者、GitHub 仓库、版本范围、描述和可选市场标签；AstrBot 官方要求发布前将代码推送到 GitHub，插件包不得超过 16 MB，市场提交由 AstrBot Cloud 发布页单独完成。详见[官方发布说明](https://docs.astrbot.app/dev/star/plugin-publish.html)。
- 发布构建通过 `python scripts/build_release.py` 生成单根目录 ZIP 并校验版本链；完整测试与包体积结果记录在本次交付说明中。

> 说明：标「未验证」的地方表示我**没有**在官方文档或源码中找到可引用的明文规定，不替官方「补规则」。市场的人工/自动审核清单（拒稿理由、CI 自动检查项）目前**没有公开发布**，只能从发布页、市场 JSON 规范与宿主源码的强制行为反推。

---

## 1. 规范事实（每条含来源 URL 与原文引用）

### 1.1 插件结构、`metadata.yaml` 与插件名

来源：<https://docs.astrbot.app/dev/star/plugin-new.html>（"AstrBot 插件开发指南 🌠"）

> "请务必修改此文件，AstrBot 识别插件元数据依赖于 `metadata.yaml` 文件。"

> "插件名格式: 推荐以 `astrbot_plugin_` 开头；不能包含空格；保持全部字母小写；尽量简短。"

> "可以在插件目录下添加 logo.png 文件作为插件的 Logo。请保持长宽比为 1:1，推荐尺寸为 256x256。"（**Logo 是文件，不是 metadata 字段**）

> "你可以在 metadata.yaml 中新增 support_platforms 字段（list[str]），声明插件支持的平台适配器。"

> "你可以在 metadata.yaml 中新增 astrbot_version 字段……格式与 pyproject.toml 依赖版本约束一致（PEP 440），且不要加 v 前缀。" / "当当前 AstrBot 版本不满足该范围时，插件会被阻止加载并提示版本不兼容。"

来源：<https://docs.astrbot.app/dev/star/plugin-publish.html>（"发布插件到插件市场"）给出的完整元数据示例（节选，原文为 YAML）：

> "name: astrbot_plugin_example # 插件标识符，英文，唯一 / display_name: 示例插件名称 / # short_desc: 一句话介绍你的插件功能 # （可选）…… / desc: …… / version: 1.0.0 # 插件版本号（遵循语义化版本规范） / author: 作者名称 / repo: https://github.com/your-name/repo / # astrbot_version / # support_platforms / # social_link / # tags"

来源：AstrBot 源码 `astrbot/core/star/updater.py`（<https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/star/updater.py>）：

> `PLUGIN_METADATA_FILENAMES = ("metadata.yaml", "metadata.yml")`  
> `PLUGIN_METADATA_REQUIRED_FIELDS = ("name", "desc", "version", "author")`  
> `PLUGIN_METADATA_MAX_BYTES = 1024 * 1024`  
> `invalid_fields = [field for field in PLUGIN_METADATA_REQUIRED_FIELDS if not isinstance(...) or not ....strip()]` → 报错文案 "必须是非空字符串"

结论（**硬性**）：`name` / `desc` / `version` / `author` 四个字段必填且必须是**非空字符串**（YAML 里未加引号的纯数字会被解析成 int，直接判失败）；`repo`、`display_name`、`short_desc`、`support_platforms`、`astrbot_version` 为可选。`astrbot/core/star/star.py` 的 `StarMetadata` 只消费 `name/author/desc/short_desc/version/repo/display_name/logo_path/support_platforms/astrbot_version/i18n/views` —— **`tags`、`social_link` 宿主不读，它们是市场侧字段**（见 1.7）。

### 1.2 `_conf_schema.json`

来源：<https://docs.astrbot.app/dev/star/guides/plugin-config.html>

> "要注册配置，首先需要在您的插件目录下添加一个 _conf_schema.json 的 json 文件。"

> "type: 此项必填。配置的类型。支持 string, text, int, float, bool, object, list, dict, template_list。当类型为 text 时，将会可视化为一个更大的可拖拽宽高的 textarea 组件。"

> "items: 可选。如果配置的类型是 object，需要添加 items 字段。"  
> "default: 可选。配置的默认值。……int 是 0，float 是 0.0，bool 是 False，string 是空串，object 是 {}，list 是 []。"  
> options / labels / subtitles / hint / obvious_hint / invisible / secret / slider / editor_mode / editor_language / editor_theme / _special 逐项均有定义；`slider` "仅对 int 和 float 生效"；`_special` 常用值含 `select_provider`，其结果为字符串。`file` 类型自 v4.13.0 引入。

> "AstrBot 在载入插件时会检测插件目录下是否有 _conf_schema.json 文件，如果有，会自动解析配置并保存在 data/config/<plugin_name>_config.json 下……并在实例化插件类时传入给 __init__()。"  
> "您在发布不同版本更新 Schema 时，AstrBot 会递归检查 Schema 的配置项，自动为缺失的配置项添加默认值、移除不存在的配置项。"

宿主强制行为（**硬性**）：`astrbot/core/config/astrbot_config.py` 的 `_config_schema_to_default_config` 执行 `if v["type"] not in DEFAULT_VALUE_MAP: raise TypeError(...)` —— **缺少 `type` 会 KeyError，未知 `type` 会 TypeError，插件配置加载即失败**。

### 1.3 生命周期与 Hook

来源：<https://docs.astrbot.app/dev/star/plugin.html>（旧指南，页面顶部仍标注 "您仍然可以参考此页进行插件开发"）

> "插件是继承自 `Star` 基类的类实现。" / "在 `__init__` 方法中会传入 `Context` 对象" / "插件类所在的文件名需要命名为 `main.py`。"  
> "可以选择实现 `terminate` 函数，当插件被卸载/停用时会调用。"  
> "请务必使用 `from astrbot.api import logger` 来获取日志对象，而不是使用 `logging` 模块。"  
> "每个插件在 `__init__` 后会自动拥有独立的 `self.logger`。"

来源：`astrbot/core/star/register/star_handler.py`（<https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/star/register/star_handler.py>）的装饰器 docstring：

> `register_on_llm_request`："请务必接收两个参数：event, req"  
> `register_on_agent_done`："请务必接收三个参数：event, run_context, response"  
> 同文件示例中官方**允许**改写 `req.system_prompt`：`req.system_prompt += "你是一个猫娘..."`

`register_command` / `register_command_group` / `register_llm_tool` 均经 `astrbot/api/event/filter/` 包再导出（`from astrbot.api.event import filter` 有效，因为 `filter` 是子包）。

### 1.4 数据目录

来源：<https://docs.astrbot.app/dev/star/guides/storage.html>

> "为了规范插件存储大文件的行为，请将大文件存储于 `data/plugin_data/{plugin_name}/` 目录下。"  
> `plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name`

来源：`astrbot/core/utils/astrbot_path.py`：`get_astrbot_plugin_data_path() -> <root>/data/plugin_data`；`get_astrbot_data_path() -> <root>/data`。

另有开发准则（<https://docs.astrbot.app/dev/star/plugin-new.html>）：

> "持久化数据请存储于 `data` 目录下，而非插件自身目录，防止更新/重装插件时数据被覆盖。"

### 1.5 可视化视图（`views/`）、`register_web_api`、bridge、安全约束

来源：<https://docs.astrbot.app/dev/star/guides/plugin-pages.html>

> "页面文件放在插件目录的 views/ 下（兼容 pages/），由 Dashboard 以受限 iframe 的方式加载；页面里的脚本通过 window.AstrBotPluginView bridge（别名 window.AstrBotPluginPage）和 Dashboard 通信。"  
> "AstrBot 只扫描 `views/<view_name>/index.html`，没有 index.html 的目录会被忽略。"  
> "使用 `context.register_web_api(route, view_handler, methods, desc)` 注册插件 API。"  
> "路由需要包含插件名作为前缀。视图端的 bridge endpoint 不需要包含插件名"（`bridge.apiGet("items/123")` → `/api/v1/plugins/extensions/<plugin_name>/items/123`）  
> `ready()` / `apiGet(endpoint, params)` / `apiPost(endpoint, body)` / `upload(endpoint, file)` / `download(endpoint, params, filename)`（"params 会作为 query 参数发送；filename 可选"）  
> 安全约束一节：iframe sandbox 为 `allow-scripts allow-forms allow-downloads`（**没有 allow-modals**）；"后端 handler 仍然要验证输入。不要信任视图传来的路径、文件名、格式或数值范围"。

**认证事实**（源码，非文档）：<https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/dashboard/api/plugins.py>

> `@router.get("/plugins/extensions/{plugin_path:path}")` / `@router.post(...)` 均带 `auth: AuthContext = Depends(require_plugin_scope)`，其中 `require_plugin_scope = ScopeDependency("plugin")`；源码注释写明 "V1 routes are not covered by the dashboard auth middleware, so path-token ..."。

即：**插件扩展 API 由宿主强制鉴权（Dashboard 用户 JWT 或具备 plugin scope 的 API Key）**，插件自身不需要、也不应再实现一套认证。

### 1.6 i18n

来源：<https://docs.astrbot.app/dev/star/guides/plugin-i18n.html>

> "插件可以在自己的目录下提供 `.astrbot-plugin/i18n/*.json`，让 WebUI 根据当前语言显示插件名称、描述和配置项文案。"  
> "当当前语言没有对应翻译、某个字段缺失，或语言文件不存在时，AstrBot 会回退到默认文案"（回退到 `metadata.yaml` 的 `display_name/short_desc/desc` 与 `_conf_schema.json` 的文案）  
> 视图元数据 `views.<view_name>.title` / `description`："是否必填：否"。

→ i18n 是**可选**能力，缺失不会导致安装/加载失败，只会回退原文案。

### 1.7 插件市场规范（身份、字段类型、ZIP 布局与体积）

来源：<https://docs.astrbot.app/dev/plugin-market/2026-06-27.html>（"AstrBot 插件市场 JSON 规范"，规范版本 2026-06-27，schema_version 1）

> "plugin_id 定义为：`metadata.author + "/" + metadata.name`"  
> "author 必须等于 metadata.yaml.author。name 必须等于 metadata.yaml.name。" / "version 必须匹配 metadata.yaml.version。"  
> 每个 PluginRecord 必须包含：author / name / version / repo / desc  
> 可选字段：display_name、short_desc、download_url、logo、tags（string[]）、category、support_platforms（string[]）、astrbot_version、social_link、updated_at、i18n、pinned、stars、download_count  
> "author 和 name：必须是非空字符串。必须去除首尾空白。不得包含 /。不得包含 ASCII 控制字符。"  
> "`repo` 必须是以下之一：`https://github.com/{owner}/{repository}` / .git / /tree/{branch}"；"repo 不得是：HTTP URL / SSH URL / 非 GitHub URL / GitHub Enterprise URL / 文件、release、pull request、issue 或子目录 URL"  
> 第 11 节校验规则："从 repo 或 download_url 解析出的包包含 metadata.yaml。metadata.yaml.author 等于 author。metadata.yaml.name 等于 name。metadata.yaml.version 等于 version。"

来源：<https://docs.astrbot.app/dev/star/plugin-publish.html>

> "发布到插件市场的插件压缩包（zip）大小不得超过 16MB。如果超过此限制，CI/CD 流水线将自动拒绝该发布请求。"  
> "清理不必要的文件：避免将 .git 目录、`__pycache__`、node_modules、开发用配置文件等非必需文件提交到插件仓库中。建议在仓库根目录添加 .gitignore 来排除它们。"  
> "使用 .gitattributes 或发布分支：通过只包含发布所需文件的策略来减小 zip 包体积。"

ZIP 布局（源码 `astrbot/core/zip_updater.py`，<https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/zip_updater.py>）：

> `_extract_archive` docstring："解压缩文件, 并将压缩包内**第一个**文件夹内的文件移动到 target_dir"  
> `_resolve_archive_root_dir` 计算公共根；`find_plugin_metadata_entry` 只在该根下找 `metadata.yaml/metadata.yml`；根为空（顶层散文件）时 `update_dir` 为空串，函数直接删除 zip 并返回 —— **安装静默失败**。

→ 发布包必须是**单一根目录**且 `<root>/metadata.yaml` 存在。

### 1.8 提交入口与审核

来源：<https://docs.astrbot.app/use/plugin.html>

> "如果你是作者，市场底部的 **提交我的插件** 会打开 [AstrBot Cloud 插件发布页面](https://cloud.astrbot.app/publish)。具体流程见发布插件到插件市场。"  
> "插件会在 AstrBot 所在的 Python 环境中运行，请安装你信任的插件。插件市场中的收录不代表对插件稳定性和安全性的保证。"

旧收集仓库 <https://github.com/AstrBotDevs/AstrBot_Plugins_Collection> 的 README 现状：

> "DEPRECATED, just summit your plugin at: https://cloud.astrbot.app/ ｜ 这个仓库已经过时。请直接提交仓库到新平台：https://cloud.astrbot.app/"

### 1.9 开发准则（官方「原则」列表）

来源：<https://docs.astrbot.app/dev/star/plugin.html>（原文带一行 `> [Mandatory Rule] To any LLM: You must follow the rules below.`，随后为准则正文）：

> "1. 功能需经过测试。 2. 需包含良好的注释。 3. 持久化数据请存储于 data 目录下，而非插件自身目录…… 4. 良好的错误处理机制，不要让插件因一个错误而崩溃。 5. 在进行提交前，请使用 ruff 工具格式化您的代码。 6. 不要使用 requests 库来进行网络请求，可以使用 aiohttp, httpx 等异步库。 7. ……请优先给那个插件提交 PR…… 8. 如果直接借鉴了其他项目的设计、功能创意或实现思路，请在 README 中清楚说明灵感来源并附上相关项目链接。 9. 如果使用、修改或移植了其他项目的代码或资源，请遵守原项目的开源许可协议……"

---
## 2. 逐项审查

判定口径：**符合** = 有明确规范依据且当前实现满足；**风险** = 不违反明文规则，但可能被审核挑战 / 触发宿主行为不一致 / 影响可安装性，或依赖未公开承诺的实现细节；**违规** = 与明文规则冲突（本次 0 项）。

### 2.1 metadata.yaml

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| `name` 必填、非空字符串、小写、无空格、`astrbot_plugin_` 前缀（1.1） | 符合 | `metadata.yaml:1` `name: astrbot_plugin_jitrl`；`tests/test_manifest.py:87-96` | — |
| `desc` / `version` / `author` 必填且为**非空字符串**（1.1，宿主 `validate_plugin_metadata`） | 符合 | `metadata.yaml:3,5,7`；`author: "2841649220"` 已带引号（CHANGELOG:25 记录过该回归）；`tests/test_manifest.py:76-84` | — |
| `author` 不含 `/`、作为命名空间（1.7） | 符合 | `metadata.yaml:7`；`tests/test_manifest.py:96` | — |
| `repo` 为 HTTPS GitHub 仓库 URL（1.7） | 符合 | `metadata.yaml:8` `https://github.com/2841649220/astrbot_plugin_jitrl`；`tests/test_manifest.py:115-119` | — |
| `astrbot_version` 为 PEP 440、无 `v` 前缀（1.1） | 符合 | `metadata.yaml:6` `">=4.29,<5"`；`tests/test_manifest.py:122-131`（用 `packaging.SpecifierSet` 解析） | — |
| `astrbot_version` 下限不应把主流用户挡在门外（1.1：不满足即"阻止加载"） | **风险** | `metadata.yaml:6` 取 `>=4.29`，而 AstrBot master 当前为 **4.29.3**（<https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/__init__.py>）→ 4.28.x 及以前**全部被拒装**。README:103-106 声称 `views/` 自 4.29 起被扫描、`on_agent_done` 自 4.24 起提供，但仓库内**没有任何测试或 CI 验证过 4.29 这个下限** | Lead 决定：用 runtime-verifier 的实测结论把下限降到真实最小值（若 `views/` 在 4.24–4.28 可用，改成 `">=4.24,<5"`）；至少要在 README 给出"4.28 会怎样"的实测结论 |
| `display_name` / `short_desc` 为字符串；`tags` 为 `string[]`；`social_link` 为 HTTPS URL（1.7 可选字段表） | 符合 | `metadata.yaml:2,4,9-15`；`tests/test_manifest.py:134-147` | — |
| 插件 Logo：`logo.png` 置于插件根目录，1:1、推荐 256×256（1.1） | **风险（可选缺失）** | 全仓库无 `logo*`（`glob **/logo*` 为空）；`tests/test_manifest.py:150-167` 的必需清单里也没有它 | 可选增强 E1 |
| `metadata.yaml` 为 UTF-8、≤1MB、位于仓库/ZIP 根（1.1） | 符合 | `metadata.yaml` 15 行、首字节 `name:`（无 BOM）；`tests/test_manifest.py:198-201` | — |
| `tags` / `social_link` 属市场字段，宿主 `StarMetadata` 不读取（1.1） | 符合（说明性） | `astrbot/core/star/star.py` 的 `StarMetadata` 字段列表；`metadata.yaml:9-15` | 保留；不要指望宿主 UI 展示 |

### 2.2 _conf_schema.json

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| 每个节点必须有合法 `type`（1.2，宿主会 KeyError/TypeError） | 符合 | 递归校验 42 个叶子节点：**0 问题**（类型全部属于文档 10 种） | — |
| `object` 必须有 `items`（1.2） | 符合 | `_conf_schema.json:24-120`(learning)、`121-202`(retrieval)、`203-304`(policy)、`305-365`(memory)、`366-420`(feedback)、`421-441`(commands)、`442-452`(debug) 均有 `items` | — |
| `options` 与 `labels` 按索引一一对应（1.2） | 符合 | 全部等长；例 `_conf_schema.json:12-21`(3/3)、`37-48`(4/4)、`157-166`(3/3)、`329-338`(3/3) | — |
| `slider` 只用于 `int`/`float`（1.2） | 符合 | 所有 `slider` 均挂在数值节点上 | — |
| `float` 默认值必须是浮点字面量（文档默认值表 0.0） | 符合 | `30.0`(:61)、`0.5`(:71)、`0.8`(:139)、`1.5`(:210) 等 | — |
| `_special: select_provider` 用于提供商选择且结果为字符串（1.2） | 符合 | `_conf_schema.json:51-57`（`evaluator_provider_id`，`type: string`，`default` 为空串） | — |
| schema `default` 必须与代码默认值一致（否则 UI 显示与实际行为不符） | 符合 | 42 个叶子 `default` 与 `JitRLConfig()` 逐项比对：**0 处不一致** | — |
| 每个 schema 键都要被代码真正读取（避免"配置了但不生效"） | 符合 | 交叉比对：`jitrl/config.py` 的 `get(...)` 覆盖 **42/42** 个叶子；反向无"代码读但 schema 未声明"的键 | — |
| schema 中不应出现未实现字段 | 符合 | 无 `obsolete`/未实现键；CHANGELOG:28 记录已移除未实现的 `debug.expose_decision` 与宿主不渲染的 `apply_mode.subtitles` | — |
| 配置项文案 i18n（1.6） | **风险（可选缺失）** | 无 `.astrbot-plugin/i18n/` 目录；英文 WebUI 用户看到中文文案 | 可选增强 E2 |
| 敏感配置需 `secret: true`（1.2） | 不适用 | 本插件 schema 无 API Key/密码类配置 | — |
| 「`obsolete` 标记」是否存在 | **未验证** | 官方文档与 `AstrBotConfig` 源码均无该关键字 | 见第 5 节 |

### 2.3 生命周期与注册面

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| 入口文件必须叫 `main.py`（1.3） | 符合 | `main.py`（1483 行） | — |
| 插件类继承 `Star`，`__init__(self, context, config)` 并调用 `super().__init__(context)`（1.3） | 符合 | `main.py:84,93-101` | — |
| 实现 `terminate()` 释放资源（1.3） | 符合 | `main.py:149-159`（取消后台任务、关闭 MemoryBank） | — |
| `on_llm_request(event, req)` 两参签名（1.3） | 符合 | `main.py:454-459` | — |
| `on_agent_done(event, run_context, response)` 三参签名（1.3） | 符合 | `main.py:613-619` | — |
| 指令组用 `@filter.command_group` + `@<group>.command`（1.3） | 符合 | `main.py:839-842`；12 个子指令 `844,876,906,937,955,988,1024,1048,1095,1115,1138,1182` | — |
| `jitrl_recall` 用 `@filter.llm_tool` 且 docstring 声明参数（宿主据此生成 schema） | 符合 | `main.py:773-785` | — |
| 日志只能走 `from astrbot.api import logger`（1.3，官方"务必"） | 符合 | `main.py:41`；全仓库 grep `import logging` / `logging.getLogger` = **0 命中**；`tests/test_manifest.py:170-195`（AST 级断言） | — |
| 不得让异常冒泡破坏对话（1.9 第 4 条） | 符合 | 所有入口 `try/except` + `logger.warning(..., exc_info=True)`，例 `main.py:480-495,609-611,668-670` | — |
| 不按请求改写 `req.system_prompt`（本仓库自设的更严格约束，AGENTS.md:6-9） | 符合 | 全仓库无 `req.system_prompt` 赋值；注入走 `main.py:576-577`（`TextPart(text=block).mark_as_temp()` 后 `req.extra_user_content_parts.append`）；`mark_as_temp` 确实存在于 `astrbot/core/agent/message.py` 的 `ContentPart` | 保留；但该"市场规则"本身**未验证**（官方 docstring 反而允许改写），见第 5 节 |
| 依赖 `astrbot.core.agent.message.TextPart`（非 `astrbot.api.*` 公开面） | **风险** | `main.py:45` 是**模块级**导入：若上游把 `TextPart` 移出该模块，插件会**整体 import 失败、无法加载**，而不是降级 | Lead：把该导入移进 `on_llm_request` 或 `_inject_block()` 内的 `try/except ImportError`，失败时跳过注入（学习路径不受影响） |

### 2.4 数据目录 / 隐私 / 安全面

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| 持久化数据放在 `data/plugin_data/<plugin_name>/`（1.4） | 符合 | `main.py:102-103`：`Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"` 得 `memory.sqlite3`；README:114 与之一致；`get_astrbot_plugin_data_path` 实存于 `astrbot/core/utils/astrbot_path.py` | — |
| 不写插件自身目录（1.9 第 3 条） | 符合 | 所有落盘路径都基于 `self._db_path.parent`（`main.py:141,1036,1471`） | — |
| 无遥测 / 无第三方外发 / 无任意代码执行（审核关注点） | 符合 | grep 全仓库：`eval(`/`exec(` = 0、`subprocess/os.system/os.popen` = 0、`requests/urllib/httpx/aiohttp/socket/urlopen` = 0、`pickle/marshal/yaml.load` = 0。唯一外呼是宿主的 `provider.text_chat`（`main.py:224-227`），发给用户自己配置的模型 | — |
| 写入前脱敏 + 截断（README:249-261 自述） | 符合 | `main.py:502` 的 `redact(user_text)[:_TASK_CHAR_LIMIT]`、`main.py:792`；`main.py:135-144` 透传 `redact_secrets` | — |
| 资源有上限（审核关注 unbounded resource use） | 符合 | `_last_episode`/`_last_decision` 均 ≤512 条且有 TTL（`main.py:756-762,583-589`）；`max_entries`/`keep_episodes`/`keep_checkpoints` 有硬上限；评估器/重排器有超时（`main.py:373`） | — |
| 日志不泄漏敏感内容 | 符合 | 日志只打计数/ID/时长（例 `main.py:444-449,763-768`）；无用户原文、无记忆正文 | — |

### 2.5 视图与 Web API

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| 视图必须是 `views/<view_name>/index.html`（1.5） | 符合 | `views/dashboard/index.html`（73 行）+ `app.js`(412) + `style.css`(193)；无多余一级目录 | — |
| 用 `window.AstrBotPluginView`（别名 `AstrBotPluginPage`）桥接（1.5） | 符合 | `app.js:9`；`app.js:404` 的 `await bridge.ready()` | — |
| 路由必须含插件名前缀（1.5） | 符合 | `main.py:1216-1245`，10 条路由全部以 `/astrbot_plugin_jitrl/` 开头 | — |
| 视图端 endpoint **不带**插件名前缀（1.5） | 符合 | `app.js:329-341`（overview/curve/actions/episodes/checkpoints）、`:285`(checkpoint)、`:314`(rollback、checkpoint_delete)、`:388`(forget)、`:354` 的 `bridge.download("export", {}, "jitrl-export.json")` | — |
| `register_web_api(route, handler, methods, desc)` 四参签名（1.5；`context.py`） | 符合 | `main.py:1253`；路由表 `main.py:1215-1246`；对旧宿主用 `getattr(self.context, "register_web_api", None)` 降级（`:1247-1250`） | — |
| 后端 handler 用 `astrbot.api.web` 的 helper（1.5） | 符合 | `json_response`/`error_response`/`file_response`/`request` 分区导入，`main.py:1264,1312,1387,1438,1466` | — |
| Web API 必须要求认证（1.5 认证事实） | 符合 | 宿主 `Depends(require_plugin_scope)` 统一兜底（`dashboard/api/plugins.py`）；插件未自造弱鉴权 | — |
| GET 参数必须做范围校验（1.5 "不要信任视图传来的……数值范围"） | 符合 | `main.py:1316-1319`(buckets 2..60)、`:1335-1336`(limit 1..200)、`:1352-1353`(limit 1..100) | — |
| POST 体必须是 JSON 对象 + 白名单校验（1.5） | 符合 | `main.py:1390-1403`（`confirm is True` 强制、`scope` 仅 all/session、session 必须有非空 `umo`）、`:1442-1451` | — |
| 破坏性操作要二次确认（审核关注点；明文规则**未验证**） | **风险** | 检查点三类操作强制 `confirm: true`（`main.py:1393-1394`），但 **`POST /forget` 只要请求体含 `scope=all` 就直接清库**（`main.py:1445-1447`），仅靠前端两步确认（`app.js:371-397`）；清库前会自动备份（README:217/256） | Lead：给 `_web_forget` 加 `payload.get("confirm") is not True → error_response(...)`，与检查点路由一致（前端 `app.js:388` 同步补 `confirm: true`） |
| 文件下载不能拼接用户输入路径（1.5） | 符合 | `main.py:1471-1480`：固定文件名 `jitrl-export.json` 落在插件数据目录，无路径拼接 | — |
| 视图必须能在无 modal 的 sandbox 里工作（1.5 安全约束） | 符合 | `app.js:4-7` 注释明说 `allow-modals` 缺失；全文件无 `alert/confirm/prompt`；改用页内两步确认 `app.js:363-397`、`:297-324` | — |
| 视图应跟随亮/暗主题（1.5） | 符合 | `style.css:1-12`（默认深色）+ `style.css:14-24`（`html[data-theme="light"]` 覆盖） | — |
| 视图元数据 i18n（1.6） | **风险（可选缺失）** | 无 `.astrbot-plugin/i18n/`，侧栏/详情页标题回退目录名或插件名 | 可选增强 E2 |
| `/jitrl export` 把服务器绝对路径回显到聊天（`main.py:1036-1043`） | **风险** | 生成 `export-<ts>.json` 后把绝对路径直接 `plain_result` 发给用户；每次导出新增文件且**永不清理**（磁盘占用无上限） | Lead：回复只给文件名（或改走面板下载），并轮换保留最近 N 份导出 |

### 2.6 发布资产（requirements / README / CHANGELOG / CI / ZIP）

| 要求（来源） | 判定 | 证据 | 修复 |
| --- | --- | --- | --- |
| `requirements.txt` 只在需要第三方依赖时声明（1.1） | 符合 | `requirements.txt:1-2` 只有注释，零第三方运行时依赖；README:112 与之一致；宿主安装空 requirements 为空操作 | — |
| 不使用 `requests` 做网络请求（1.9 第 6 条） | 符合 | grep = 0 命中 | — |
| `ruff` 格式要求（1.9 第 5 条） | 符合 | `.github/workflows/ci.yml:34-37` 跑 `ruff check .` + `ruff format --check .`；`pyproject.toml:7-29` 固定规则 | 建议 pin ruff 版本（E4） |
| README 必须有安装方式（1.1/1.8） | 符合 | `README.md:94-114`（git clone / 放目录 / Release ZIP / AstrBot 版本要求 / 零依赖 / 数据库位置） | — |
| README 必须有配置说明与配置表 | 符合 | `README.md:226-245`（关键配置 + 默认值，与 schema 抽查一致） | — |
| README 必须有用法/指令清单 | 符合 | `README.md:150-222`（12 条指令 + 检查点流程 + Agent 工具 + 面板 10 个接口） | — |
| README 必须有许可证说明（1.9 第 9 条） | 符合 | `README.md:388-390`（AGPL-3.0，与 AstrBot 一致）；`LICENSE` 为 AGPL-3.0 全文 | — |
| README 必须说明灵感/引用来源（1.9 第 8 条） | 符合 | `README.md:359-374`（BibTeX + "独立实现、未移植论文官方代码"声明） | — |
| README 版本徽章与 metadata 一致（AGENTS.md:10-11） | 符合 | `README.md:4` 的 `badge/version-0.1.1` ↔ `metadata.yaml:5` ↔ `jitrl/__init__.py:17` ↔ `CHANGELOG.md:8`；`scripts/build_release.py:67-95` 与 `tests/test_manifest.py:99-112` 双重校验 | — |
| README 截图/效果图 | **风险（可选缺失）** | README 无任何图片；市场卡片体验受影响（明文规则**未验证**） | 可选增强 E3 |
| CHANGELOG 结构（Keep a Changelog + 版本链接） | 符合 | `CHANGELOG.md:5-6,8,72-73` | — |
| CI 覆盖 lint + 单测 + 前端测试 | 符合 | `.github/workflows/ci.yml:34-42,44-50`（ruff / pytest / `node --test tests/test_dashboard.cjs`） | — |
| CI 应验证发布 ZIP 与 16MB 上限（1.7） | **风险** | `ci.yml` 未执行 `python scripts/build_release.py`；ZIP 布局/体积只在本地靠人跑 | 可选增强 E4（本地 `dist/astrbot_plugin_jitrl-v0.1.1.zip` 存在，脚本会自检） |
| ZIP 必须单根目录且根下有 `metadata.yaml`（1.7） | 符合 | `scripts/build_release.py:26` 定义 `PACKAGE = "astrbot_plugin_jitrl"`；归档名统一为 `{PACKAGE}/{name}`（`:122,132`）；实测安装产物布局含 `installed/metadata.yaml` 与 `installed/jitrl/` | — |
| ZIP 必须排除开发工具/测试/缓存/数据库/检查点（1.7） | 符合 | `scripts/build_release.py:32-43`（固定 include：main.py/metadata.yaml/_conf_schema.json/requirements.txt/README/CHANGELOG/LICENSE/jitrl/views）+ `:46-47`（排除 `__pycache__/.pytest_cache/.ruff_cache` 与 `*.pyc/*.pyo`）；tests/scripts/.github/pyproject/data/dist 根本不在 include 列表 | E4：再加后缀/基名拒绝表（`*.sqlite3`、`*.db*`、`jitrl-export.json`、`export-*.json`、`.DS_Store`、`node_modules`），防止把调试库丢进 `jitrl/` 或 `views/` 后被一起打包 |
| ZIP ≤16MB（1.7） | 符合 | `scripts/build_release.py:29` 的 `MAX_ARCHIVE_BYTES = 16 * 1024 * 1024`，超限 `unlink` + `SystemExit`（`:157-159`） | — |
| 仓库不得提交运行时数据/构建产物（1.7 "清理不必要的文件"） | 符合 | `git ls-files` 41 个文件，**无** data/、dist/、.tmp-pytest/、__pycache__/、*.db；`.gitignore:1-23` 覆盖这些；`.gitattributes:6-13` 对 tests/scripts/pyproject/.github/dist 设 `export-ignore`（GitHub 归档安装路径同样干净） | — |
| 新增 `docs/` 不应进入安装包（1.7） | **风险** | 本审计新增 `docs/compliance-audit.md`；`build_release.py` 的 include 列表不含 docs（ZIP 安全），但 `.gitattributes` 没有 `/docs export-ignore`，GitHub 归档安装会带上它 | Lead：在 `.gitattributes` 增加一行 `/docs export-ignore`（1 行改动，零风险） |
| 市场身份 author/name 与仓库内 metadata 一致、全局唯一（1.7） | 符合 | `plugin_id = 2841649220/astrbot_plugin_jitrl`；`metadata.yaml:1,7`；`name` 是合法 Python 标识符（宿主 `_validate_importable_name` 要求） | — |

---

## 3. 必须修复（按优先级）

> 「必须」= 会实际影响市场审核 / 可安装性 / 用户数据安全；不含纯风格问题。5 条里 1 条需 Lead 与 runtime-verifier 协同，其余 4 条是 1–15 行的确定性改动。

| # | 优先级 | 文件:行 | 改动 | 谁来做 |
| --- | --- | --- | --- | --- |
| M1 | P0 | `main.py:45` | 把 `from astrbot.core.agent.message import TextPart` 从模块顶层移进 `on_llm_request`（或新的 `_inject_block()` helper）内的 `try/except ImportError`，失败时 `logger.debug` 后跳过注入。理由：这是全插件唯一的**非 astrbot.api.* 公开面导入**，一旦上游搬迁该类，插件**整体导入失败**（宿主判为"加载失败插件"），而不是优雅降级 | **Lead** |
| M2 | P0 | `metadata.yaml:6` | 用 runtime-verifier 实测的最小版本替换 `>=4.29,<5`。AstrBot 最新版就是 4.29.3，`>=4.29` 等于"只允许最新一版"，4.24–4.28 用户安装时会被宿主硬拦（"插件会被阻止加载并提示版本不兼容"）。若实测确认 `on_agent_done`(4.24+) 与 `views/` 在 4.28 均可用，改为 `>=4.24,<5` 并在 README:103-106 写明依据 | **Lead**（依赖 runtime-verifier 结论） |
| M3 | P1 | `main.py:1432-1453` + `views/dashboard/app.js:388` | `POST /astrbot_plugin_jitrl/forget` 增加 `if payload.get("confirm") is not True: return error_response("清空记忆需要 confirm: true。")`；前端 `bridge.apiPost("forget", { scope: "all" })` 改为 `{ scope: "all", confirm: true }`。理由：与 `checkpoint/rollback/checkpoint_delete` 的三处 `confirm` 校验（`main.py:1393-1394`）保持一致——这是审核最容易提的"破坏性操作需二次确认"点 | **Lead** |
| M4 | P1 | `main.py:1036-1043` | `/jitrl export` 不再回显服务器绝对路径（只回文件名 + 提示走面板下载），并在写入前清理超过 5 份的旧 `export-*.json`。理由：向聊天回显 `C:/.../data/plugin_data/...` 属信息泄漏（群聊场景下管理员执行也会留痕），且导出文件无上限增长 | **Lead** |
| M5 | P1 | `.gitattributes` | 追加 `/docs export-ignore`（本审计新增了 `docs/`）。理由：GitHub 归档安装会带上 docs，属"发布包应只含必需文件"的清洁度要求；`build_release.py` 的 include 列表已天然排除 docs | **Lead**（或授权我加，1 行） |

同批建议（不阻塞）：`scripts/build_release.py:46-47` 的排除机制增加后缀/基名拒绝表（`*.sqlite3`、`*.db*`、`jitrl-export.json`、`export-*.json`），防止调试库被误打包。

## 4. 可选增强（市场观感 / 审核加分）

| # | 项 | 具体做法 |
| --- | --- | --- |
| E1 | 插件 Logo | 加 `logo.png`（1:1、256×256、<50KB）到仓库根；这是官方描述的 Logo 机制（文件，不是 metadata 字段），市场卡片与插件详情页都会用到 |
| E2 | i18n | 加 `.astrbot-plugin/i18n/zh-CN.json` 与 `en-US.json`，至少覆盖 metadata 的 display_name/short_desc/desc、高频 config description/hint、`views.dashboard.title/description`。缺失不影响加载（官方明确回退），但英文 WebUI 下现在全是中文 |
| E3 | README 截图 + 隐私披露 | 面板 KPI / 学习曲线 / 检查点三张图 + 图注（市场详情页会渲染 README 图片）。并在 README 补一条：`apply_mode=logit` 与 `learning.evaluator_mode=llm/auto` 会把轨迹文本（已脱敏）发送给你配置的模型提供商用于打分/重排 —— 目前 README:249-261 只讲了存储侧脱敏，没讲评估器外发 |
| E4 | CI 加发布校验 | `ci.yml` 增加一步 `python scripts/build_release.py` 并 `actions/upload-artifact` 上传 `dist/*.zip`，把"单根目录/版本链/16MB"从本地人工变成 CI 门禁；同时给 `ruff` 固定版本，避免 `ruff format --check` 随上游默认规则漂移而误红 |
| E5 | `support_platforms` | 运行时代码确实平台无关（只用 `event.get_platform_name()` 做可选分片）。可在 metadata 声明常用 key，或干脆不声明并在 README 写明"纯 Agent 层、平台无关"。缺失不违规（可选字段），但市场筛选会少一个维度 |

## 5. 未验证（没有找到明文规则，不做断言）

1. **「`obsolete` 标记」**：官方 schema 文档列出的字段（type/description/hint/obvious_hint/default/items/invisible/secret/options/labels/subtitles/slider/editor_mode/editor_language/editor_theme/_special）里**没有** `obsolete`，`AstrBotConfig` 源码也没有 —— 该机制是否存在、语义为何未验证，建议不要使用。
2. **「市场审核要求插件不得改写 `req.system_prompt`」**：AGENTS.md:6-9 与 CHANGELOG:31 把它表述成"市场复审规则"，但官方文档只有**相反**证据 —— `register_on_llm_request` 的官方 docstring 示例就是 `req.system_prompt += ...`。本仓库做法（`extra_user_content_parts` + `mark_as_temp()`）客观上更优（不破坏 prompt 缓存、不落库），**建议保留**，但请把它描述为"本项目的实现选择"而非"官方硬规定"，否则审核方可能要求出处。
3. **「日志必须走 `astrbot.api.logger`」**：这条有官方明文（"请务必使用"，见 1.3），但它出现在**插件开发指南**而非市场审核规范里；README:318-320 称其为"市场安全审查要求"属归属错误，但结论正确。
4. **市场审核的具体清单 / 自动检查项 / 常见拒稿理由**：发布入口 `https://cloud.astrbot.app/publish` 是 SPA（HTML 仅 1.9KB，内容由 `/_app/index-*.js` 渲染），未提供可引用的静态审核条款；旧收集仓库已 DEPRECATED。故本审计只能以"发布文档 + 市场 JSON 规范 + 宿主源码的强制行为"为准。
5. **`astrbot_version` 4.29 下限的技术依据**：README:103-106 声称 `views/` 自 4.29 起被扫描、`on_agent_done` 自 4.24 起提供。我只验证了"master(4.29.3) 同时具备二者"，**没有**逐版本回溯引入点，故"`views/` 是否 4.29 才引入"标记未验证（这也是 M2 的前提）。
6. **16MB 之外的体积/文件数限制**：文档只写了 16MB；是否有单文件上限、文件数上限未验证。

---

## 6. 复现方式（审计方法，供复核）

- 规范抓取：`web_fetch` 取 docs.astrbot.app 各页（VitePress 站点），并用 `r.jina.ai` 文本代理绕过 100KB 截断拿到 `plugin-pages` 全文（bridge API / 安全约束两节）。
- 宿主行为核对：直接读 `raw.githubusercontent.com/AstrBotDevs/AstrBot/master/...` 的 `star/updater.py`、`star/star_manager.py`、`star/star.py`、`star/context.py`、`config/astrbot_config.py`、`zip_updater.py`、`utils/astrbot_path.py`、`agent/message.py`、`dashboard/api/plugins.py`、`api/web.py`、`api/event/filter/__init__.py`。
- schema 校验：递归遍历 `_conf_schema.json` 全部 42 个叶子（type/items/slider/options-labels/default），并与 `jitrl/config.py` 的 `JitRLConfig()` 默认值双向比对（0 差异、0 孤儿键）。临时脚本仅用于本次核对，已删除，未在仓库留下文件。
- 安全面：对 `main.py` + `jitrl/` 做危险 API grep（eval/exec/subprocess/网络库/反序列化/stdlib logging），全部 0 命中。
- 仓库洁净度：`git ls-files`（只读）确认无运行期数据与构建产物入库。
