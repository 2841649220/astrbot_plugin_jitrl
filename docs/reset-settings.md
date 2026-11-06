# 数据清除与参数设置

V0.1.5 和 V0.2.4 共用以下管理行为。

## 清除数据

面板选择机器人后，点击“清除数据并初始化”，在 6 秒内再次点击确认；聊天中也可使用 `/jitrl forget all`。
清除范围是所选机器人的经验、Episodes、检查点（手动、自动及旧安全备份）、导出文件、旧迁移副本、反馈与决策缓存、运行累计计数。
不会创建删除前备份。其他机器人有各自独立的数据。

**清除数据保留超参数及 AstrBot 已保存的插件配置。** 清除后自动检查点从零重新计数；空库重载不会创建空基线。
清除前已开始的请求和评估结果通过 generation 校验失效，新的对话可以正常学习。
仅清理会话可使用 `/jitrl forget session`，不创建新备份；已有跨会话检查点仍须单独管理。

## 参数设置

面板“参数设置”按组提供所有配置字段：数字输入框、开关、选项菜单与关键词列表。
字段说明与边界复用 `_conf_schema.json`，数字必须有限，整数、范围和枚举在后端再次校验。
“保存并应用”通过 AstrBot 提供给插件的配置对象调用 `save_config()` 持久化，并更新已打开的经验库与策略缓存。
参数对本插件的所有机器人生效；降低存储上限在下次写入/淘汰时执行。
尚无机器人时也可编辑设置。

**“重置超参数”是独立按钮**，将表单填回默认值，再点击“保存并应用”确认。该操作不清理学习数据。
“重新加载设置”放弃表单中未保存的改动并读取当前配置。多页面同时编辑时，旧版本保存返回冲突提示，不覆盖新设置。
保存失败保留当前配置和表单内容。未提供 `save_config()` 的测试或定制宿主会明确提示仅本次运行生效。

## AstrBot 接口

沿用官方 [Plugin Views](https://github.com/AstrBotDevs/AstrBot/blob/86c24e846c0de78830f6a0bc402049edb0f0c671/docs/en/dev/star/guides/plugin-pages.md)
提供的 `register_web_api`、`astrbot.api.web` 请求代理与桥接 `apiGet/apiPost` 契约；
配置保存使用官方 [AstrBotConfig.save_config](https://github.com/AstrBotDevs/AstrBot/blob/86c24e846c0de78830f6a0bc402049edb0f0c671/astrbot/core/config/astrbot_config.py)。
设置接口为 `GET /astrbot_plugin_jitrl/settings` 与 `POST /astrbot_plugin_jitrl/settings/save`。

本地回归覆盖数据库清空、损坏快照、自动保存间隔、旧评估回写、重载、机器人隔离、配置持久化及保存失败、
数据清除保留自定义参数、默认值按钮与无机器人时的设置入口。完整 AstrBot 部署验证仍需宿主环境。
