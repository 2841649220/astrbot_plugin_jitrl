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
数据清除保留自定义参数、默认值按钮与无机器人时的设置入口。另外使用 AstrBot v4.29.0 与 master（`86c24e8`）的真实 PluginRequest/json_response 请求响应辅助代码验证了设置保存与清空后的查询；完整 AstrBot 部署验证仍需宿主环境。

## V0.2.x 神经扩展

V0.2.4 的全量清除同时删除当前机器人的 Neural Lens 决策、执行/奖励记录、模型引用与本地权重；保留 epoch 屏障阻止旧任务提交。
未加载的神经目录也会清除，无需安装 PyTorch。`/jitrl lens reset confirm` 继续支持只清除当前会话的神经数据。
“参数设置”包含全部神经配置及模型维度、注意力头数、检索预算、回报尺度和随机种子。
保存时检查维度整除关系与联合预算，等待在运行的模型工作完成后更新配置并清理模型缓存，历史检索快照保持原样。
更改主干或模型结构后需要重新训练本地权重；默认仍为四层真正的 MiTA，训练默认关闭。
