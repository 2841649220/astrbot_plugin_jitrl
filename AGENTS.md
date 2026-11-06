# Project release policy

- For routine features and fixes, increment only the patch (last) version number.
- Increment the middle version number only for large architectural improvements.
- The current release is `0.1.5` (2026-11-07), adding full robot-data reset while preserving hyperparameters, persistent dashboard parameter settings and an independent parameter-reset button. V0.2.4 carries the same mainline changes on `v0.2.0-neural-lens`; keep neural work off main.
- Market review rules to keep satisfying: log only through `from astrbot.api import logger`
  (never the standard library `logging`), and never rewrite `req.system_prompt` per request —
  injected context goes to `req.extra_user_content_parts` and is marked `mark_as_temp()` so it
  is not persisted and the prompt-cache prefix stays stable.
- Keep `metadata.yaml`, `jitrl/__init__.py`, the README version badge, changelog
  and release tag consistent.
- Build installable ZIPs with a single `astrbot_plugin_jitrl/` root directory.
  `python scripts/build_release.py` enforces the layout and version chain.
  Include Python modules, metadata, configuration schema, requirements, dashboard
  assets, license and documentation. Exclude development tooling, tests, caches,
  local databases, checkpoints and exports.
