# Project release policy

- For routine features and fixes, increment only the patch (last) version number.
- Increment the middle version number only for large architectural improvements.
- The current release is `0.1.4` (2026-10-28), improving exact retrieval, current-turn reward attribution and conservative tool gating. `0.1.3` was re-released on 2026-10-27 with the robot-identity fix (key on the platform instance id, not `self_id`, and consolidate legacy directories); `0.1.2` added retrieval and checkpoint improvements, `0.1.1` was re-released with market fixes, and `0.1.0` was the first public version.
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
