# Project release policy

- For routine features and fixes, increment only the patch (last) version number.
- Increment the middle version number only for large architectural improvements.
- This branch targets the architectural release `0.2.0`. Its V0.1 baseline is `0.1.2`; `0.1.1` was re-released with market fixes and `0.1.0` was the first public version.
- Keep V0.2 neural work on `codex/v0.2.0-neural-lens`; do not merge it into the V0.1 line without an explicit request.
- Neural training is optional and scope-isolated. Preserve exact historical retrieval snapshots, exclude pending rewards from supervised targets, and never claim local candidate scores are remote logits or causal uplift.
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
