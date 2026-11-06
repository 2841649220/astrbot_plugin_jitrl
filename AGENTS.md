# Project release policy

- For routine features and fixes, increment only the patch (last) version number.
- Increment the middle version number only for large architectural improvements.
- The current V0.2 release is `0.2.4` (2026-11-07), carrying V0.1.5 full data reset and persistent dashboard settings, with independent hyperparameter reset and neural-data/weight cleanup. Its V0.1 compatibility baseline is `0.1.5`; routine follow-ups increment the patch version.
- Keep V0.2 neural work on `v0.2.0-neural-lens`; do not merge it into the V0.1 line without an explicit request.
- Neural training is optional and scope-isolated. Preserve exact historical retrieval snapshots, exclude pending rewards from supervised targets, and never claim local candidate scores are remote logits or causal uplift.
- The default neural backbone uses actual MiTA in all four token-attention blocks. Keep landmark compression, reusable top-k KV experts and jointly normalized routing intact; GQA/hybrid/legacy are explicit comparisons, not silent substitutions.
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
