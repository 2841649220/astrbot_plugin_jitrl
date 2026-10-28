# Project release policy

- For routine features and fixes, increment only the patch (last) version number.
- Increment the middle version number only for large architectural improvements.
- This branch targets patch release `0.2.2` on the `0.2.0` architecture and incorporates the `0.1.4` mainline retrieval, learning-attribution and tool-gating improvements; keep neural and MiTA work under `0.2.2` until the user explicitly requests another version. `0.1.3` and the original `0.2.1` build introduced robot-instance isolation with a `platform_id:self_id` key; the re-released `0.2.1` fixes the key to the platform instance alone and consolidates the legacy directories. Its V0.1 compatibility baseline is `0.1.4`; `0.1.2` added retrieval and checkpoint improvements, `0.1.1` was re-released with market fixes, and `0.1.0` was the first public version.
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
