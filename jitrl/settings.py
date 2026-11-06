"""Schema-driven dashboard settings shared by both release lines."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import JitRLConfig
from .neural.config import LensConfig


def settings_schema() -> dict[str, Any]:
    """Use the same field descriptions and bounds as AstrBot's config editor."""
    path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def settings_values(config: JitRLConfig) -> dict[str, Any]:
    """Return JSON values (including lists instead of dataclass tuples)."""
    return json.loads(json.dumps(asdict(config)))


def update_settings(
    config: JitRLConfig, patch: Any, schema: dict[str, Any]
) -> JitRLConfig:
    """Validate a partial update strictly; invalid forms never silently clamp."""
    values = deepcopy(settings_values(config))

    def merge(target: dict, updates: Any, fields: dict, prefix: str = "") -> None:
        if not isinstance(updates, dict) or not updates:
            raise ValueError(f"{prefix or 'values'} 必须是非空 JSON 对象。")
        for key, value in updates.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in fields:
                raise ValueError(f"未知设置：{path}")
            spec = fields[key]
            kind = spec["type"]
            if kind == "object":
                merge(target[key], value, spec["items"], path)
                continue
            valid = False
            if kind == "bool":
                valid = isinstance(value, bool)
            elif kind in {"int", "float"}:
                try:
                    valid = (
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and (kind != "int" or int(value) == value)
                    )
                except OverflowError:
                    valid = False
                bounds = spec.get("slider", {})
                if valid:
                    valid = (
                        bounds.get("min", -math.inf)
                        <= value
                        <= bounds.get("max", math.inf)
                    )
            elif kind in {"string", "text"}:
                valid = isinstance(value, str)
            elif kind == "list":
                valid = (
                    isinstance(value, list)
                    and bool(value)
                    and all(isinstance(item, str) and item.strip() for item in value)
                )
            if not valid or ("options" in spec and value not in spec["options"]):
                raise ValueError(f"设置 {path} 的类型、范围或选项无效。")
            target[key] = int(value) if kind == "int" else value

    merge(values, patch, schema)
    LensConfig(**values["neural"])
    return JitRLConfig.from_raw(values)
