"""Settings form coverage and numerical validation at the shared schema boundary."""

from dataclasses import fields, is_dataclass

import pytest

from jitrl.config import JitRLConfig
from jitrl.settings import settings_schema, settings_values, update_settings


def test_every_runtime_setting_has_an_editable_schema_field():
    def check(config, schema):
        assert {field.name for field in fields(config)} == set(schema)
        for field in fields(config):
            value = getattr(config, field.name)
            if is_dataclass(value):
                assert schema[field.name]["type"] == "object"
                check(value, schema[field.name]["items"])
            else:
                assert schema[field.name]["type"] in {
                    "bool",
                    "int",
                    "float",
                    "string",
                    "text",
                    "list",
                }

    check(JitRLConfig(), settings_schema())


def test_settings_roundtrip_and_numeric_bounds_match_runtime():
    config = JitRLConfig()
    schema = settings_schema()
    assert settings_values(
        update_settings(config, settings_values(config), schema)
    ) == settings_values(config)
    for group, spec in schema.items():
        for key, field in spec.get("items", {}).items():
            if field["type"] not in {"int", "float"}:
                continue
            for value in field["slider"]["min"], field["slider"]["max"]:
                try:
                    updated = update_settings(config, {group: {key: value}}, schema)
                except ValueError:
                    # Neural tensor shapes and memory budgets have joint bounds.
                    assert group == "neural"
                    continue
                assert getattr(getattr(updated, group), key) == value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_config_falls_back_to_valid_default(bad):
    config = JitRLConfig.from_raw({"policy": {"beta": bad}, "retrieval": {"k": bad}})
    assert config.policy.beta == 1.5 and config.retrieval.k == 10
