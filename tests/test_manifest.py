"""Release-manifest checks that run without an AstrBot installation.

The plugin is published through the AstrBot plugin market, which reads
``metadata.yaml`` from the repository and expects the same version to appear in
the package, the README and the changelog. Those rules used to be checked by
hand, and one YAML quoting slip (``author: 2841649220`` parsing as an integer)
was enough to make AstrBot reject the metadata and degrade the plugin's identity
to ``unknown/unknown``. These tests make the contract executable.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from jitrl import __version__  # noqa: E402

METADATA = PLUGIN_ROOT / "metadata.yaml"
README = PLUGIN_ROOT / "README.md"
CHANGELOG = PLUGIN_ROOT / "CHANGELOG.md"
SCHEMA = PLUGIN_ROOT / "_conf_schema.json"

REQUIRED_FIELDS = ("name", "desc", "version", "author")
"""Fields AstrBot refuses to load a plugin without."""

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
GITHUB_REPO = re.compile(r"^https://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
README_VERSION = re.compile(r"badge/version-(\d+\.\d+\.\d+)-")
CHANGELOG_VERSION = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)
SCHEMA_TYPES = {
    "bool",
    "int",
    "float",
    "string",
    "text",
    "list",
    "file",
    "object",
    "template_list",
    "dict",
}
"""Node types AstrBot's configuration renderer understands."""

SHIPPED_SOURCES = (
    PLUGIN_ROOT / "main.py",
    *sorted((PLUGIN_ROOT / "jitrl").glob("*.py")),
)
"""Modules inside the published package (tests and tooling are not shipped)."""


@pytest.fixture(scope="module")
def metadata() -> dict[str, Any]:
    """Return the parsed plugin manifest.

    Returns:
        The `metadata.yaml` mapping.
    """
    with METADATA.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), "metadata.yaml must contain a mapping"
    return data


def test_required_fields_are_non_empty_strings(metadata: dict[str, Any]) -> None:
    """Reject YAML scalars AstrBot would coerce to another type."""
    for field in REQUIRED_FIELDS:
        assert field in metadata, f"metadata.yaml is missing {field}"
        value = metadata[field]
        assert isinstance(value, str), (
            f"metadata.yaml: {field} must be a string, got {type(value).__name__}"
        )
        assert value.strip(), f"metadata.yaml: {field} must not be blank"


def test_plugin_name_is_importable(metadata: dict[str, Any]) -> None:
    """Keep the package name usable by AstrBot's importlib loader."""
    name = metadata["name"]
    assert name.startswith("astrbot_plugin_"), (
        "plugin names use the astrbot_plugin_ prefix"
    )
    assert name == name.lower(), "plugin names are lower case"
    assert " " not in name, "plugin names contain no spaces"
    assert name.isidentifier(), "plugin names are valid Python identifiers"
    assert "/" not in metadata["author"], "author is a namespace, not a path"


def test_version_is_semver_and_shared_everywhere(metadata: dict[str, Any]) -> None:
    """Keep metadata, code, README badge and changelog in lockstep."""
    version = metadata["version"]
    assert SEMVER.match(version), f"version {version!r} is not MAJOR.MINOR.PATCH"
    assert version == __version__, "metadata.yaml and jitrl.__version__ disagree"

    readme_badge = README_VERSION.search(README.read_text(encoding="utf-8"))
    assert readme_badge, "README has no version badge"
    assert readme_badge.group(1) == version, "README badge is out of date"

    changelog = CHANGELOG.read_text(encoding="utf-8")
    latest = CHANGELOG_VERSION.search(changelog)
    assert latest, "CHANGELOG has no version heading"
    assert latest.group(1) == version, "CHANGELOG's newest entry is not this version"


def test_repository_url_matches_the_market_spec(metadata: dict[str, Any]) -> None:
    """Only canonical public GitHub repository URLs are accepted."""
    repo = metadata["repo"]
    assert GITHUB_REPO.match(repo), f"repo {repo!r} is not a GitHub repository URL"
    assert not repo.endswith("/"), "repo URL must not end with a slash"


def test_astrbot_version_is_a_pep440_range(metadata: dict[str, Any]) -> None:
    """The host parses this field with `packaging`, without a `v` prefix."""
    specifier = metadata["astrbot_version"]
    assert isinstance(specifier, str)
    assert not specifier.startswith("v"), "astrbot_version must not use a v prefix"
    try:
        parsed = SpecifierSet(specifier)
    except InvalidSpecifier as exc:  # pragma: no cover - failure path
        pytest.fail(f"astrbot_version {specifier!r} is not a PEP 440 range: {exc}")
    assert any(item.operator == ">=" for item in parsed), "declare a lower bound"


def test_optional_market_fields_have_the_documented_types(
    metadata: dict[str, Any],
) -> None:
    """Catch optional fields that would fail the market schema."""
    for field in ("display_name", "short_desc"):
        if field in metadata:
            assert isinstance(metadata[field], str)
            assert metadata[field].strip()
    if "tags" in metadata:
        tags = metadata["tags"]
        assert isinstance(tags, list) and tags, "tags must be a non-empty list"
        assert all(isinstance(tag, str) and tag.strip() for tag in tags)
    if "social_link" in metadata:
        assert str(metadata["social_link"]).startswith("https://")


def test_installable_layout_exists() -> None:
    """Keep every file the market package and AstrBot loader rely on."""
    required = (
        "main.py",
        "__init__.py",
        "metadata.yaml",
        "_conf_schema.json",
        "requirements.txt",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "jitrl/__init__.py",
        "views/dashboard/index.html",
        "views/dashboard/app.js",
        "views/dashboard/style.css",
    )
    missing = [path for path in required if not (PLUGIN_ROOT / path).is_file()]
    assert not missing, f"missing release files: {missing}"


def test_plugin_sources_log_through_astrbot_only() -> None:
    """The market safety review rejects stdlib logging inside a plugin.

    Two rules are enforced: no shipped module may import the standard library
    ``logging`` package, and every module that logs must take the host logger
    from ``astrbot.api``.
    """
    offenders: list[str] = []
    missing_host_logger: list[str] = []
    for path in SHIPPED_SOURCES:
        relative = path.relative_to(PLUGIN_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                if "logging" in names:
                    offenders.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] == "logging":
                    offenders.append(f"{relative}:{node.lineno}")
        if "logger." in source and "from astrbot.api import logger" not in source:
            missing_host_logger.append(relative)
    assert not offenders, f"stdlib logging is not allowed in the plugin: {offenders}"
    assert not missing_host_logger, (
        f"these modules log without importing astrbot.api.logger: {missing_host_logger}"
    )


def test_metadata_file_is_plain_utf8() -> None:
    """AstrBot reads the manifest as UTF-8 and chokes on a BOM."""
    raw = METADATA.read_bytes()
    assert raw.startswith(b"name:"), "metadata.yaml must not start with a BOM"


def test_config_schema_is_renderable() -> None:
    """Check the node shapes AstrBot's configuration UI depends on."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert isinstance(schema, dict) and schema

    def walk(node: dict[str, Any], path: str) -> None:
        assert "type" in node, f"{path} has no type"
        assert node["type"] in SCHEMA_TYPES, f"{path} uses unknown type {node['type']}"
        if node["type"] == "object":
            assert isinstance(node.get("items"), dict), f"{path} has no items"

    for key, node in schema.items():
        assert isinstance(node, dict), f"{key} is not a schema node"
        walk(node, key)
