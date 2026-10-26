"""Build the installable AstrBot plugin archive.

AstrBot installs a plugin from a ZIP whose entries all live under exactly one
plugin root directory containing ``metadata.yaml``. The plugin market also
rejects oversized archives, and a release is only trustworthy when the version
in ``metadata.yaml``, ``jitrl/__init__.py``, the README badge and the changelog
agree. This script enforces all of that, packs a fixed include list (so tests,
caches and local databases can never leak into a release) and prints the
SHA-256 of the archive for the release notes.

Usage::

    python scripts/build_release.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE = "astrbot_plugin_jitrl"
"""Root directory name inside the archive and inside ``data/plugins``."""

MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
"""The plugin market rejects archives larger than 16 MiB."""

INCLUDE: tuple[str, ...] = (
    "__init__.py",
    "main.py",
    "metadata.yaml",
    "_conf_schema.json",
    "requirements.txt",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "jitrl",
    "views",
)
"""Release payload, in the order it is written."""

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

VERSION_LINE = re.compile(r"^version:\s*(\S+)\s*$", re.MULTILINE)
CODE_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
README_VERSION = re.compile(r"badge/version-(\d+\.\d+\.\d+)-")
CHANGELOG_VERSION = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _read(path: Path) -> str:
    """Return the UTF-8 text of a repository file.

    Args:
        path: File to read.

    Returns:
        The decoded contents.
    """
    return path.read_text(encoding="utf-8")


def _release_version() -> str:
    """Return the release version after checking every place it is declared.

    Returns:
        The version shared by the manifest, the package, the badge and the
        changelog.

    Raises:
        SystemExit: When one of those places disagrees.
    """
    metadata = _read(ROOT / "metadata.yaml")
    match = VERSION_LINE.search(metadata)
    if not match:
        raise SystemExit("metadata.yaml has no version field")
    version = match.group(1).strip().strip("\"'")

    declared = {
        "metadata.yaml": version,
        "jitrl/__init__.py": _first(
            CODE_VERSION, _read(ROOT / "jitrl" / "__init__.py")
        ),
        "README.md": _first(README_VERSION, _read(ROOT / "README.md")),
        "CHANGELOG.md": _first(CHANGELOG_VERSION, _read(ROOT / "CHANGELOG.md")),
    }
    mismatched = {name: value for name, value in declared.items() if value != version}
    if mismatched:
        detail = ", ".join(f"{name}={value}" for name, value in mismatched.items())
        raise SystemExit(f"version mismatch: expected {version}, found {detail}")
    return version


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the first captured group of a pattern, if any.

    Args:
        pattern: Compiled regular expression with one group.
        text: Text to search.

    Returns:
        The captured value or `None`.
    """
    match = pattern.search(text)
    return match.group(1) if match else None


def _payload() -> list[tuple[Path, str]]:
    """Collect the release files as `(source, archive name)` pairs.

    Returns:
        Pairs in deterministic order.
    """
    entries: list[tuple[Path, str]] = []
    for name in INCLUDE:
        source = ROOT / name
        if source.is_file():
            entries.append((source, f"{PACKAGE}/{name}"))
            continue
        if not source.is_dir():
            raise SystemExit(f"missing release file: {name}")
        for child in sorted(source.rglob("*")):
            relative = child.relative_to(ROOT).as_posix()
            if any(part in EXCLUDED_DIRS for part in child.parts):
                continue
            if child.suffix in EXCLUDED_SUFFIXES or not child.is_file():
                continue
            entries.append((child, f"{PACKAGE}/{relative}"))
    return entries


def _build(version: str, entries: list[tuple[Path, str]]) -> Path:
    """Write the archive and report its size.

    Args:
        version: Release version used in the file name.
        entries: Files to pack.

    Returns:
        The path of the archive.

    Raises:
        SystemExit: When the archive exceeds the market size limit.
    """
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"{PACKAGE}-v{version}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for source, name in entries:
            handle.write(source, arcname=name)
    size = archive.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        archive.unlink(missing_ok=True)
        raise SystemExit(f"archive is {size} bytes, over the 16 MiB market limit")
    return archive


def main() -> int:
    """Build the release archive.

    Returns:
        Process exit code.
    """
    version = _release_version()
    entries = _payload()
    archive = _build(version, entries)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"built {archive.relative_to(ROOT).as_posix()}")
    print(f"files: {len(entries)}  bytes: {archive.stat().st_size}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
