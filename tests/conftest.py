"""Pytest bootstrap for a plugin that is usually developed inside AstrBot.

Four environment concerns are handled here, all before any test module imports
AstrBot:

1. ``jitrl`` is a pure-*Python* package: it imports no AstrBot runtime service,
   but it does take its logger from ``astrbot.api`` (AstrBot's plugin review
   requires every log line to come from the host logger). When AstrBot is not
   installed - plain CI, static analysis - a minimal stub keeps the package
   importable; ``test_integration.py`` detects that stub and skips.
2. The plugin root is put on ``sys.path`` to let the tests import ``jitrl``.
3. ``test_integration.py`` imports the plugin class the way AstrBot does. The
   host imports it as ``data.plugins.<dir>.main``, which only resolves when the
   checkout sits inside ``data/plugins/``. Registering the checkout as a
   top-level ``astrbot_plugin_jitrl`` package makes the same class importable
   from a plain clone as well, so the integration suite runs locally and in CI.
4. ``test_integration.py`` also imports AstrBot. AstrBot resolves its data
   directory from ``ASTRBOT_ROOT`` and otherwise falls back to the process
   working directory, which would litter this checkout with a ``data/`` tree.
   Pointing ``ASTRBOT_ROOT`` at a throwaway directory keeps the test run
   hermetic; an already exported value is respected so CI and developers can
   still override it.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

PLUGIN_PACKAGE = "astrbot_plugin_jitrl"
"""Import name the plugin is registered under for the integration tests."""


def _install_astrbot_stub() -> None:
    """Provide ``astrbot.api.logger`` when the real host is not installed.

    The stub is deliberately tiny - the logger is the only symbol the plugin
    needs at import time - and it is tagged with ``__jitrl_stub__`` so
    ``test_integration.py`` can tell it apart from a real AstrBot install and
    skip instead of failing on a missing submodule.
    """
    if importlib.util.find_spec("astrbot") is not None:
        return
    host = ModuleType("astrbot")
    host.__path__ = []  # type: ignore[attr-defined]  # no submodules on purpose
    host.__jitrl_stub__ = True  # type: ignore[attr-defined]
    api = ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot")
    host.api = api
    sys.modules["astrbot"] = host
    sys.modules["astrbot.api"] = api


_install_astrbot_stub()


def _register_plugin_package() -> None:
    """Expose the checkout as a package regardless of where it was cloned.

    The plugin's ``main.py`` uses relative imports (``from .jitrl...``), so it
    has to be imported as a submodule of a package. AstrBot provides that
    package implicitly through ``data.plugins``; outside an AstrBot checkout we
    build it from the plugin's own ``__init__.py`` instead.
    """
    if PLUGIN_PACKAGE in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PLUGIN_PACKAGE,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the {PLUGIN_PACKAGE} package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PLUGIN_PACKAGE] = module
    spec.loader.exec_module(module)


_register_plugin_package()

if not os.environ.get("ASTRBOT_ROOT"):
    os.environ["ASTRBOT_ROOT"] = tempfile.mkdtemp(prefix="jitrl-astrbot-root-")
