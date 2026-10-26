"""Pytest bootstrap for a plugin that is usually developed inside AstrBot.

Two environment concerns are handled here, both before any test module imports
AstrBot:

1. ``jitrl`` is a pure-Python package that never imports AstrBot, so the plugin
   root is put on ``sys.path`` to let the tests import it directly.
2. ``test_integration.py`` does import AstrBot. AstrBot resolves its data
   directory from ``ASTRBOT_ROOT`` and otherwise falls back to the process
   working directory, which would litter this checkout with a ``data/`` tree.
   Pointing ``ASTRBOT_ROOT`` at a throwaway directory keeps the test run
   hermetic; an already exported value is respected so CI and developers can
   still override it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

if not os.environ.get("ASTRBOT_ROOT"):
    os.environ["ASTRBOT_ROOT"] = tempfile.mkdtemp(prefix="jitrl-astrbot-root-")
