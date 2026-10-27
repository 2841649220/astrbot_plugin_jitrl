"""Static checks on the WebUI view (views/dashboard).

The view runs inside AstrBot's plugin iframe, which the dashboard creates with
``sandbox="allow-scripts allow-forms allow-downloads"``. That sandbox has no
``allow-modals``, so ``window.alert`` / ``confirm`` / ``prompt`` are silently
suppressed and ``confirm()`` returns ``false``. A destructive action guarded by
a modal therefore does nothing at all, with no error to debug from — which is
exactly the bug these checks exist to prevent from coming back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

VIEW_DIR = Path(__file__).resolve().parents[1] / "views" / "dashboard"
APP_JS = VIEW_DIR / "app.js"
INDEX_HTML = VIEW_DIR / "index.html"
STYLE_CSS = VIEW_DIR / "style.css"

MODAL_CALLS = ("window.alert", "window.confirm", "window.prompt")
"""Browser modal APIs that a sandboxed plugin view may not rely on."""


def _strip_comments(source: str) -> str:
    """Remove JavaScript comments before scanning for forbidden calls.

    Documentation legitimately mentions the APIs the view must not call, so the
    scan has to look at executable code only.

    Args:
        source: JavaScript source text.

    Returns:
        The source with block comments and whole-line comments removed.
    """
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    kept = [
        line for line in without_block.splitlines() if not line.strip().startswith("//")
    ]
    return "\n".join(kept)


def _read(path: Path) -> str:
    """Read a view asset as UTF-8 text.

    Args:
        path: Asset path.

    Returns:
        The file contents.

    Raises:
        AssertionError: When the asset is missing, so a moved file cannot
            silently disable these checks.
    """
    assert path.is_file(), f"missing view asset: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("call", MODAL_CALLS)
def test_view_does_not_use_modal_dialogs(call: str) -> None:
    """Modal dialogs are suppressed inside AstrBot's sandboxed iframe.

    Args:
        call: The browser modal API to look for.
    """
    code = _strip_comments(_read(APP_JS))
    bare = call.split(".")[1]
    assert call + "(" not in code and bare + "(" not in code, (
        f"{bare}() is suppressed in the plugin iframe sandbox (no allow-modals) "
        "and confirm() returns false, so the guarded action never runs. "
        "Use an in-page two-step confirmation instead."
    )
    assert f"window.{bare}" not in code


def test_forget_action_uses_an_in_page_confirmation() -> None:
    """The destructive action must arm itself instead of opening a modal."""
    js = _read(APP_JS)
    assert "FORGET_ARM_MS" in js
    assert 'classList.add("btn-armed")' in js
    assert "setTimeout(disarmForget" in js
    assert "再次点击确认清空" in js
    css = _read(STYLE_CSS)
    assert ".btn-armed" in css, "the armed state needs a visible style"


def test_export_uses_the_bridge_download_signature() -> None:
    """bridge.download is (endpoint, params, filename); params must be an object."""
    js = _read(APP_JS)
    match = re.search(r"bridge\.download\(([^)]*)\)", js)
    assert match is not None, "the view no longer calls bridge.download"
    args = [part.strip() for part in match.group(1).split(",")]
    assert len(args) == 3, (
        "bridge.download takes (endpoint, params, filename); passing a string as "
        "params drops the download filename"
    )
    assert args[1].startswith("{") and "robot:" in args[1], (
        "export must include the selected robot ID in its query parameters"
    )
    assert args[2].startswith('"') and len(args[2]) > 2, "filename is missing"


def test_every_element_lookup_exists_in_the_markup() -> None:
    """A typo in an element id would throw on load and blank the whole page."""
    js = _read(APP_JS)
    html = _read(INDEX_HTML)
    ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    assert ids, "app.js looks up no elements, which cannot be right"
    declared = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(ids - declared)
    assert not missing, (
        f"app.js looks up ids that index.html does not define: {missing}"
    )


def test_view_assets_are_served_by_the_dashboard_contract() -> None:
    """AstrBot discovers views as views/<name>/index.html and serves siblings."""
    assert INDEX_HTML.is_file()
    assert APP_JS.is_file()
    assert STYLE_CSS.is_file()
    html = _read(INDEX_HTML)
    assert 'src="./app.js"' in html
    assert 'href="./style.css"' in html
    assert 'type="module"' in html, "the bridge is an ES module; load app.js as one"
