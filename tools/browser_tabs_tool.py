#!/usr/bin/env python3
"""Live-Chrome tab discovery and switching helpers.

These tools are intentionally gated to the same live-CDP mode as
``browser_cdp``. They exist to make the agent practical against a user's
already-open, already-logged-in Chrome session without forcing the model to
manually speak raw DevTools protocol for common tab-management tasks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


def _browser_tabs_check() -> bool:
    """Expose tab tools only when live CDP mode is available."""
    try:
        from tools.browser_cdp_tool import _browser_cdp_check

        return bool(_browser_cdp_check())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("browser tab tools check failed: %s", exc)
        return False


def _call_browser_cdp(
    method: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    target_id: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Invoke the public browser_cdp tool and parse its JSON response."""
    try:
        from tools.browser_cdp_tool import browser_cdp
    except Exception as exc:  # pragma: no cover - defensive
        return {"success": False, "error": f"browser_cdp unavailable: {exc}"}

    raw = browser_cdp(
        method=method,
        params=params,
        target_id=target_id,
        timeout=timeout,
    )
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        return {
            "success": False,
            "error": f"browser_cdp returned non-JSON output for {method}",
            "raw": raw,
        }
    return parsed


def _normalize_tabs_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Target.getTargets output into a compact, page-tab list."""
    if not payload.get("success"):
        return payload

    target_infos = payload.get("result", {}).get("targetInfos", [])
    if not isinstance(target_infos, list):
        return {
            "success": False,
            "error": "Target.getTargets returned an invalid targetInfos payload",
        }

    tabs: List[Dict[str, Any]] = []
    for target in target_infos:
        if not isinstance(target, dict):
            continue
        if target.get("type") != "page":
            continue
        tabs.append(
            {
                "index": len(tabs) + 1,
                "tab_id": str(target.get("targetId") or ""),
                "title": str(target.get("title") or ""),
                "url": str(target.get("url") or ""),
                "attached": bool(target.get("attached", False)),
            }
        )

    return {
        "success": True,
        "tabs": tabs,
        "count": len(tabs),
    }


def browser_list_tabs(task_id: Optional[str] = None) -> str:
    """List open tabs in the connected live Chrome session."""
    del task_id  # accepted for uniformity with other browser tools

    payload = _call_browser_cdp("Target.getTargets", params={})
    normalized = _normalize_tabs_payload(payload)
    return json.dumps(normalized, ensure_ascii=False)


def _resolve_tab_selector(
    tabs: List[Dict[str, Any]],
    *,
    tab_id: Optional[str] = None,
    index: Optional[int] = None,
    title_contains: Optional[str] = None,
    url_contains: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one tab from a list using exactly one selector."""
    selectors = {
        "tab_id": tab_id,
        "index": index,
        "title_contains": title_contains,
        "url_contains": url_contains,
    }
    provided = [name for name, value in selectors.items() if value not in (None, "")]
    if not provided:
        return {
            "success": False,
            "error": "Provide exactly one selector: tab_id, index, title_contains, or url_contains.",
        }
    if len(provided) > 1:
        return {
            "success": False,
            "error": "Provide only one selector at a time: tab_id, index, title_contains, or url_contains.",
        }

    selector = provided[0]
    if selector == "index":
        if not isinstance(index, int) or index < 1:
            return {
                "success": False,
                "error": "index must be a positive integer starting at 1.",
            }
        if index > len(tabs):
            return {
                "success": False,
                "error": f"Tab index {index} is out of range; only {len(tabs)} tab(s) available.",
            }
        return {"success": True, "tab": tabs[index - 1]}

    if selector == "tab_id":
        for tab in tabs:
            if tab.get("tab_id") == tab_id:
                return {"success": True, "tab": tab}
        return {
            "success": False,
            "error": f"No tab found with tab_id={tab_id!r}.",
        }

    needle = str(title_contains if selector == "title_contains" else url_contains).strip().lower()
    field = "title" if selector == "title_contains" else "url"
    matches = [tab for tab in tabs if needle in str(tab.get(field) or "").lower()]

    if not matches:
        return {
            "success": False,
            "error": f"No tab matched {selector}={needle!r}.",
        }
    if len(matches) > 1:
        return {
            "success": False,
            "error": f"Multiple tabs matched {selector}={needle!r}; refine the selector or use tab_id/index.",
            "matches": matches,
        }
    return {"success": True, "tab": matches[0]}


def browser_switch_tab(
    tab_id: Optional[str] = None,
    index: Optional[int] = None,
    title_contains: Optional[str] = None,
    url_contains: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Activate one existing tab in the connected live Chrome session."""
    del task_id  # accepted for uniformity with other browser tools

    tabs_payload = _normalize_tabs_payload(_call_browser_cdp("Target.getTargets", params={}))
    if not tabs_payload.get("success"):
        return json.dumps(tabs_payload, ensure_ascii=False)

    tabs = tabs_payload.get("tabs", [])
    resolved = _resolve_tab_selector(
        tabs,
        tab_id=tab_id,
        index=index,
        title_contains=title_contains,
        url_contains=url_contains,
    )
    if not resolved.get("success"):
        return json.dumps(resolved, ensure_ascii=False)

    tab = resolved["tab"]
    activate = _call_browser_cdp(
        "Target.activateTarget",
        params={"targetId": tab["tab_id"]},
    )
    if not activate.get("success"):
        return json.dumps(activate, ensure_ascii=False)

    # Best effort: some Chrome builds respond more predictably when the page
    # target is also asked to bring itself to the front after activation.
    bring_to_front = _call_browser_cdp(
        "Page.bringToFront",
        target_id=tab["tab_id"],
    )

    response = {
        "success": True,
        "tab": tab,
        "note": "Tab activated. Use browser_snapshot or other browser tools against the now-active tab.",
    }
    if not bring_to_front.get("success"):
        response["warning"] = (
            "Tab was activated via Target.activateTarget, but Page.bringToFront "
            "did not confirm successfully."
        )
        response["bring_to_front_error"] = bring_to_front.get("error")

    return json.dumps(response, ensure_ascii=False)


BROWSER_TAB_SCHEMAS = [
    {
        "name": "browser_list_tabs",
        "description": (
            "List open tabs in the connected live Chrome session. Use this before navigating "
            "when the user may already have a relevant logged-in tab open. Available only "
            "when Hermes is attached to a live Chrome via CDP."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "browser_switch_tab",
        "description": (
            "Switch to one existing tab in the connected live Chrome session. Select a tab by "
            "index from browser_list_tabs, by exact tab_id, or by a title/url substring. "
            "Available only when Hermes is attached to a live Chrome via CDP."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Exact tab_id from browser_list_tabs.",
                },
                "index": {
                    "type": "integer",
                    "description": "1-based tab index from browser_list_tabs.",
                },
                "title_contains": {
                    "type": "string",
                    "description": "Case-insensitive substring match against the tab title.",
                },
                "url_contains": {
                    "type": "string",
                    "description": "Case-insensitive substring match against the tab URL.",
                },
            },
            "required": [],
        },
    },
]


_BROWSER_TAB_SCHEMA_MAP = {schema["name"]: schema for schema in BROWSER_TAB_SCHEMAS}

registry.register(
    name="browser_list_tabs",
    toolset="browser",
    schema=_BROWSER_TAB_SCHEMA_MAP["browser_list_tabs"],
    handler=lambda args, **kw: browser_list_tabs(task_id=kw.get("task_id")),
    check_fn=_browser_tabs_check,
    emoji="🗂️",
)

registry.register(
    name="browser_switch_tab",
    toolset="browser",
    schema=_BROWSER_TAB_SCHEMA_MAP["browser_switch_tab"],
    handler=lambda args, **kw: browser_switch_tab(
        tab_id=args.get("tab_id"),
        index=args.get("index"),
        title_contains=args.get("title_contains"),
        url_contains=args.get("url_contains"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_browser_tabs_check,
    emoji="📑",
)
