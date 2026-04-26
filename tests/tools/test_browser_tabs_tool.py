"""Tests for live-CDP browser tab helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

from model_tools import _LEGACY_TOOLSET_MAP
from tools.registry import registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS


def _cdp_targets_payload(*targets):
    return json.dumps(
        {
            "success": True,
            "method": "Target.getTargets",
            "result": {"targetInfos": list(targets)},
        }
    )


def _cdp_success_payload(method: str, result: dict | None = None):
    return json.dumps(
        {
            "success": True,
            "method": method,
            "result": result or {},
        }
    )


class TestBrowserListTabs:
    def test_filters_non_page_targets_and_numbers_tabs(self):
        from tools.browser_tabs_tool import browser_list_tabs

        payload = _cdp_targets_payload(
            {"targetId": "page-a", "type": "page", "title": "Inbox", "url": "https://mail.example"},
            {"targetId": "worker-a", "type": "service_worker", "title": "", "url": ""},
            {"targetId": "page-b", "type": "page", "title": "GitHub", "url": "https://github.com"},
        )

        with patch("tools.browser_cdp_tool.browser_cdp", return_value=payload) as mock_cdp:
            result = json.loads(browser_list_tabs())

        assert result["success"] is True
        assert result["count"] == 2
        assert [tab["index"] for tab in result["tabs"]] == [1, 2]
        assert [tab["tab_id"] for tab in result["tabs"]] == ["page-a", "page-b"]
        mock_cdp.assert_called_once_with(
            method="Target.getTargets",
            params={},
            target_id=None,
            timeout=30.0,
        )


class TestBrowserSwitchTab:
    def test_switches_by_index(self):
        from tools.browser_tabs_tool import browser_switch_tab

        with patch(
            "tools.browser_cdp_tool.browser_cdp",
            side_effect=[
                _cdp_targets_payload(
                    {"targetId": "page-a", "type": "page", "title": "Inbox", "url": "https://mail.example"},
                    {"targetId": "page-b", "type": "page", "title": "GitHub", "url": "https://github.com"},
                ),
                _cdp_success_payload("Target.activateTarget"),
                _cdp_success_payload("Page.bringToFront"),
            ],
        ) as mock_cdp:
            result = json.loads(browser_switch_tab(index=2))

        assert result["success"] is True
        assert result["tab"]["tab_id"] == "page-b"
        assert result["tab"]["index"] == 2
        assert mock_cdp.call_args_list[1].kwargs == {
            "method": "Target.activateTarget",
            "params": {"targetId": "page-b"},
            "target_id": None,
            "timeout": 30.0,
        }
        assert mock_cdp.call_args_list[2].kwargs == {
            "method": "Page.bringToFront",
            "params": None,
            "target_id": "page-b",
            "timeout": 30.0,
        }

    def test_switch_by_title_contains_returns_ambiguous_matches(self):
        from tools.browser_tabs_tool import browser_switch_tab

        with patch(
            "tools.browser_cdp_tool.browser_cdp",
            return_value=_cdp_targets_payload(
                {"targetId": "page-a", "type": "page", "title": "GitHub Notifications", "url": "https://github.com/notifications"},
                {"targetId": "page-b", "type": "page", "title": "GitHub Pull Requests", "url": "https://github.com/pulls"},
            ),
        ):
            result = json.loads(browser_switch_tab(title_contains="github"))

        assert result["success"] is False
        assert "Multiple tabs matched" in result["error"]
        assert len(result["matches"]) == 2

    def test_switch_requires_exactly_one_selector(self):
        from tools.browser_tabs_tool import browser_switch_tab

        with patch(
            "tools.browser_cdp_tool.browser_cdp",
            return_value=_cdp_targets_payload(
                {"targetId": "page-a", "type": "page", "title": "Inbox", "url": "https://mail.example"},
            ),
        ):
            result = json.loads(browser_switch_tab())

        assert result["success"] is False
        assert "exactly one selector" in result["error"]


class TestBrowserTabsRegistration:
    def test_registered_in_browser_toolset(self):
        for tool_name in ("browser_list_tabs", "browser_switch_tab"):
            entry = registry.get_entry(tool_name)
            assert entry is not None
            assert entry.toolset == "browser"

    def test_exposed_in_core_and_legacy_browser_toolsets(self):
        assert "browser_list_tabs" in TOOLSETS["browser"]["tools"]
        assert "browser_switch_tab" in TOOLSETS["browser"]["tools"]
        assert "browser_list_tabs" in _HERMES_CORE_TOOLS
        assert "browser_switch_tab" in _HERMES_CORE_TOOLS
        assert "browser_list_tabs" in _LEGACY_TOOLSET_MAP["browser_tools"]
        assert "browser_switch_tab" in _LEGACY_TOOLSET_MAP["browser_tools"]

    def test_gate_follows_browser_cdp_gate(self, monkeypatch):
        from tools.browser_tabs_tool import _browser_tabs_check

        monkeypatch.setattr("tools.browser_cdp_tool._browser_cdp_check", lambda: True)
        assert _browser_tabs_check() is True

        monkeypatch.setattr("tools.browser_cdp_tool._browser_cdp_check", lambda: False)
        assert _browser_tabs_check() is False
