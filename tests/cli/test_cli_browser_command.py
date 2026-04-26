"""Regression tests for live Chrome browser slash-command wiring."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.agent = None
    cli_obj.conversation_history = []
    return cli_obj


def test_refresh_agent_tools_updates_active_agent_and_history():
    cli_obj = _make_cli()
    cli_obj.agent = SimpleNamespace(
        enabled_toolsets=["browser"],
        disabled_toolsets=None,
        _persist_session=MagicMock(),
    )
    tools = [
        {"function": {"name": "browser_list_tabs"}},
        {"function": {"name": "browser_switch_tab"}},
    ]

    with patch("cli.get_tool_definitions", return_value=tools) as defs_mock:
        cli_obj._refresh_agent_tools("[SYSTEM: browser tools updated]")

    defs_mock.assert_called_once_with(
        enabled_toolsets=["browser"],
        disabled_toolsets=None,
        quiet_mode=True,
    )
    assert cli_obj.agent.tools == tools
    assert cli_obj.agent.valid_tool_names == {"browser_list_tabs", "browser_switch_tab"}
    assert cli_obj.conversation_history[-1]["content"] == "[SYSTEM: browser tools updated]"
    cli_obj.agent._persist_session.assert_called_once_with(
        cli_obj.conversation_history,
        cli_obj.conversation_history,
    )


def test_browser_connect_refreshes_active_agent_tools(monkeypatch):
    cli_obj = _make_cli()
    fake_socket = MagicMock()
    fake_socket.connect.side_effect = OSError("closed")

    with patch("socket.socket", return_value=fake_socket), \
         patch("tools.browser_tool.cleanup_all_browsers"), \
         patch.object(cli_obj, "_refresh_agent_tools") as refresh_mock:
        cli_obj._handle_browser_command("/browser connect http://127.0.0.1:9333")

    assert os.environ.get("BROWSER_CDP_URL") == "http://127.0.0.1:9333"
    refresh_mock.assert_called_once()
    assert "browser_list_tabs" in refresh_mock.call_args.args[0]
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)


def test_browser_disconnect_refreshes_active_agent_tools(monkeypatch):
    cli_obj = _make_cli()
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")

    with patch("tools.browser_tool.cleanup_all_browsers"), \
         patch.object(cli_obj, "_refresh_agent_tools") as refresh_mock:
        cli_obj._handle_browser_command("/browser disconnect")

    assert os.environ.get("BROWSER_CDP_URL") is None
    refresh_mock.assert_called_once_with(
        "[SYSTEM: The user has disconnected the browser tools from their live Chrome. "
        "Browser tools are back to default mode (headless local browser or cloud provider).]"
    )
