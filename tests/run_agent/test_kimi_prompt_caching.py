import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent
import run_agent


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_kimi_agent(monkeypatch, *, session_id="kimi-session"):
    _patch_agent_bootstrap(monkeypatch)
    agent = AIAgent(
        model="kimi-k2.6",
        provider="kimi-coding",
        base_url="https://api.moonshot.ai/v1",
        api_key="moonshot-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    return agent


def test_prompt_cache_key_policy_detects_kimi_hosts():
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "custom"
    agent.base_url = "https://api.moonshot.ai/v1"
    agent.api_mode = "chat_completions"

    assert agent._prompt_cache_key_policy() is True
    assert agent._prompt_cache_key_policy(base_url="https://api.fireworks.ai/inference/v1") is False


def test_build_api_kwargs_kimi_includes_prompt_cache_key_in_extra_body(monkeypatch):
    agent = _build_kimi_agent(monkeypatch, session_id="kimi-session-123")

    kwargs = agent._build_api_kwargs(
        [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Ping"},
        ]
    )

    assert kwargs["model"] == "kimi-k2.6"
    assert kwargs["extra_body"]["prompt_cache_key"] == "kimi-session-123"
    assert "prompt_cache_key" not in kwargs


def test_build_api_kwargs_kimi_omits_prompt_cache_key_without_session_id(monkeypatch):
    agent = _build_kimi_agent(monkeypatch, session_id=None)
    agent.session_id = None

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Ping"}])

    assert kwargs.get("extra_body", {}).get("prompt_cache_key") is None
