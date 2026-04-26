"""Tests for the Twin memory provider."""

from __future__ import annotations

import json

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.twin import TwinMemoryProvider


def _provider(tmp_path, config=None) -> TwinMemoryProvider:
    provider = TwinMemoryProvider(config=config)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    provider.initialize("session-1", hermes_home=str(hermes_home), agent_context="primary")
    return provider


def _call(provider: TwinMemoryProvider, tool_name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(tool_name, args))


def test_twin_provider_is_discoverable():
    providers = {name for name, _desc, _available in discover_memory_providers()}
    assert "twin" in providers
    loaded = load_memory_provider("twin")
    assert isinstance(loaded, TwinMemoryProvider)


def test_constitution_write_requires_confirmation(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "constitution",
            "key": "red_line.commitments",
            "value": "Do not make commitments without approval.",
            "source": "explicit_user_statement",
        },
    )
    assert result["success"] is False
    assert "confirmed=true" in result["error"]

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "constitution",
            "key": "red_line.commitments",
            "value": "Do not make commitments without approval.",
            "source": "explicit_user_statement",
            "confirmed": True,
            "confidence": 0.98,
        },
    )
    assert result["success"] is True

    brief = provider.prefetch("send a message")
    assert "red_line.commitments" in brief
    assert "Do not make commitments without approval." in brief


def test_simulation_output_cannot_write_core_identity(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "profile",
            "key": "communication.directness",
            "value": "very high",
            "source": "simulation_hypothesis",
            "confidence": 0.9,
        },
    )

    assert result["success"] is False
    assert "Simulation output cannot write directly to durable profile" in result["error"]


def test_state_records_expire_from_prefetch(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "state",
            "key": "current.priority",
            "value": "Finish the twin provider.",
            "source": "explicit_user_statement",
            "confirmed": True,
            "expires_hours": -1,
        },
    )
    assert result["success"] is True

    brief = provider.prefetch("current priorities")
    assert "Finish the twin provider." not in brief


def test_correction_demotes_active_record(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "profile",
            "key": "communication.style",
            "value": "prefers long explanations",
            "source": "explicit_user_statement",
            "confirmed": True,
            "confidence": 0.9,
        },
    )
    assert result["success"] is True

    correction = _call(
        provider,
        "twin_feedback",
        {
            "kind": "correction",
            "content": "Actually prefers concise responses.",
            "target_scope": "profile",
            "target_key": "communication.style",
        },
    )
    assert correction["success"] is True
    assert correction["updated"] == 1

    brief = provider.prefetch("communication")
    assert "prefers long explanations" not in brief


def test_policy_defaults_simulate_external_messages(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_policy",
        {
            "action_class": "external_message",
            "proposed_action": "Send a negotiation message.",
            "stakes": "medium",
        },
    )

    assert result["success"] is True
    assert result["decision"] == "simulate"
    assert result["requires_simulation"] is True
    assert result["requires_user_approval"] is True


def test_action_policy_override_requires_confirmation(tmp_path):
    provider = _provider(tmp_path)

    denied = _call(
        provider,
        "twin_note",
        {
            "scope": "action_policy",
            "key": "external_message",
            "value": "allow",
            "source": "explicit_user_statement",
        },
    )
    assert denied["success"] is False
    assert "confirmed=true" in denied["error"]

    accepted = _call(
        provider,
        "twin_note",
        {
            "scope": "action_policy",
            "key": "external_message",
            "value": "ask",
            "source": "explicit_user_statement",
            "confirmed": True,
            "reason": "User wants all sends approved.",
        },
    )
    assert accepted["success"] is True

    policy = _call(provider, "twin_policy", {"action_class": "external_message"})
    assert policy["decision"] == "ask"
    assert policy["requires_simulation"] is False
    assert policy["requires_user_approval"] is True


def test_entity_belief_stores_world_model_hypotheses(tmp_path):
    provider = _provider(tmp_path)

    result = _call(
        provider,
        "twin_note",
        {
            "scope": "entity_belief",
            "entity": "Project Atlas",
            "entity_type": "project",
            "key": "risk",
            "value": "Stakeholders may react poorly to abrupt timeline changes.",
            "source": "simulation_hypothesis",
            "confidence": 0.52,
        },
    )
    assert result["success"] is True

    entities = _call(provider, "twin_entities", {"query": "Atlas"})
    assert entities["success"] is True
    assert entities["entities"][0]["name"] == "Project Atlas"
    assert entities["entities"][0]["beliefs"][0]["source"] == "simulation_hypothesis"
