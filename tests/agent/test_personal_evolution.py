import json
from unittest.mock import Mock, patch

from agent.personal_evolution import (
    EvolutionConfig,
    PersonalEvolutionStore,
    STRATEGY_RATIOS,
    _fit_payload_json_to_prompt,
    adapt_gene_from_learning,
    build_learning_signals,
    classify_failure_mode,
    compute_drift_intensity,
    distill_evolution_mutations,
    extract_signals,
    handle_evolution_command,
)


def test_extracts_identity_and_execution_signals():
    signals = extract_signals(
        "let's go full aggressive and build my digital identity world model",
        "Implemented the evolution layer.",
        ["terminal", "apply_patch"],
    )

    kinds = {s["kind"] for s in signals}
    assert "operating_style" in kinds
    assert "identity_model" in kinds
    assert "tool_pattern" in kinds


def test_store_creates_reinforces_and_selects_genes(tmp_path):
    store = PersonalEvolutionStore(
        EvolutionConfig(enabled=True, distillation_enabled=False, heuristic_fallback=True, max_active_genes=4),
        base_dir=tmp_path / "evolution",
    )

    first = store.evolve_turn(
        user_message="my goal is a true personal assistant that acts on my behalf",
        assistant_response="I will build the personal evolution layer.",
        tool_names=["terminal"],
        session_id="s1",
        platform="cli",
    )
    second = store.evolve_turn(
        user_message="go ahead and implement the personal assistant world model",
        assistant_response="Implemented it.",
        tool_names=["terminal"],
        session_id="s1",
        platform="cli",
    )

    assert first["changed"] is True
    assert second["changed"] is True
    assert store.genes
    assert (tmp_path / "evolution" / "personal_genes.json").exists()
    assert (tmp_path / "evolution" / "capsules.json").exists()
    assert (tmp_path / "evolution" / "evolution_events.jsonl").exists()
    assert (tmp_path / "evolution" / "evolution_narrative.md").exists()
    assert (tmp_path / "evolution" / "validation_reports.jsonl").exists()

    context = store.select_context("personal assistant should act on my behalf")
    assert "Personal Evolution Genes" in context
    assert "Strategy intent ratios" in context
    assert "identity" in context or "assistant" in context

    payload = json.loads((tmp_path / "evolution" / "personal_genes.json").read_text())
    assert payload["version"] == 1
    assert payload["genes"]
    assert payload["genes"][0]["type"] == "Gene"
    assert "category" in payload["genes"][0]
    assert "signals_match" in payload["genes"][0]
    assert "strategy" in payload["genes"][0]
    capsule_payload = json.loads((tmp_path / "evolution" / "capsules.json").read_text())
    assert capsule_payload["capsules"]
    assert capsule_payload["capsules"][0]["type"] == "Capsule"


def test_evolution_command_lists_inspects_and_archives(tmp_path):
    store = PersonalEvolutionStore(
        EvolutionConfig(enabled=True, distillation_enabled=False, heuristic_fallback=True),
        base_dir=tmp_path / "evolution",
    )
    store.evolve_turn(
        user_message="I prefer aggressive implementation for personal assistant work",
        assistant_response="Noted.",
        tool_names=[],
    )

    gene_id = store.genes[0]["id"]
    listing = handle_evolution_command("/evolution", store=store)
    assert "Personal Evolution Genes" in listing
    assert gene_id[:12] in listing

    inspected = handle_evolution_command(f"/evolution inspect {gene_id[:10]}", store=store)
    assert "Signals Match:" in inspected
    assert "Strategy:" in inspected

    archived = handle_evolution_command(f"/evolution archive {gene_id[:10]}", store=store)
    assert "Archived gene" in archived
    assert store.genes[0]["status"] == "archived"

    listing_after_archive = handle_evolution_command("/evolution list", store=store)
    assert gene_id[:12] not in listing_after_archive

    capsules = handle_evolution_command("/evolution capsules", store=store)
    assert "Evolution Capsules" in capsules

    narrative = handle_evolution_command("/evolution narrative", store=store)
    assert "Evolution Narrative" in narrative


def test_evolution_command_reset_requires_confirmation(tmp_path):
    store = PersonalEvolutionStore(
        EvolutionConfig(enabled=True, distillation_enabled=False, heuristic_fallback=True),
        base_dir=tmp_path / "evolution",
    )
    store.evolve_turn(
        user_message="my goal is a true personal assistant",
        assistant_response="Stored.",
    )

    denied = handle_evolution_command("/evolution reset", store=store)
    assert "reset confirm" in denied
    assert store.genes

    reset = handle_evolution_command("/evolution reset confirm", store=store)
    assert reset == "Personal evolution store reset."
    assert store.genes == []


def test_distiller_parses_llm_mutations():
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = json.dumps({
        "mutations": [{
            "action": "create",
            "category": "innovate",
            "key": "personal_assistant_identity",
            "summary": "Maintain a world model for the user.",
            "signals_match": ["personal", "assistant"],
            "strategy": ["Maintain a world model for the user."],
            "validation": ["Uses the world model in future assistant decisions."],
            "confidence": 0.9,
            "importance": 0.95,
            "rationale": "Explicit long-term goal.",
        }]
    })

    with patch("agent.auxiliary_client.call_llm", return_value=response):
        mutations = distill_evolution_mutations(
            user_message="I want Hermes to be my personal assistant",
            assistant_response="Understood.",
        )

    assert mutations == [{
        "action": "create",
        "gene_id": "",
        "category": "innovate",
        "key": "personal_assistant_identity",
        "summary": "Maintain a world model for the user.",
        "signals_match": ["personal", "assistant"],
        "strategy": ["Maintain a world model for the user."],
        "validation": ["Uses the world model in future assistant decisions."],
        "confidence": 0.9,
        "importance": 0.95,
        "rationale": "Explicit long-term goal.",
    }]


def test_distiller_payload_uses_prompt_level_budget():
    prompt_prefix = "Turn payload:\n"
    payload = {
        "user_message": "u" * 12000,
        "assistant_response": "a" * 12000,
        "tool_names": ["terminal"],
        "current_genes": [],
        "evolve_strategy": "balanced",
    }

    full = _fit_payload_json_to_prompt(payload, prompt_prefix=prompt_prefix, max_chars=24000)
    assert len(prompt_prefix) + len(full) <= 24000
    assert len(json.loads(full)["user_message"]) > 4000
    assert len(json.loads(full)["assistant_response"]) > 4000


def test_store_uses_distiller_before_heuristic_fallback(tmp_path):
    store = PersonalEvolutionStore(
        EvolutionConfig(enabled=True, distillation_enabled=True),
        base_dir=tmp_path / "evolution",
    )

    with patch("agent.personal_evolution.distill_evolution_mutations", return_value=[{
        "action": "create",
        "category": "innovate",
        "key": "project_context",
        "summary": "Track project context as durable user world-model state.",
        "signals_match": ["project", "context"],
        "strategy": ["Track project context as durable user world-model state."],
        "validation": ["Uses project context on later turns."],
        "confidence": 0.8,
        "importance": 0.9,
        "rationale": "The turn indicates persistent project state.",
    }]):
        result = store.evolve_turn(
            user_message="random wording without heuristic triggers",
            assistant_response="Tracked.",
        )

    assert result["changed"] is True
    assert store.genes[0]["type"] == "Gene"
    assert store.genes[0]["category"] == "innovate"
    assert store.genes[0]["summary"] == "Track project context as durable user world-model state."


def test_compute_drift_intensity_matches_evolver_shape():
    assert compute_drift_intensity(drift_enabled=False, gene_pool_size=10, memory_evidence=0) == min(1, 1 / (10 ** 0.5))
    assert compute_drift_intensity(drift_enabled=True, gene_pool_size=10, memory_evidence=0) == min(1, 1 / (10 ** 0.5) + 0.3)
    mature = compute_drift_intensity(drift_enabled=True, gene_pool_size=10, memory_evidence=100)
    assert round(mature, 6) == round(min(1, 1 / (10 ** 0.5) + 0.02), 6)


def test_strategy_ratios_match_evolver_presets():
    assert STRATEGY_RATIOS["balanced"] == {"innovate": 0.50, "optimize": 0.30, "repair": 0.20}
    assert STRATEGY_RATIOS["innovate"] == {"innovate": 0.80, "optimize": 0.15, "repair": 0.05}
    assert STRATEGY_RATIOS["harden"] == {"innovate": 0.20, "optimize": 0.40, "repair": 0.40}
    assert STRATEGY_RATIOS["repair-only"] == {"innovate": 0.00, "optimize": 0.20, "repair": 0.80}


def test_adapt_gene_learning_success_and_failure():
    gene = {
        "type": "Gene",
        "id": "gene_test",
        "signals_match": ["error"],
    }
    adapt_gene_from_learning(
        gene=gene,
        outcome_status="success",
        learning_signals=["problem:performance", "action:optimize", "area:orchestration"],
        failure_mode={"mode": "none", "reasonClass": None, "retryable": False},
    )
    assert "problem:performance" in gene["signals_match"]
    assert "area:orchestration" in gene["signals_match"]
    assert "action:optimize" not in gene["signals_match"]
    assert gene["learning_history"][0]["outcome"] == "success"

    adapt_gene_from_learning(
        gene=gene,
        outcome_status="failed",
        learning_signals=["problem:protocol", "risk:validation"],
        failure_mode=classify_failure_mode(validation_failed=True),
    )
    assert gene["anti_patterns"][0]["mode"] == "soft"
    assert gene["anti_patterns"][0]["reason_class"] == "validation"


def test_build_learning_signals_extracts_structured_tags():
    tags = build_learning_signals(
        signals=["perf_bottleneck"],
        category="optimize",
        summary="Validation failed because latency remained high",
    )
    assert "intent:optimize" in tags
    assert "problem:performance" in tags
    assert "risk:validation" in tags
