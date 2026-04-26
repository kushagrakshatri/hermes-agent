"""Personal evolution layer for Hermes.

This module mirrors Evolver's GEP-style asset model for Hermes' personal agent:
select signal-matching Genes, apply structured mutations, and inject selected
Genes into the next model call as ephemeral context.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOPWORDS = {
    "about", "after", "again", "agent", "also", "because", "before", "being",
    "could", "from", "have", "hermes", "into", "just", "like", "make",
    "more", "need", "should", "that", "there", "this", "through", "using",
    "want", "were", "what", "when", "where", "which", "with", "would",
    "your", "you", "the", "and", "for", "are", "but", "not",
}

EVOLVER_PROMPT_MAX_CHARS = 24000
EVOLUTION_DISTILLATION_MAX_TOKENS = 4000


@dataclass(frozen=True)
class EvolutionConfig:
    enabled: bool = True
    strategy: str = "balanced"
    distillation_enabled: bool = True
    heuristic_fallback: bool = False
    max_active_genes: int = 12
    max_gene_chars: int = 9000
    min_signal_score: int = 1
    decay: float = 0.985
    drift_enabled: bool = True
    narrative_enabled: bool = True


def load_config(raw: dict[str, Any] | None) -> EvolutionConfig:
    raw = raw or {}
    strategy = str(os.getenv("EVOLVE_STRATEGY") or raw.get("strategy") or "balanced").strip().lower()
    if strategy not in {"balanced", "innovate", "harden", "repair-only"}:
        strategy = "balanced"
    return EvolutionConfig(
        enabled=_as_bool(raw.get("enabled", True)),
        strategy=strategy,
        distillation_enabled=_as_bool(raw.get("distillation_enabled", True)),
        heuristic_fallback=_as_bool(raw.get("heuristic_fallback", False)),
        max_active_genes=max(1, int(raw.get("max_active_genes", 12))),
        max_gene_chars=max(1000, int(raw.get("max_gene_chars", 9000))),
        min_signal_score=max(0, int(raw.get("min_signal_score", 1))),
        decay=float(raw.get("decay", 0.985)),
        drift_enabled=_as_bool(raw.get("drift_enabled", True)),
        narrative_enabled=_as_bool(raw.get("narrative_enabled", True)),
    )


STRATEGY_RATIOS = {
    "balanced": {"innovate": 0.50, "optimize": 0.30, "repair": 0.20},
    "innovate": {"innovate": 0.80, "optimize": 0.15, "repair": 0.05},
    "harden": {"innovate": 0.20, "optimize": 0.40, "repair": 0.40},
    "repair-only": {"innovate": 0.00, "optimize": 0.20, "repair": 0.80},
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "always"}
    return bool(value)


class PersonalEvolutionStore:
    """Persistent gene and event store scoped to HERMES_HOME."""

    def __init__(self, config: EvolutionConfig | None = None, base_dir: Path | None = None):
        self.config = config or EvolutionConfig()
        self.base_dir = base_dir or get_hermes_home() / "evolution"
        self.genes_path = self.base_dir / "personal_genes.json"
        self.capsules_path = self.base_dir / "capsules.json"
        self.events_path = self.base_dir / "evolution_events.jsonl"
        self.narrative_path = self.base_dir / "evolution_narrative.md"
        self.validation_reports_path = self.base_dir / "validation_reports.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.genes: list[dict[str, Any]] = []
        self.capsules: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            if self.genes_path.exists():
                data = json.loads(self.genes_path.read_text(encoding="utf-8"))
                genes = data.get("genes", []) if isinstance(data, dict) else []
                self.genes = [g for g in genes if isinstance(g, dict)]
            if self.capsules_path.exists():
                data = json.loads(self.capsules_path.read_text(encoding="utf-8"))
                capsules = data.get("capsules", []) if isinstance(data, dict) else []
                self.capsules = [c for c in capsules if isinstance(c, dict)]
        except Exception as exc:
            logger.warning("Failed to load personal evolution genes: %s", exc)
            self.genes = []
            self.capsules = []

    def select_context(self, query: str, *, max_genes: int | None = None) -> str:
        if not self.config.enabled or not self.genes:
            return ""
        query_terms = set(_keywords(query))
        scored: list[tuple[float, dict[str, Any]]] = []
        now = time.time()
        memory_evidence = sum(int(g.get("evidence_count", 0)) for g in self.genes)
        drift = compute_drift_intensity(
            drift_enabled=self.config.drift_enabled,
            gene_pool_size=len(self.genes),
            memory_evidence=memory_evidence,
        )
        for gene in self.genes:
            if gene.get("status", "active") != "active":
                continue
            signals = set(_gene_signals(gene))
            overlap = _signal_overlap(query_terms, signals)
            base = float(gene.get("score", 0))
            recency = max(0.0, 1.0 - ((now - float(gene.get("updated_at", now))) / (86400 * 30)))
            category_weight = _strategy_category_weight(self.config.strategy, _gene_category(gene))
            anti_pattern_penalty = min(0.75, 0.15 * len(gene.get("anti_patterns", []) or []))
            score = (base + overlap * 3 + recency + drift) * category_weight - anti_pattern_penalty
            if score >= self.config.min_signal_score:
                scored.append((score, gene))
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [g for _, g in scored[: max_genes or self.config.max_active_genes]]
        selected_capsules = select_capsules(self.capsules, query_terms, selected)
        lines = [
            "Personal Evolution Genes selected by GEP signal matching.",
            f"EVOLVE_STRATEGY={self.config.strategy}. Treat selected Genes as high-priority operating context unless the current user message explicitly contradicts them.",
            f"Strategy intent ratios: {json.dumps(STRATEGY_RATIOS.get(self.config.strategy, STRATEGY_RATIOS['balanced']), sort_keys=True)}",
        ]
        used = 0
        for gene in selected:
            strategy_steps = _gene_strategy(gene)
            rule = " ".join(strategy_steps) if strategy_steps else str(gene.get("rule", ""))
            line = (
                f"- [{_gene_category(gene)}] {rule.strip()} "
                f"(signals_match: {', '.join(_gene_signals(gene)[:8])}; "
                f"score: {float(gene.get('score', 0)):.1f})"
            )
            if used + len(line) > self.config.max_gene_chars:
                break
            lines.append(line)
            used += len(line)
        if selected_capsules:
            lines.append("Reusable Capsules:")
            for capsule in selected_capsules[:5]:
                line = (
                    f"- [{capsule.get('gene', '')}] {str(capsule.get('summary', '')).strip()} "
                    f"(trigger: {', '.join(_clean_signals(capsule.get('trigger', []))[:8])}; "
                    f"confidence: {float(capsule.get('confidence', 0)):.2f})"
                )
                if used + len(line) > self.config.max_gene_chars:
                    break
                lines.append(line)
                used += len(line)
        return "\n".join(lines)

    def evolve_turn(
        self,
        *,
        user_message: str,
        assistant_response: str,
        tool_names: Iterable[str] = (),
        session_id: str | None = None,
        platform: str | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"changed": False, "events": []}
        mutations = []
        distiller_error = None
        if self.config.distillation_enabled:
            try:
                mutations = distill_evolution_mutations(
                    user_message=user_message,
                    assistant_response=assistant_response,
                    tool_names=tool_names,
                    current_genes=self.genes,
                    strategy=self.config.strategy,
                )
            except Exception as exc:
                distiller_error = str(exc)
                logger.debug("Personal evolution distillation failed: %s", exc)
        if not mutations and self.config.heuristic_fallback:
            mutations = _signals_to_mutations(
                extract_signals(user_message, assistant_response, tool_names)
            )
        if not mutations:
            event = self._distillation_audit_event(
                user_message=user_message,
                assistant_response=assistant_response,
                tool_names=tool_names,
                session_id=session_id,
                platform=platform,
                status="error" if distiller_error else "empty",
                reason=distiller_error or "Distiller returned no mutations.",
                now=time.time(),
            )
            self._save([event])
            return {"changed": False, "events": [event], "distiller_error": distiller_error}
        now = time.time()
        events: list[dict[str, Any]] = []
        changed = False
        for mutation in mutations:
            event = self._apply_mutation(
                mutation,
                now=now,
                session_id=session_id,
                platform=platform,
                distiller_error=distiller_error,
            )
            if event:
                events.append(event)
                changed = True
        if changed:
            self._prune()
            self._save(events)
        return {"changed": changed, "events": events}

    def _distillation_audit_event(
        self,
        *,
        user_message: str,
        assistant_response: str,
        tool_names: Iterable[str],
        session_id: str | None,
        platform: str | None,
        status: str,
        reason: str,
        now: float,
    ) -> dict[str, Any]:
        return {
            "id": "evt_" + uuid.uuid4().hex[:12],
            "schema_version": "1.6.0",
            "type": "EvolutionEvent",
            "action": "distillation_" + status,
            "intent": "audit",
            "category": "audit",
            "signals": _keywords((user_message or "") + " " + (assistant_response or ""))[:24],
            "genes_used": [],
            "mutation": {
                "summary": "No Gene mutation was applied for this turn.",
                "reason": reason,
            },
            "outcome": {"status": status, "score": 0.0},
            "tool_names": list(dict.fromkeys(str(t) for t in tool_names if t))[:20],
            "session_id": session_id,
            "platform": platform,
            "created_at": now,
        }

    def _find_gene(self, category: str, key: str) -> dict[str, Any] | None:
        for gene in self.genes:
            if _gene_category(gene) == category and gene.get("key") == key:
                return gene
        return None

    def _prune(self) -> None:
        self.genes.sort(
            key=lambda g: (g.get("status") == "active", float(g.get("score", 0)), float(g.get("updated_at", 0))),
            reverse=True,
        )
        self.genes = self.genes[:200]

    def _save(self, events: list[dict[str, Any]]) -> None:
        if not any(event.get("action") == "reset" for event in events):
            self._merge_existing_assets()
        payload = {"version": 1, "updated_at": time.time(), "genes": self.genes}
        atomic_json_write(self.genes_path, payload, indent=2)
        capsule_payload = {"version": 1, "updated_at": time.time(), "capsules": self.capsules}
        atomic_json_write(self.capsules_path, capsule_payload, indent=2)
        with self.events_path.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        if self.config.narrative_enabled and events:
            self._append_narrative(events)

    def _merge_existing_assets(self) -> None:
        existing_genes = _read_asset_list(self.genes_path, "genes")
        existing_capsules = _read_asset_list(self.capsules_path, "capsules")
        if existing_genes:
            merged = {str(g.get("id")): g for g in existing_genes if g.get("id")}
            for gene in self.genes:
                if gene.get("id"):
                    merged[str(gene.get("id"))] = gene
            self.genes = list(merged.values())
        if existing_capsules:
            merged_capsules = {str(c.get("id")): c for c in existing_capsules if c.get("id")}
            for capsule in self.capsules:
                if capsule.get("id"):
                    merged_capsules[str(capsule.get("id"))] = capsule
            self.capsules = list(merged_capsules.values())

    def _append_narrative(self, events: list[dict[str, Any]]) -> None:
        with self.narrative_path.open("a", encoding="utf-8") as fh:
            for event in events:
                ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(event.get("created_at", time.time()))))
                mutation = event.get("mutation") if isinstance(event.get("mutation"), dict) else {}
                outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
                fh.write(
                    f"- {ts} `{event.get('action', event.get('type', 'event'))}` "
                    f"{event.get('category', '')} gene={event.get('gene_id', '')} "
                    f"score={outcome.get('score', '')} capsule={event.get('capsule_id', '')}: "
                    f"{mutation.get('summary') or event.get('reason', '')}\n"
                )

    def archive_gene(self, gene_id: str, *, reason: str = "Archived by user") -> bool:
        now = time.time()
        for gene in self.genes:
            if _matches_gene(gene, gene_id):
                gene["status"] = "archived"
                gene["archived_reason"] = reason
                gene["updated_at"] = now
                self._save([{
                    "id": "evt_" + uuid.uuid4().hex[:12],
                    "type": "EvolutionEvent",
                    "action": "archive",
                    "gene_id": gene.get("id"),
                    "category": _gene_category(gene),
                    "reason": reason,
                    "created_at": now,
                }])
                return True
        return False

    def reset(self) -> None:
        now = time.time()
        previous_count = len(self.genes)
        previous_capsule_count = len(self.capsules)
        self.genes = []
        self.capsules = []
        self._save([{
            "id": "evt_" + uuid.uuid4().hex[:12],
            "type": "EvolutionEvent",
            "action": "reset",
            "previous_gene_count": previous_count,
            "previous_capsule_count": previous_capsule_count,
            "created_at": now,
        }])

    def _apply_mutation(
        self,
        mutation: dict[str, Any],
        *,
        now: float,
        session_id: str | None,
        platform: str | None,
        distiller_error: str | None,
    ) -> dict[str, Any] | None:
        action = str(mutation.get("action", "")).strip().lower()
        category = _normalize_category(mutation.get("category") or mutation.get("kind"))
        signals = _clean_signals(mutation.get("signals_match") or mutation.get("signals", []))
        reason = str(mutation.get("reason") or mutation.get("rationale") or "").strip()
        confidence = _clamp_float(mutation.get("confidence", 0.75), 0.0, 1.0)
        importance = _clamp_float(mutation.get("importance", 0.75), 0.0, 1.0)
        gene_ref = str(mutation.get("gene_id") or "").strip()
        key = str(mutation.get("key") or "").strip()
        strategy_steps = _clean_text_list(mutation.get("strategy", []))
        summary = str(mutation.get("summary") or mutation.get("rule") or "").strip()
        if not strategy_steps and summary:
            strategy_steps = [summary]
        validation = _clean_text_list(mutation.get("validation", []))

        gene = _find_matching_gene(self.genes, gene_ref) if gene_ref else None
        if gene is None and key:
            gene = self._find_gene(category, key)
        if gene is None and action in {"reinforce", "revise", "archive"} and key:
            gene = self._find_gene(category, key)

        if action == "archive":
            if gene is None:
                return None
            gene["status"] = "archived"
            gene["archived_reason"] = reason or "Archived by evolution distiller"
            gene["updated_at"] = now
        elif action == "reinforce":
            if gene is None:
                if not strategy_steps:
                    return None
                key = key or _stable_key(category, signals or strategy_steps)
                gene = self._new_gene(category, key, summary, signals, strategy_steps, validation, now, confidence, importance)
                self.genes.append(gene)
                action = "create"
            else:
                if summary:
                    gene["summary"] = summary
                    gene["rule"] = summary
                if strategy_steps:
                    gene["strategy"] = _merge_text_lists(_gene_strategy(gene), strategy_steps, limit=12)
                if validation:
                    gene["validation"] = _merge_text_lists(gene.get("validation", []), validation, limit=12)
                gene["signals_match"] = _merge_text_lists(_gene_signals(gene), signals, limit=24)
                gene["signals"] = gene["signals_match"]
                gene["score"] = round(float(gene.get("score", 0)) * self.config.decay + confidence, 3)
                gene["updated_at"] = now
                gene["evidence_count"] = int(gene.get("evidence_count", 0)) + 1
                _record_learning(gene, action, signals, confidence, importance, reason)
        elif action in {"create", "revise"}:
            if not strategy_steps:
                return None
            if gene is None:
                key = key or _stable_key(category, signals or strategy_steps)
                gene = self._find_gene(category, key)
            if gene is None:
                gene = self._new_gene(category, key, summary, signals, strategy_steps, validation, now, confidence, importance)
                self.genes.append(gene)
            else:
                if summary:
                    gene["summary"] = summary
                    gene["rule"] = summary
                gene["strategy"] = strategy_steps
                if validation:
                    gene["validation"] = validation
                gene["signals_match"] = _merge_text_lists(_gene_signals(gene), signals, limit=24)
                gene["signals"] = gene["signals_match"]
                gene["score"] = round(float(gene.get("score", 0)) * self.config.decay + confidence, 3)
                gene["updated_at"] = now
                gene["evidence_count"] = int(gene.get("evidence_count", 0)) + 1
                gene["status"] = "active"
                _record_learning(gene, action, signals, confidence, importance, reason)
        else:
            return None

        capsule = None
        if action in {"create", "reinforce", "revise"}:
            capsule = self._upsert_capsule(gene, summary, signals, confidence, now)
            adapt_gene_from_learning(
                gene=gene,
                outcome_status="success",
                learning_signals=build_learning_signals(signals=signals, category=_gene_category(gene), summary=summary),
                failure_mode={"mode": "none", "reasonClass": None, "retryable": False},
            )

        event = {
            "id": "evt_" + uuid.uuid4().hex[:12],
            "schema_version": "1.6.0",
            "type": "EvolutionEvent",
            "intent": _gene_category(gene),
            "action": action,
            "gene_id": gene.get("id"),
            "genes_used": [gene.get("id")],
            "category": _gene_category(gene),
            "signals": signals,
            "capsule_id": capsule.get("id") if capsule else None,
            "source_type": "hermes_personal_evolution",
            "mutation": {
                "summary": summary,
                "strategy": strategy_steps,
                "validation": validation,
                "reason": reason,
            },
            "outcome": {
                "status": "success" if action != "archive" else "archived",
                "confidence": confidence,
                "importance": importance,
                "score": float(gene.get("score", 0)),
            },
            "validation_report_id": self._write_validation_report(gene, action, now),
            "confidence": confidence,
            "importance": importance,
            "session_id": session_id,
            "platform": platform,
            "created_at": now,
        }
        if distiller_error:
            event["distiller_error"] = distiller_error
        return event

    def _upsert_capsule(
        self,
        gene: dict[str, Any],
        summary: str,
        signals: list[str],
        confidence: float,
        now: float,
    ) -> dict[str, Any]:
        gene_id = str(gene.get("id", ""))
        trigger = signals[:8] or _gene_signals(gene)[:8]
        existing = None
        for capsule in self.capsules:
            if capsule.get("gene") == gene_id and set(_clean_signals(capsule.get("trigger", []))) == set(trigger):
                existing = capsule
                break
        if existing is None:
            existing = {
                "type": "Capsule",
                "id": "cap_" + uuid.uuid4().hex[:12],
                "trigger": trigger,
                "gene": gene_id,
                "summary": summary or gene.get("summary", ""),
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
                "success_streak": 1,
            }
            self.capsules.append(existing)
        else:
            existing["summary"] = summary or existing.get("summary") or gene.get("summary", "")
            existing["confidence"] = round(max(float(existing.get("confidence", 0)), confidence), 3)
            existing["updated_at"] = now
            existing["success_streak"] = int(existing.get("success_streak", 0)) + 1
        return existing

    def _write_validation_report(self, gene: dict[str, Any], action: str, now: float) -> str:
        report_id = "valrpt_" + uuid.uuid4().hex[:12]
        report = {
            "type": "ValidationReport",
            "schema_version": "1.6.0",
            "id": report_id,
            "gene_id": gene.get("id"),
            "action": action,
            "validation": gene.get("validation", []),
            "outcome": {"status": "recorded", "score": float(gene.get("score", 0))},
            "created_at": now,
        }
        with self.validation_reports_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, ensure_ascii=True, sort_keys=True) + "\n")
        return report_id

    def _new_gene(
        self,
        category: str,
        key: str,
        summary: str,
        signals: list[str],
        strategy_steps: list[str],
        validation: list[str],
        now: float,
        confidence: float,
        importance: float,
    ) -> dict[str, Any]:
        score = confidence * 0.6 + importance * 0.4
        summary = summary or (strategy_steps[0] if strategy_steps else "")
        return {
            "type": "Gene",
            "id": "gene_" + uuid.uuid4().hex[:12],
            "category": category,
            "kind": category,
            "key": key,
            "summary": summary,
            "rule": summary,
            "signals_match": signals,
            "signals": signals,
            "score": round(score, 3),
            "strategy": strategy_steps,
            "validation": validation,
            "anti_patterns": [],
            "learning_history": [{
                "action": "create",
                "signals": signals,
                "confidence": confidence,
                "importance": importance,
                "created_at": now,
            }],
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "evidence_count": 1,
        }


def extract_signals(
    user_message: str,
    assistant_response: str,
    tool_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    text = (user_message or "").strip()
    response = (assistant_response or "").strip()
    lowered = text.lower()
    keywords = _keywords(text + " " + response)
    signals: list[dict[str, Any]] = []

    if any(p in lowered for p in ("i prefer", "i like", "i want", "my idea", "my goal", "my whole idea")):
        key = _stable_key("preference", keywords[:8] or [lowered[:48]])
        signals.append({
            "kind": "preference",
            "key": key,
            "rule": _rule_from_text("Respect this user preference/goal", text),
            "signals": keywords[:16],
        })

    if any(p in lowered for p in ("go ahead", "let's go ahead", "do it", "implement", "full aggressive", "don't be conservative")):
        key = "execution_bias"
        signals.append({
            "kind": "operating_style",
            "key": key,
            "rule": "Bias toward concrete implementation and forward progress rather than extended planning when the user gives directional approval.",
            "signals": sorted(set(keywords[:12] + ["execute", "implementation", "autonomy"])),
        })

    if any(p in lowered for p in ("personal assistant", "digital identity", "world model", "on my behalf")):
        key = "personal_assistant_identity"
        signals.append({
            "kind": "identity_model",
            "key": key,
            "rule": "Build and use a persistent model of the user's identity, goals, preferences, and operating context to take better actions on their behalf.",
            "signals": sorted(set(keywords[:16] + ["identity", "world-model", "assistant"])),
        })

    if tool_names:
        names = sorted({str(t) for t in tool_names if t})
        if names:
            key = _stable_key("tool_pattern", names)
            signals.append({
                "kind": "tool_pattern",
                "key": key,
                "rule": "When similar work appears, consider reusing the tool workflow that just succeeded: " + ", ".join(names[:10]) + ".",
                "signals": sorted(set(keywords[:12] + names[:10])),
            })

    return _dedupe(signals)


def distill_evolution_mutations(
    *,
    user_message: str,
    assistant_response: str,
    tool_names: Iterable[str] = (),
    current_genes: list[dict[str, Any]] | None = None,
    strategy: str = "balanced",
) -> list[dict[str, Any]]:
    """Use the auxiliary LLM to produce structured evolution mutations."""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    current = []
    for gene in (current_genes or []):
        if gene.get("status", "active") != "active":
            continue
        current.append({
            "id": gene.get("id"),
            "type": gene.get("type", "Gene"),
            "category": _gene_category(gene),
            "key": gene.get("key"),
            "summary": gene.get("summary") or gene.get("rule"),
            "signals_match": _gene_signals(gene)[:12],
            "strategy": _gene_strategy(gene)[:8],
            "validation": gene.get("validation", [])[:8],
            "score": gene.get("score", 0),
            "evidence_count": gene.get("evidence_count", 0),
        })
        if len(current) >= 20:
            break

    payload = {
        "user_message": user_message or "",
        "assistant_response": assistant_response or "",
        "tool_names": list(dict.fromkeys(str(t) for t in tool_names if t))[:20],
        "current_genes": current,
        "evolve_strategy": strategy,
    }
    system = (
        "You are Hermes' Evolver GEP distiller. Convert a completed turn into "
        "durable Gene mutations. Genes use the Evolver asset shape: type=Gene, "
        "category repair|optimize|innovate, signals_match, strategy, validation. "
        "Prefer revising or reinforcing existing Genes over creating duplicates. "
        "Archive Genes only when the turn clearly contradicts or invalidates them. "
        "Think if needed, but the final assistant content must be strict JSON only, no markdown."
    )
    user_prefix = (
        "Return this exact JSON shape:\n"
        "{\"mutations\":[{\"action\":\"create|reinforce|revise|archive\","
        "\"gene_id\":\"existing id when applicable\",\"category\":\"repair|optimize|innovate\","
        "\"key\":\"stable semantic key\",\"summary\":\"short durable summary\","
        "\"signals_match\":[\"short\",\"trigger\",\"tokens\"],"
        "\"strategy\":[\"concrete operating rule or step\"],"
        "\"validation\":[\"how to verify this gene was applied\"],"
        "\"confidence\":0.0,\"importance\":0.0,"
        "\"rationale\":\"why this mutation is justified\"}]}\n\n"
        "Rules:\n"
        "- Do not create a mutation for trivial one-off details.\n"
        "- Do not create broad tool workflow genes from incidental tool usage.\n"
        "- balanced: preserve a balanced mix of repair, optimize, and innovate genes.\n"
        "- innovate: prefer new capability-gap and future-behavior genes.\n"
        "- harden: prefer constraints, anti-regression checks, validation, and reliability genes.\n"
        "- repair-only: only emit repair genes; omit optimize/innovate mutations.\n"
        "- If the user expresses a long-term goal, preference, identity, operating style, "
        "or reusable workflow, create or revise a Gene.\n"
        "- confidence and importance must be numbers from 0 to 1.\n"
        "- If no durable learning exists, return {\"mutations\":[]}.\n\n"
        f"Strategy ratios: {json.dumps(STRATEGY_RATIOS.get(strategy, STRATEGY_RATIOS['balanced']), sort_keys=True)}\n"
        "Turn payload:\n"
    )
    user = user_prefix + _fit_payload_json_to_prompt(
        payload,
        prompt_prefix=user_prefix,
        max_chars=_evolution_prompt_max_chars(),
    )
    response = call_llm(
        "evolution",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=_evolution_distillation_max_tokens(),
    )
    finish_reason = getattr(response.choices[0], "finish_reason", None) if getattr(response, "choices", None) else None
    if finish_reason == "length":
        raise ValueError("evolution distiller hit max_tokens before producing final JSON")
    content = extract_content_or_reasoning(response)
    if not str(content or "").strip():
        raise ValueError("evolution distiller returned empty content")
    data = _parse_json_object(content)
    mutations = data.get("mutations", []) if isinstance(data, dict) else []
    if not isinstance(mutations, list):
        return []
    return [_normalize_mutation(m) for m in mutations if isinstance(m, dict)]


def handle_evolution_command(command: str, *, store: PersonalEvolutionStore | None = None) -> str:
    """Handle /evolution commands for CLI and gateway."""
    parts = command.strip().split()
    subcommand = parts[1].lower() if len(parts) > 1 else "list"
    store = store or PersonalEvolutionStore()

    if subcommand in {"help", "-h", "--help"}:
        return _evolution_help()
    if subcommand in {"list", "ls"}:
        return format_evolution_list(store)
    if subcommand in {"events", "event"}:
        limit = _parse_limit(parts[2:] if len(parts) > 2 else [], default=10)
        return format_evolution_events(store, limit=limit)
    if subcommand in {"capsules", "capsule"}:
        return format_evolution_capsules(store)
    if subcommand in {"narrative", "story"}:
        limit = _parse_limit(parts[2:] if len(parts) > 2 else [], default=20)
        return format_evolution_narrative(store, limit=limit)
    if subcommand in {"inspect", "show"}:
        if len(parts) < 3:
            return "Usage: /evolution inspect <gene_id|prefix|key>"
        return format_gene_inspect(store, parts[2])
    if subcommand in {"archive", "disable"}:
        if len(parts) < 3:
            return "Usage: /evolution archive <gene_id|prefix|key>"
        ok = store.archive_gene(parts[2])
        return f"Archived gene `{parts[2]}`." if ok else f"No gene matched `{parts[2]}`."
    if subcommand == "reset":
        if len(parts) < 3 or parts[2].lower() not in {"confirm", "--confirm"}:
            return "Destructive command. Use: /evolution reset confirm"
        store.reset()
        return "Personal evolution store reset."

    return f"Unknown evolution command `{subcommand}`.\n\n{_evolution_help()}"


def format_evolution_list(store: PersonalEvolutionStore, *, include_archived: bool = False) -> str:
    genes = [
        g for g in store.genes
        if include_archived or g.get("status", "active") == "active"
    ]
    if not genes:
        return (
            "No active personal evolution genes yet.\n"
            "Send preference/goal messages such as \"my goal is...\" or \"I prefer...\" to seed the layer."
        )
    genes.sort(key=lambda g: (float(g.get("score", 0)), float(g.get("updated_at", 0))), reverse=True)
    lines = [f"Personal Evolution Genes ({len(genes)} active)"]
    for gene in genes[:20]:
        gid = str(gene.get("id", ""))[:18]
        rule = _truncate(str(gene.get("summary") or gene.get("rule", "")).strip(), 150)
        lines.append(
            f"- `{gid}` {_gene_category(gene)} "
            f"score={float(gene.get('score', 0)):.2f} evidence={int(gene.get('evidence_count', 0))}: {rule}"
        )
    lines.append("\nUse `/evolution inspect <gene_id>` for details or `/evolution events` for recent mutations.")
    return "\n".join(lines)


def format_gene_inspect(store: PersonalEvolutionStore, gene_ref: str) -> str:
    gene = _find_matching_gene(store.genes, gene_ref)
    if not gene:
        return f"No gene matched `{gene_ref}`."
    return "\n".join([
        f"Gene `{gene.get('id')}`",
        f"Status: {gene.get('status', 'active')}",
        f"Type: {gene.get('type', 'Gene')}",
        f"Category: {_gene_category(gene)}",
        f"Score: {float(gene.get('score', 0)):.2f}",
        f"Evidence: {int(gene.get('evidence_count', 0))}",
        f"Key: {gene.get('key', '')}",
        f"Summary: {gene.get('summary') or gene.get('rule', '')}",
        "Signals Match: " + ", ".join(_gene_signals(gene)),
        "Strategy: " + "; ".join(_gene_strategy(gene)),
        "Validation: " + "; ".join(_clean_text_list(gene.get("validation", []))),
    ])


def format_evolution_events(store: PersonalEvolutionStore, *, limit: int = 10) -> str:
    if not store.events_path.exists():
        return "No evolution events recorded yet."
    try:
        lines = store.events_path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception as exc:
        return f"Failed to read evolution events: {exc}"
    if not lines:
        return "No evolution events recorded yet."
    rendered = [f"Recent Evolution Events ({len(lines)})"]
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        rendered.append(
            f"- `{event.get('id', '')}` {event.get('action') or event.get('type', 'event')} "
            f"{event.get('category') or event.get('kind', '')} gene={event.get('gene_id', '')}"
        )
    return "\n".join(rendered)


def format_evolution_capsules(store: PersonalEvolutionStore) -> str:
    if not store.capsules:
        return "No evolution capsules recorded yet."
    capsules = sorted(store.capsules, key=lambda c: (float(c.get("confidence", 0)), float(c.get("updated_at", 0))), reverse=True)
    lines = [f"Evolution Capsules ({len(capsules)})"]
    for capsule in capsules[:20]:
        lines.append(
            f"- `{capsule.get('id', '')}` gene={capsule.get('gene', '')} "
            f"confidence={float(capsule.get('confidence', 0)):.2f} "
            f"streak={int(capsule.get('success_streak', 0))}: "
            f"{_truncate(str(capsule.get('summary', '')), 150)}"
        )
    return "\n".join(lines)


def format_evolution_narrative(store: PersonalEvolutionStore, *, limit: int = 20) -> str:
    if not store.narrative_path.exists():
        return "No evolution narrative recorded yet."
    try:
        lines = store.narrative_path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception as exc:
        return f"Failed to read evolution narrative: {exc}"
    if not lines:
        return "No evolution narrative recorded yet."
    return "Evolution Narrative\n" + "\n".join(lines)


def _evolution_help() -> str:
    return "\n".join([
        "Usage: /evolution [list|events|capsules|narrative|inspect|archive|reset]",
        "- /evolution list",
        "- /evolution events [limit]",
        "- /evolution capsules",
        "- /evolution narrative [limit]",
        "- /evolution inspect <gene_id|prefix|key>",
        "- /evolution archive <gene_id|prefix|key>",
        "- /evolution reset confirm",
    ])


def _keywords(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw in _STOPWORDS or raw in seen:
            continue
        seen.add(raw)
        out.append(raw[:48])
    return out[:64]


def _stable_key(prefix: str, parts: Iterable[str]) -> str:
    joined = "|".join(str(p).lower() for p in parts if p)
    import hashlib
    return f"{prefix}:{hashlib.sha1(joined.encode('utf-8')).hexdigest()[:12]}"


def _rule_from_text(prefix: str, text: str) -> str:
    compact = " ".join((text or "").split())
    if len(compact) > 240:
        compact = compact[:237].rstrip() + "..."
    return f"{prefix}: {compact}"


def _dedupe(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for signal in signals:
        ident = (signal["kind"], signal["key"])
        if ident in seen:
            continue
        seen.add(ident)
        out.append(signal)
    return out


def _signals_to_mutations(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutations = []
    for signal in signals:
        mutations.append({
            "action": "create",
            "category": _legacy_kind_to_category(signal.get("kind", "optimize")),
            "key": signal.get("key", ""),
            "summary": signal.get("rule", ""),
            "signals_match": signal.get("signals", []),
            "strategy": [signal.get("rule", "")] if signal.get("rule") else [],
            "validation": [],
            "confidence": 0.65,
            "importance": 0.65,
            "rationale": "Heuristic fallback after evolution distiller was unavailable.",
        })
    return mutations


def _normalize_mutation(mutation: dict[str, Any]) -> dict[str, Any]:
    action = str(mutation.get("action", "")).strip().lower()
    if action not in {"create", "reinforce", "revise", "archive"}:
        action = "create"
    category = _normalize_category(mutation.get("category") or mutation.get("kind"))
    summary = str(mutation.get("summary") or mutation.get("rule") or "").strip()
    signals_match = _clean_signals(mutation.get("signals_match") or mutation.get("signals", []))
    strategy_steps = _clean_text_list(mutation.get("strategy", []))
    if not strategy_steps and summary:
        strategy_steps = [summary]
    return {
        "action": action,
        "gene_id": str(mutation.get("gene_id") or "").strip(),
        "category": category,
        "key": str(mutation.get("key") or "").strip(),
        "summary": summary,
        "signals_match": signals_match,
        "strategy": strategy_steps,
        "validation": _clean_text_list(mutation.get("validation", [])),
        "confidence": _clamp_float(mutation.get("confidence", 0.75), 0.0, 1.0),
        "importance": _clamp_float(mutation.get("importance", 0.75), 0.0, 1.0),
        "rationale": str(mutation.get("rationale") or mutation.get("reason") or "").strip(),
    }


def _clean_signals(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        token = str(item).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token[:48])
        if len(out) >= 24:
            break
    return out


def _clean_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        text = " ".join(str(item).strip().split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:240])
        if len(out) >= 24:
            break
    return out


def _merge_text_lists(existing: Any, incoming: Any, *, limit: int) -> list[str]:
    out = []
    seen = set()
    for text in _clean_text_list(existing) + _clean_text_list(incoming):
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _legacy_kind_to_category(kind: Any) -> str:
    raw = str(kind or "").strip().lower()
    if raw in {"constraint", "repair"}:
        return "repair"
    if raw in {"identity_model", "world_model", "workflow", "tool_pattern", "innovate"}:
        return "innovate"
    return "optimize"


def _normalize_category(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"repair", "optimize", "innovate"}:
        return raw
    return _legacy_kind_to_category(raw)


def _gene_category(gene: dict[str, Any]) -> str:
    return _normalize_category(gene.get("category") or gene.get("kind"))


def _gene_signals(gene: dict[str, Any]) -> list[str]:
    return _clean_signals(gene.get("signals_match") or gene.get("signals", []))


def _gene_strategy(gene: dict[str, Any]) -> list[str]:
    steps = _clean_text_list(gene.get("strategy", []))
    if not steps and gene.get("rule"):
        steps = _clean_text_list([gene.get("rule")])
    return steps


def _signal_base(signal: str) -> str:
    return signal.split(":", 1)[0].strip().lower()


def _signal_overlap(query_terms: set[str], signals: set[str]) -> int:
    if not query_terms or not signals:
        return 0
    signal_bases = {_signal_base(s) for s in signals}
    overlap = len(query_terms & signals)
    overlap += len({_signal_base(term) for term in query_terms} & signal_bases)
    return overlap


def select_capsules(
    capsules: list[dict[str, Any]],
    query_terms: set[str],
    selected_genes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_gene_ids = {str(g.get("id", "")) for g in selected_genes}
    scored: list[tuple[float, dict[str, Any]]] = []
    for capsule in capsules:
        if capsule.get("type", "Capsule") != "Capsule":
            continue
        trigger = set(_clean_signals(capsule.get("trigger", [])))
        overlap = _signal_overlap(query_terms, trigger)
        if capsule.get("gene") in selected_gene_ids:
            overlap += 2
        if overlap <= 0:
            continue
        score = overlap + float(capsule.get("confidence", 0))
        scored.append((score, capsule))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [capsule for _, capsule in scored[:8]]


def _strategy_category_weight(strategy: str, category: str) -> float:
    if strategy == "innovate":
        return 1.35 if category == "innovate" else 0.85
    if strategy == "harden":
        return 1.35 if category == "repair" else 0.9
    if strategy == "repair-only":
        return 1.0 if category == "repair" else 0.0
    return 1.0


def compute_drift_intensity(*, drift_enabled: bool, gene_pool_size: int, memory_evidence: int) -> float:
    pool = max(1, int(gene_pool_size or 1))
    base = min(1.0, 1.0 / math.sqrt(pool))
    if not drift_enabled:
        return base
    maturity = min(1.0, max(0, int(memory_evidence or 0)) / float(pool * 10))
    offset = 0.3 - (0.28 * maturity)
    return min(1.0, base + offset)


def build_learning_signals(*, signals: list[str], category: str, summary: str) -> list[str]:
    out = list(signals)
    if category:
        out.append(f"intent:{category}")
    lowered = (summary or "").lower()
    if any(term in lowered for term in ("performance", "latency", "slow", "throughput")):
        out.append("problem:performance")
    if any(term in lowered for term in ("validation", "test", "verify", "regression")):
        out.append("risk:validation")
    if any(term in lowered for term in ("protocol", "policy", "constraint")):
        out.append("problem:protocol")
    return _clean_signals(out)


def classify_failure_mode(*, failure_reason: str = "", validation_failed: bool = False, hard_violation: bool = False) -> dict[str, Any]:
    reason = (failure_reason or "").lower()
    if hard_violation or any(term in reason for term in ("destructive", "forbidden", "critical_file_deleted")):
        return {"mode": "hard", "reasonClass": "constraint_destructive", "retryable": False}
    if validation_failed or "validation" in reason or "test failed" in reason:
        return {"mode": "soft", "reasonClass": "validation", "retryable": True}
    if reason:
        return {"mode": "soft", "reasonClass": "runtime", "retryable": True}
    return {"mode": "none", "reasonClass": None, "retryable": False}


def adapt_gene_from_learning(
    *,
    gene: dict[str, Any],
    outcome_status: str,
    learning_signals: list[str],
    failure_mode: dict[str, Any],
) -> None:
    clean = [s for s in _clean_signals(learning_signals) if not s.startswith("action:")]
    history = gene.setdefault("learning_history", [])
    if not isinstance(history, list):
        history = []
        gene["learning_history"] = history
    history.append({
        "outcome": outcome_status,
        "mode": failure_mode.get("mode", "none"),
        "reason_class": failure_mode.get("reasonClass"),
        "learning_signals": clean,
        "created_at": time.time(),
    })
    del history[:-20]
    if outcome_status == "success":
        gene["signals_match"] = _merge_text_lists(_gene_signals(gene), clean, limit=24)
        gene["signals"] = gene["signals_match"]
        return
    anti_patterns = gene.setdefault("anti_patterns", [])
    if not isinstance(anti_patterns, list):
        anti_patterns = []
        gene["anti_patterns"] = anti_patterns
    anti_patterns.append({
        "mode": failure_mode.get("mode", "soft"),
        "reason_class": failure_mode.get("reasonClass"),
        "retryable": bool(failure_mode.get("retryable", True)),
        "learning_signals": clean,
        "created_at": time.time(),
    })
    del anti_patterns[:-20]


def _record_learning(
    gene: dict[str, Any],
    action: str,
    signals: list[str],
    confidence: float,
    importance: float,
    reason: str,
) -> None:
    history = gene.setdefault("learning_history", [])
    if not isinstance(history, list):
        history = []
        gene["learning_history"] = history
    history.append({
        "action": action,
        "signals": signals,
        "confidence": confidence,
        "importance": importance,
        "reason": reason,
        "created_at": time.time(),
    })
    del history[:-20]


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        raise


def _matches_gene(gene: dict[str, Any], ref: str) -> bool:
    ref = (ref or "").strip()
    return bool(ref) and (
        str(gene.get("id", "")) == ref
        or str(gene.get("id", "")).startswith(ref)
        or str(gene.get("key", "")) == ref
    )


def _find_matching_gene(genes: list[dict[str, Any]], ref: str) -> dict[str, Any] | None:
    for gene in genes:
        if _matches_gene(gene, ref):
            return gene
    return None


def _read_asset_list(path: Path, key: str) -> list[dict[str, Any]]:
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get(key, []) if isinstance(data, dict) else []
        return [item for item in items if isinstance(item, dict)]
    except Exception:
        return []


def _evolution_prompt_max_chars() -> int:
    for key in ("GEP_PROMPT_MAX_CHARS", "EVOLVER_PROMPT_MAX_CHARS", "HERMES_EVOLUTION_PROMPT_MAX_CHARS"):
        raw = os.getenv(key)
        if raw:
            try:
                return max(4000, int(raw))
            except ValueError:
                continue
    return EVOLVER_PROMPT_MAX_CHARS


def _evolution_distillation_max_tokens() -> int:
    for key in ("HERMES_EVOLUTION_DISTILLATION_MAX_TOKENS", "EVOLVER_DISTILLATION_MAX_TOKENS"):
        raw = os.getenv(key)
        if raw:
            try:
                return max(1024, int(raw))
            except ValueError:
                continue
    return EVOLUTION_DISTILLATION_MAX_TOKENS


def _fit_payload_json_to_prompt(payload: dict[str, Any], *, prompt_prefix: str, max_chars: int) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)
    if len(prompt_prefix) + len(payload_json) <= max_chars:
        return payload_json

    compact = dict(payload)
    fixed_payload = dict(compact)
    fixed_payload["user_message"] = ""
    fixed_payload["assistant_response"] = ""
    fixed_len = len(json.dumps(fixed_payload, ensure_ascii=True))
    remaining = max(1000, max_chars - len(prompt_prefix) - fixed_len - 16)
    user_text = str(payload.get("user_message") or "")
    assistant_text = str(payload.get("assistant_response") or "")
    total_text = max(1, len(user_text) + len(assistant_text))
    user_limit = max(250, int(remaining * len(user_text) / total_text))
    assistant_limit = max(250, remaining - user_limit)
    compact["user_message"] = _truncate(user_text, user_limit)
    compact["assistant_response"] = _truncate(assistant_text, assistant_limit)

    payload_json = json.dumps(compact, ensure_ascii=True)
    if len(prompt_prefix) + len(payload_json) <= max_chars:
        return payload_json
    return _truncate(payload_json, max(1000, max_chars - len(prompt_prefix)))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _parse_limit(args: list[str], *, default: int) -> int:
    if not args:
        return default
    try:
        return max(1, min(50, int(args[0])))
    except (TypeError, ValueError):
        return default
