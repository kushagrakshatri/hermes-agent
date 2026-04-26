"""Personal digital twin memory provider.

Phase 1 implements a local, conservative twin store. It gives Hermes a compact
identity/state/policy brief and explicit tools for user-confirmed writes,
corrections, entity beliefs, and action-policy lookup.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "workspace": "personal",
    "autonomy_mode": "conservative",
    "policy": {
        "default_external_action": "ask",
        "simulate_before_external_message": True,
        "simulate_before_public_post": True,
        "deny_money_movement": True,
    },
    "learning": {
        "promote_profile_after_repetitions": 3,
        "max_auto_profile_confidence": 0.75,
        "state_ttl_hours": 168,
    },
}

ALLOWED_DECISIONS = {"allow", "simulate", "ask", "deny"}
ALLOWED_RECORD_SCOPES = {"constitution", "profile", "state"}
ALLOWED_NOTE_SCOPES = {"constitution", "profile", "state", "entity_belief", "action_policy"}
SIMULATION_SOURCE = "simulation_hypothesis"


TWIN_PROFILE_SCHEMA = {
    "name": "twin_profile",
    "description": (
        "Inspect the user's personal digital twin. Returns confidence-tagged "
        "constitution, profile, current state, world-model, and policy context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": ["all", "constitution", "profile", "state", "world", "policy"],
                "description": "Twin section to inspect.",
            },
            "query": {"type": "string", "description": "Optional task or topic for relevance filtering."},
            "limit": {"type": "integer", "description": "Maximum records per section (default 6, max 20)."},
        },
        "required": [],
    },
}

TWIN_ENTITIES_SCHEMA = {
    "name": "twin_entities",
    "description": (
        "Inspect the twin world model: people, projects, organizations, and "
        "their confidence-tagged beliefs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Entity name or topic filter."},
            "limit": {"type": "integer", "description": "Maximum entities to return (default 8, max 25)."},
        },
        "required": [],
    },
}

TWIN_FEEDBACK_SCHEMA = {
    "name": "twin_feedback",
    "description": (
        "Record a user correction, confirmation, or observed outcome. Use this "
        "when the user says the twin is wrong or when a real-world result is known."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["correction", "confirmation", "outcome"]},
            "content": {"type": "string", "description": "Correction, confirmation, or outcome text."},
            "target_scope": {
                "type": "string",
                "enum": ["constitution", "profile", "state", "entity_belief", "action_policy"],
            },
            "target_key": {"type": "string", "description": "Key to correct or confirm."},
            "action_class": {"type": "string", "description": "Action class for outcome feedback."},
            "action_summary": {"type": "string", "description": "Brief description of the real-world action."},
            "predicted_ref": {"type": "string", "description": "Optional simulation or prediction reference."},
        },
        "required": ["kind", "content"],
    },
}

TWIN_POLICY_SCHEMA = {
    "name": "twin_policy",
    "description": (
        "Classify whether a candidate action is allowed, should be simulated, "
        "needs user approval, or is denied by the user's twin policy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action_class": {
                "type": "string",
                "description": "Class such as external_message, public_post, local_draft, money_movement.",
            },
            "proposed_action": {"type": "string", "description": "Short description of the candidate action."},
            "stakes": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["action_class"],
    },
}

TWIN_NOTE_SCHEMA = {
    "name": "twin_note",
    "description": (
        "Store an explicit, confidence-tagged twin fact. Constitution and action "
        "policy writes require confirmed=true. Simulation output cannot write to "
        "constitution or durable profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["constitution", "profile", "state", "entity_belief", "action_policy"],
            },
            "key": {"type": "string", "description": "Stable key, e.g. communication.directness."},
            "value": {"type": "string", "description": "Fact, belief, state, or policy decision."},
            "confidence": {"type": "number", "description": "Confidence from 0.0 to 1.0."},
            "source": {
                "type": "string",
                "enum": [
                    "explicit_user_statement",
                    "user_correction",
                    "repeated_observation",
                    "observed_reality",
                    "simulation_hypothesis",
                    "inference",
                ],
            },
            "confirmed": {
                "type": "boolean",
                "description": "True only when the user explicitly confirmed the write.",
            },
            "entity": {"type": "string", "description": "Entity name for entity_belief writes."},
            "entity_type": {"type": "string", "description": "person, project, organization, community, etc."},
            "expires_hours": {"type": "integer", "description": "Optional TTL for state records."},
            "reason": {"type": "string", "description": "Reason for action_policy decisions."},
        },
        "required": ["scope", "key", "value"],
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _expiry_after(hours: Optional[int]) -> Optional[str]:
    if hours is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(hours=int(hours))).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _clamp_confidence(value: Any, default: float = 0.6) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _coerce_limit(value: Any, default: int, maximum: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(maximum, limit))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(hermes_home: Path) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    path = hermes_home / "twin.json"
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = _deep_merge(config, loaded)
        except Exception as exc:
            logger.warning("Failed to read twin config %s: %s", path, exc)
    return config


def _jsonable_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_to_record(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "scope": row["scope"],
        "key": row["key"],
        "value": row["value"],
        "confidence": row["confidence"],
        "source": row["source"],
        "status": row["status"],
        "observed_at": row["observed_at"],
        "last_confirmed_at": row["last_confirmed_at"],
        "expires_at": row["expires_at"],
        "provenance_ref": row["provenance_ref"],
    }


class TwinMemoryProvider(MemoryProvider):
    """Local personal digital twin memory provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config_override = config or None
        self._config: Dict[str, Any] = {}
        self._hermes_home: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._session_id = ""
        self._agent_context = "primary"

    @property
    def name(self) -> str:
        return "twin"

    def is_available(self) -> bool:
        if self._config_override is not None:
            return bool(self._config_override.get("enabled", True))
        try:
            from hermes_constants import get_hermes_home

            return bool(_load_config(get_hermes_home()).get("enabled", True))
        except Exception:
            return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home_arg = kwargs.get("hermes_home")
        if hermes_home_arg:
            hermes_home = Path(str(hermes_home_arg))
        else:
            from hermes_constants import get_hermes_home

            hermes_home = get_hermes_home()

        hermes_home.mkdir(parents=True, exist_ok=True)
        twin_dir = hermes_home / "twin"
        twin_dir.mkdir(parents=True, exist_ok=True)

        self._hermes_home = hermes_home
        self._config = _deep_merge(_load_config(hermes_home), self._config_override or {})
        self._db_path = twin_dir / "twin.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._session_id = session_id
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._ensure_schema()
        self._seed_default_policies()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "twin.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        values = dict(values)
        policy_updates = {}
        if "default_external_action" in values:
            policy_updates["default_external_action"] = values.pop("default_external_action")
        if policy_updates:
            values["policy"] = _deep_merge(values.get("policy", {}), policy_updates)
        existing = _deep_merge(existing, values)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "workspace", "description": "Twin workspace name", "default": "personal"},
            {
                "key": "autonomy_mode",
                "description": "Default autonomy posture",
                "default": "conservative",
                "choices": ["conservative", "balanced", "permissive"],
            },
            {
                "key": "default_external_action",
                "description": "Default policy for external actions",
                "default": "ask",
                "choices": ["simulate", "ask", "deny"],
            },
        ]

    def system_prompt_block(self) -> str:
        return (
            "# Personal Twin\n"
            "A local personal digital twin is active. Treat constitution, durable profile, "
            "current state, world-model hypotheses, and observed outcomes as distinct layers. "
            "Simulation hypotheses are not observed reality. Do not rewrite core identity, "
            "red lines, or delegation scope unless the user explicitly confirms the change."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._conn:
            return ""
        try:
            return self._build_brief(query=query or "", section="all", limit=6)
        except Exception as exc:
            logger.debug("Twin prefetch failed: %s", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Phase 1 uses direct SQLite reads; no background work is needed.
        return None

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Phase 1 intentionally avoids implicit identity updates from normal turns.
        # Explicit user-confirmed writes go through twin_note/twin_feedback.
        return None

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._conn:
            self._conn.commit()

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if self._agent_context != "primary" or not self._conn:
            return
        try:
            now = _now()
            self._conn.execute(
                """
                INSERT INTO delegation_observations
                    (task, result, child_session_id, observed_at)
                VALUES (?, ?, ?, ?)
                """,
                (task, result, child_session_id, now),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("Twin delegation observation failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            TWIN_PROFILE_SCHEMA,
            TWIN_ENTITIES_SCHEMA,
            TWIN_FEEDBACK_SCHEMA,
            TWIN_POLICY_SCHEMA,
            TWIN_NOTE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._conn:
            return tool_error("Twin provider is not initialized")
        try:
            if tool_name == "twin_profile":
                return self._handle_profile(args)
            if tool_name == "twin_entities":
                return self._handle_entities(args)
            if tool_name == "twin_feedback":
                return self._handle_feedback(args)
            if tool_name == "twin_policy":
                return self._handle_policy(args)
            if tool_name == "twin_note":
                return self._handle_note(args)
            return tool_error(f"Unknown twin tool: {tool_name}")
        except ValueError as exc:
            return tool_error(str(exc), success=False)
        except Exception as exc:
            logger.exception("Twin tool %s failed", tool_name)
            return tool_error(f"Twin tool failed: {exc}", success=False)

    def shutdown(self) -> None:
        if self._conn:
            try:
                self._conn.commit()
                self._conn.close()
            finally:
                self._conn = None

    # -- Schema -------------------------------------------------------------

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.6,
                source TEXT NOT NULL DEFAULT 'explicit_user_statement',
                status TEXT NOT NULL DEFAULT 'active',
                observed_at TEXT NOT NULL,
                last_confirmed_at TEXT,
                expires_at TEXT,
                provenance_ref TEXT,
                UNIQUE(scope, key, status)
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT,
                summary TEXT,
                confidence REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT 'inference',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_beliefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT 'inference',
                observed_at TEXT NOT NULL,
                expires_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                FOREIGN KEY(entity_id) REFERENCES entities(id),
                UNIQUE(entity_id, key, status)
            );

            CREATE TABLE IF NOT EXISTS action_policies (
                action_class TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                reason TEXT,
                source TEXT NOT NULL DEFAULT 'default',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observed_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_class TEXT,
                action_summary TEXT,
                predicted_ref TEXT,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_scope TEXT,
                target_key TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS delegation_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                result TEXT NOT NULL,
                child_session_id TEXT,
                observed_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _seed_default_policies(self) -> None:
        assert self._conn is not None
        now = _now()
        defaults = {
            "local_draft": ("allow", "Local drafts do not cross an external boundary."),
            "file_read": ("allow", "Reading local context is low stakes."),
            "external_message": (
                "simulate" if self._policy_bool("simulate_before_external_message") else self._default_external_action(),
                "External messages can affect relationships and commitments.",
            ),
            "public_post": (
                "simulate" if self._policy_bool("simulate_before_public_post") else "ask",
                "Public posts carry reputation risk.",
            ),
            "money_movement": (
                "deny" if self._policy_bool("deny_money_movement") else "ask",
                "Money movement is outside default delegation scope.",
            ),
            "account_or_credentials_change": ("deny", "Credentials and account changes require direct user control."),
        }
        for action_class, (decision, reason) in defaults.items():
            self._conn.execute(
                """
                INSERT OR IGNORE INTO action_policies
                    (action_class, decision, reason, source, updated_at)
                VALUES (?, ?, ?, 'default', ?)
                """,
                (action_class, decision, reason, now),
            )
        self._conn.commit()

    # -- Tool handlers ------------------------------------------------------

    def _handle_profile(self, args: Dict[str, Any]) -> str:
        section = str(args.get("section") or "all").strip() or "all"
        limit = _coerce_limit(args.get("limit"), default=6, maximum=20)
        query = str(args.get("query") or "")
        if section not in {"all", "constitution", "profile", "state", "world", "policy"}:
            raise ValueError(f"Invalid twin section: {section}")
        return tool_result(
            success=True,
            section=section,
            brief=self._build_brief(query=query, section=section, limit=limit),
            data=self._profile_data(query=query, section=section, limit=limit),
        )

    def _handle_entities(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "")
        limit = _coerce_limit(args.get("limit"), default=8, maximum=25)
        return tool_result(success=True, entities=self._entities(query=query, limit=limit))

    def _handle_policy(self, args: Dict[str, Any]) -> str:
        action_class = str(args.get("action_class") or "").strip()
        if not action_class:
            raise ValueError("action_class is required")
        stakes = str(args.get("stakes") or "medium").strip().lower()
        proposed_action = str(args.get("proposed_action") or "")
        policy = self._classify_policy(action_class, stakes=stakes, proposed_action=proposed_action)
        return tool_result(success=True, **policy)

    def _handle_note(self, args: Dict[str, Any]) -> str:
        scope = str(args.get("scope") or "").strip()
        key = str(args.get("key") or "").strip()
        value = args.get("value")
        source = str(args.get("source") or "explicit_user_statement").strip()
        confirmed = bool(args.get("confirmed", False))
        confidence = _clamp_confidence(args.get("confidence"), default=0.7 if confirmed else 0.55)

        if scope not in ALLOWED_NOTE_SCOPES:
            raise ValueError(f"Invalid twin note scope: {scope}")
        if not key:
            raise ValueError("key is required")
        if value is None or str(value).strip() == "":
            raise ValueError("value is required")

        if scope in ALLOWED_RECORD_SCOPES:
            record_id = self._upsert_record(
                scope=scope,
                key=key,
                value=_jsonable_value(value),
                confidence=confidence,
                source=source,
                confirmed=confirmed,
                expires_hours=args.get("expires_hours"),
            )
            return tool_result(success=True, id=record_id, scope=scope, key=key)

        if scope == "entity_belief":
            entity = str(args.get("entity") or "").strip()
            if not entity:
                raise ValueError("entity is required for entity_belief notes")
            entity_id = self._upsert_entity(
                name=entity,
                entity_type=str(args.get("entity_type") or "unknown"),
                summary="",
                confidence=confidence,
                source=source,
            )
            belief_id = self._upsert_entity_belief(
                entity_id=entity_id,
                key=key,
                value=_jsonable_value(value),
                confidence=confidence,
                source=source,
                expires_hours=args.get("expires_hours"),
            )
            return tool_result(success=True, id=belief_id, scope=scope, entity=entity, key=key)

        if scope == "action_policy":
            if not confirmed:
                raise ValueError("Action policy writes require confirmed=true")
            decision = str(value).strip().lower()
            if decision not in ALLOWED_DECISIONS:
                raise ValueError(f"Action policy decision must be one of: {', '.join(sorted(ALLOWED_DECISIONS))}")
            reason = str(args.get("reason") or f"User-confirmed policy for {key}")
            self._upsert_action_policy(action_class=key, decision=decision, reason=reason, source=source)
            return tool_result(success=True, scope=scope, action_class=key, decision=decision)

        raise ValueError(f"Unsupported scope: {scope}")

    def _handle_feedback(self, args: Dict[str, Any]) -> str:
        kind = str(args.get("kind") or "").strip()
        content = str(args.get("content") or "").strip()
        if kind not in {"correction", "confirmation", "outcome"}:
            raise ValueError("kind must be correction, confirmation, or outcome")
        if not content:
            raise ValueError("content is required")

        if kind == "outcome":
            now = _now()
            self._conn.execute(
                """
                INSERT INTO observed_outcomes
                    (action_class, action_summary, predicted_ref, outcome, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(args.get("action_class") or ""),
                    str(args.get("action_summary") or ""),
                    str(args.get("predicted_ref") or ""),
                    content,
                    now,
                ),
            )
            self._conn.commit()
            return tool_result(success=True, kind=kind)

        target_scope = str(args.get("target_scope") or "").strip()
        target_key = str(args.get("target_key") or "").strip()
        now = _now()
        self._conn.execute(
            """
            INSERT INTO corrections (target_scope, target_key, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (target_scope, target_key, content, now),
        )

        updated = 0
        if target_scope and target_key:
            if kind == "correction":
                updated = self._demote_target(target_scope, target_key)
            elif kind == "confirmation":
                updated = self._confirm_target(target_scope, target_key)
        self._conn.commit()
        return tool_result(success=True, kind=kind, updated=updated)

    # -- Write helpers ------------------------------------------------------

    def _upsert_record(
        self,
        *,
        scope: str,
        key: str,
        value: str,
        confidence: float,
        source: str,
        confirmed: bool,
        expires_hours: Any = None,
    ) -> int:
        if scope == "constitution":
            if not confirmed:
                raise ValueError("Constitution writes require confirmed=true")
            if source not in {"explicit_user_statement", "user_correction"}:
                raise ValueError("Constitution writes require explicit user confirmation or correction")
        if scope == "profile" and source == SIMULATION_SOURCE:
            raise ValueError("Simulation output cannot write directly to durable profile")
        if scope == "constitution" and source == SIMULATION_SOURCE:
            raise ValueError("Simulation output cannot write to constitution")

        if scope == "profile" and source != "explicit_user_statement":
            max_auto = _clamp_confidence(
                self._config.get("learning", {}).get("max_auto_profile_confidence"),
                default=0.75,
            )
            confidence = min(confidence, max_auto)

        if scope == "state" and expires_hours is None:
            expires_hours = self._config.get("learning", {}).get("state_ttl_hours", 168)

        now = _now()
        expires_at = _expiry_after(expires_hours) if expires_hours is not None else None
        last_confirmed = now if confirmed or source in {"explicit_user_statement", "user_correction"} else None
        self._conn.execute(
            """
            INSERT INTO records
                (scope, key, value, confidence, source, status, observed_at,
                 last_confirmed_at, expires_at, provenance_ref)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            ON CONFLICT(scope, key, status) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                source=excluded.source,
                observed_at=excluded.observed_at,
                last_confirmed_at=excluded.last_confirmed_at,
                expires_at=excluded.expires_at,
                provenance_ref=excluded.provenance_ref
            """,
            (scope, key, value, confidence, source, now, last_confirmed, expires_at, self._session_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM records WHERE scope=? AND key=? AND status='active'",
            (scope, key),
        ).fetchone()
        return int(row["id"])

    def _upsert_entity(self, *, name: str, entity_type: str, summary: str, confidence: float, source: str) -> int:
        now = _now()
        self._conn.execute(
            """
            INSERT INTO entities
                (name, type, summary, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                type=COALESCE(NULLIF(excluded.type, ''), type),
                summary=COALESCE(NULLIF(excluded.summary, ''), summary),
                confidence=MAX(confidence, excluded.confidence),
                updated_at=excluded.updated_at
            """,
            (name, entity_type, summary, confidence, source, now, now),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        return int(row["id"])

    def _upsert_entity_belief(
        self,
        *,
        entity_id: int,
        key: str,
        value: str,
        confidence: float,
        source: str,
        expires_hours: Any = None,
    ) -> int:
        now = _now()
        expires_at = _expiry_after(expires_hours) if expires_hours is not None else None
        self._conn.execute(
            """
            INSERT INTO entity_beliefs
                (entity_id, key, value, confidence, source, observed_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
            ON CONFLICT(entity_id, key, status) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                source=excluded.source,
                observed_at=excluded.observed_at,
                expires_at=excluded.expires_at
            """,
            (entity_id, key, value, confidence, source, now, expires_at),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM entity_beliefs WHERE entity_id=? AND key=? AND status='active'",
            (entity_id, key),
        ).fetchone()
        return int(row["id"])

    def _upsert_action_policy(self, *, action_class: str, decision: str, reason: str, source: str) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT INTO action_policies
                (action_class, decision, reason, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(action_class) DO UPDATE SET
                decision=excluded.decision,
                reason=excluded.reason,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (action_class, decision, reason, source, now),
        )
        self._conn.commit()

    def _demote_target(self, target_scope: str, target_key: str) -> int:
        if target_scope in ALLOWED_RECORD_SCOPES:
            cur = self._conn.execute(
                """
                UPDATE records
                SET confidence=MIN(confidence, 0.2), status='corrected:' || id
                WHERE scope=? AND key=? AND status='active'
                """,
                (target_scope, target_key),
            )
            return cur.rowcount
        if target_scope == "entity_belief":
            cur = self._conn.execute(
                """
                UPDATE entity_beliefs
                SET confidence=MIN(confidence, 0.2), status='corrected:' || id
                WHERE key=? AND status='active'
                """,
                (target_key,),
            )
            return cur.rowcount
        if target_scope == "action_policy":
            cur = self._conn.execute(
                "DELETE FROM action_policies WHERE action_class=? AND source!='default'",
                (target_key,),
            )
            return cur.rowcount
        return 0

    def _confirm_target(self, target_scope: str, target_key: str) -> int:
        now = _now()
        if target_scope in ALLOWED_RECORD_SCOPES:
            cur = self._conn.execute(
                """
                UPDATE records
                SET confidence=MAX(confidence, 0.9), last_confirmed_at=?
                WHERE scope=? AND key=? AND status='active'
                """,
                (now, target_scope, target_key),
            )
            return cur.rowcount
        if target_scope == "entity_belief":
            cur = self._conn.execute(
                """
                UPDATE entity_beliefs
                SET confidence=MAX(confidence, 0.85)
                WHERE key=? AND status='active'
                """,
                (target_key,),
            )
            return cur.rowcount
        return 0

    # -- Read helpers -------------------------------------------------------

    def _active_records(self, scope: str, limit: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM records
            WHERE scope=? AND status='active'
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY confidence DESC, observed_at DESC
            LIMIT ?
            """,
            (scope, _now(), limit),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def _policies(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT action_class, decision, reason, source, updated_at
            FROM action_policies
            ORDER BY action_class ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _entities(self, *, query: str, limit: int) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = ""
        terms = [term for term in query.lower().split() if len(term) >= 2]
        if terms:
            clauses = []
            for term in terms[:4]:
                clauses.append("(LOWER(e.name) LIKE ? OR LOWER(COALESCE(e.summary, '')) LIKE ?)")
                needle = f"%{term}%"
                params.extend([needle, needle])
            where = "WHERE " + " OR ".join(clauses)

        rows = self._conn.execute(
            f"""
            SELECT e.*
            FROM entities e
            {where}
            ORDER BY e.confidence DESC, e.updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        entities = []
        for row in rows:
            beliefs = self._conn.execute(
                """
                SELECT key, value, confidence, source, observed_at, expires_at
                FROM entity_beliefs
                WHERE entity_id=? AND status='active'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY confidence DESC, observed_at DESC
                LIMIT 8
                """,
                (row["id"], _now()),
            ).fetchall()
            entities.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "summary": row["summary"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "beliefs": [dict(b) for b in beliefs],
                }
            )
        return entities

    def _profile_data(self, *, query: str, section: str, limit: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if section in {"all", "constitution"}:
            data["constitution"] = self._active_records("constitution", limit)
        if section in {"all", "profile"}:
            data["profile"] = self._active_records("profile", limit)
        if section in {"all", "state"}:
            data["state"] = self._active_records("state", limit)
        if section in {"all", "world"}:
            data["world"] = self._entities(query=query, limit=min(limit, 10))
        if section in {"all", "policy"}:
            data["policy"] = self._policies(limit=limit)
        return data

    def _build_brief(self, *, query: str, section: str, limit: int) -> str:
        data = self._profile_data(query=query, section=section, limit=limit)
        lines = ["## Personal Twin Context"]
        for scope in ("constitution", "profile", "state"):
            records = data.get(scope) or []
            if records:
                lines.append(f"{scope.title()}:")
                for record in records:
                    lines.append(
                        f"- {record['key']}: {record['value']} "
                        f"(confidence {record['confidence']:.2f}, source {record['source']})"
                    )
        entities = data.get("world") or []
        if entities:
            lines.append("World Model:")
            for entity in entities:
                lines.append(
                    f"- {entity['name']} ({entity.get('type') or 'entity'}, "
                    f"confidence {entity['confidence']:.2f})"
                )
                for belief in entity.get("beliefs", [])[:3]:
                    lines.append(
                        f"  - {belief['key']}: {belief['value']} "
                        f"(confidence {belief['confidence']:.2f}, source {belief['source']})"
                    )
        policies = data.get("policy") or []
        if policies:
            lines.append("Action Policy:")
            for policy in policies[:limit]:
                reason = f" - {policy['reason']}" if policy.get("reason") else ""
                lines.append(f"- {policy['action_class']}: {policy['decision']}{reason}")
        if len(lines) == 1:
            lines.append("- No active twin records yet. Use twin_note for explicit, confirmed facts.")
        return "\n".join(lines)

    # -- Policy -------------------------------------------------------------

    def _policy_bool(self, key: str) -> bool:
        return bool(self._config.get("policy", {}).get(key, DEFAULT_CONFIG["policy"].get(key)))

    def _default_external_action(self) -> str:
        value = str(self._config.get("policy", {}).get("default_external_action", "ask")).lower()
        return value if value in {"simulate", "ask", "deny"} else "ask"

    def _classify_policy(self, action_class: str, *, stakes: str, proposed_action: str) -> Dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT action_class, decision, reason, source, updated_at
            FROM action_policies
            WHERE action_class=?
            """,
            (action_class,),
        ).fetchone()

        if row:
            decision = row["decision"]
            reason = row["reason"] or ""
            source = row["source"]
        else:
            decision = self._default_decision_for_unknown(stakes=stakes)
            reason = "No explicit policy exists; using autonomy-mode default."
            source = "fallback"

        if stakes == "high" and decision == "allow":
            decision = "ask"
            reason = "High-stakes actions require user approval even if the base policy allows them."

        return {
            "action_class": action_class,
            "decision": decision,
            "reason": reason,
            "source": source,
            "requires_simulation": decision == "simulate",
            "requires_user_approval": decision in {"simulate", "ask"},
            "proposed_action": proposed_action,
        }

    def _default_decision_for_unknown(self, *, stakes: str) -> str:
        mode = str(self._config.get("autonomy_mode", "conservative")).lower()
        if stakes == "high":
            return "ask"
        if mode == "permissive" and stakes == "low":
            return "allow"
        if mode == "balanced" and stakes == "low":
            return "allow"
        return "ask"


def register(ctx) -> None:
    """Register Twin as a memory provider plugin."""
    ctx.register_memory_provider(TwinMemoryProvider())
