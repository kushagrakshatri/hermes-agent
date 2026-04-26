#!/usr/bin/env python3
"""MiroFish integration tools.

These tools are intentionally thin HTTP wrappers around a running MiroFish
backend. They expose graph, simulation, interview, environment, and report
operations without granting autonomous action authority to simulation output.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result


class MiroFishError(RuntimeError):
    """Raised when the MiroFish backend cannot satisfy a request."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _load_mirofish_config() -> dict[str, Any]:
    """Load optional MiroFish settings from HERMES_HOME/twin.json."""

    path = get_hermes_home() / "twin.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}

    section = data.get("mirofish")
    return section if isinstance(section, dict) else {}


def _config() -> dict[str, Any]:
    configured = _load_mirofish_config()
    base_url = (
        os.getenv("MIROFISH_BASE_URL", "").strip()
        or str(configured.get("base_url") or "").strip()
    ).rstrip("/")

    timeout_raw = os.getenv("MIROFISH_TIMEOUT", "").strip() or configured.get("timeout", 30)
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        timeout = 30.0

    enabled = configured.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}

    return {
        "base_url": base_url,
        "api_key": os.getenv("MIROFISH_API_KEY", "").strip() or configured.get("api_key"),
        "timeout": timeout,
        "enabled": bool(enabled),
        "default_project": configured.get("default_project") or "personal-twin",
    }


def check_mirofish_requirements() -> bool:
    """Return True when the local MiroFish backend is configured."""

    cfg = _config()
    return bool(cfg["enabled"] and cfg["base_url"])


def _json_loads_maybe(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config()
    if not cfg["enabled"]:
        raise MiroFishError("MiroFish integration is disabled in twin.json")
    if not cfg["base_url"]:
        raise MiroFishError(
            "MiroFish is not configured. Set MIROFISH_BASE_URL or twin.json mirofish.base_url."
        )

    query = {
        key: value
        for key, value in (params or {}).items()
        if value is not None and value != ""
    }
    url = f"{cfg['base_url']}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"

    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "hermes-agent-mirofish/1",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout"]) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = _json_loads_maybe(body)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = _json_loads_maybe(body)
        message = parsed.get("error") if isinstance(parsed, dict) else None
        raise MiroFishError(
            message or f"MiroFish HTTP {exc.code}",
            status_code=exc.code,
            body=parsed,
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise MiroFishError(f"Could not reach MiroFish backend: {reason}") from exc


def _multipart_request(
    path: str,
    *,
    fields: dict[str, Any],
    files: list[tuple[str, str, bytes, str]],
) -> dict[str, Any]:
    cfg = _config()
    if not cfg["enabled"]:
        raise MiroFishError("MiroFish integration is disabled in twin.json")
    if not cfg["base_url"]:
        raise MiroFishError(
            "MiroFish is not configured. Set MIROFISH_BASE_URL or twin.json mirofish.base_url."
        )

    boundary = f"----HermesMiroFish{os.urandom(8).hex()}"
    parts: list[bytes] = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    for field_name, filename, content, content_type in files:
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            content,
            b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "hermes-agent-mirofish/1",
    }
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    request = urllib.request.Request(
        f"{cfg['base_url']}{path}",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout"]) as response:
            parsed = _json_loads_maybe(response.read().decode("utf-8", errors="replace"))
            return parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        parsed = _json_loads_maybe(body_text)
        message = parsed.get("error") if isinstance(parsed, dict) else None
        raise MiroFishError(
            message or f"MiroFish HTTP {exc.code}",
            status_code=exc.code,
            body=parsed,
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise MiroFishError(f"Could not reach MiroFish backend: {reason}") from exc


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _payload(args: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    raw_payload = args.get("payload")
    result = dict(raw_payload) if isinstance(raw_payload, dict) else {}
    for key in keys:
        if key in args and _present(args[key]):
            result[key] = args[key]
    return result


def _find_id(data: Any, names: tuple[str, ...]) -> str | None:
    if isinstance(data, dict):
        for name in names:
            value = data.get(name)
            if isinstance(value, str) and value:
                return value
        nested = data.get("data")
        if nested is not data:
            found = _find_id(nested, names)
            if found:
                return found
    return None


def _success(raw: dict[str, Any]) -> bool:
    return bool(raw.get("success", "error" not in raw))


def _data(raw: dict[str, Any]) -> dict[str, Any]:
    data = raw.get("data")
    return data if isinstance(data, dict) else {}


def _result(raw: dict[str, Any], *, endpoint: str, **extra: Any) -> str:
    success = raw.get("success")
    if success is None:
        success = "error" not in raw

    body = {
        "success": bool(success),
        "endpoint": endpoint,
        "mirofish": raw,
    }
    for key, value in extra.items():
        if _present(value):
            body[key] = value
    if not success and raw.get("error"):
        body["error"] = raw["error"]
    return tool_result(body)


def _error_result(exc: Exception) -> str:
    if isinstance(exc, MiroFishError):
        extra: dict[str, Any] = {}
        if exc.status_code is not None:
            extra["status_code"] = exc.status_code
        if exc.body is not None:
            extra["body"] = exc.body
        return tool_error(str(exc), success=False, **extra)
    return tool_error(str(exc), success=False)


def mirofish_build_graph(args: dict[str, Any], **_: Any) -> str:
    """Start a MiroFish graph build for an existing project."""

    try:
        payload = _payload(
            args,
            ["project_id", "graph_name", "chunk_size", "chunk_overlap", "force"],
        )
        if not payload.get("project_id"):
            payload["project_id"] = _config()["default_project"]
        raw = _request("POST", "/api/graph/build", payload=payload)
        return _result(
            raw,
            endpoint="/api/graph/build",
            project_id=payload.get("project_id"),
            task_id=_find_id(raw, ("task_id",)),
            graph_id=_find_id(raw, ("graph_id",)),
        )
    except Exception as exc:
        return _error_result(exc)


def mirofish_prepare_simulation(args: dict[str, Any], **_: Any) -> str:
    """Prepare a simulation, optionally creating it first from project_id."""

    try:
        simulation_id = args.get("simulation_id")
        create_response: dict[str, Any] | None = None

        if not simulation_id:
            create_payload = _payload(
                args,
                ["project_id", "graph_id", "enable_twitter", "enable_reddit"],
            )
            if not create_payload.get("project_id"):
                create_payload["project_id"] = _config()["default_project"]
            create_response = _request("POST", "/api/simulation/create", payload=create_payload)
            if create_response.get("success") is False:
                return _result(create_response, endpoint="/api/simulation/create")
            simulation_id = _find_id(create_response, ("simulation_id",))
            if not simulation_id:
                return tool_error(
                    "MiroFish did not return simulation_id from /api/simulation/create",
                    success=False,
                    mirofish=create_response,
                )

        prepare_payload = _payload(
            args,
            [
                "entity_types",
                "use_llm_for_profiles",
                "parallel_profile_count",
                "force_regenerate",
            ],
        )
        prepare_payload["simulation_id"] = simulation_id
        raw = _request("POST", "/api/simulation/prepare", payload=prepare_payload)
        return _result(
            raw,
            endpoint="/api/simulation/prepare",
            simulation_id=simulation_id,
            task_id=_find_id(raw, ("task_id",)),
            create_response=create_response,
        )
    except Exception as exc:
        return _error_result(exc)


def mirofish_run_simulation(args: dict[str, Any], **_: Any) -> str:
    """Start a prepared MiroFish simulation."""

    try:
        payload = _payload(
            args,
            [
                "simulation_id",
                "platform",
                "max_rounds",
                "enable_graph_memory_update",
                "force",
            ],
        )
        if not payload.get("simulation_id"):
            return tool_error("simulation_id is required", success=False)
        raw = _request("POST", "/api/simulation/start", payload=payload)
        return _result(
            raw,
            endpoint="/api/simulation/start",
            simulation_id=payload.get("simulation_id"),
        )
    except Exception as exc:
        return _error_result(exc)


def mirofish_simulation_status(args: dict[str, Any], **_: Any) -> str:
    """Read simulation run, prepare, or environment status."""

    try:
        simulation_id = args.get("simulation_id")
        status_type = (args.get("status_type") or "run").strip().lower()
        task_id = args.get("task_id")

        if status_type == "prepare":
            payload = _payload(args, ["task_id", "simulation_id"])
            raw = _request("POST", "/api/simulation/prepare/status", payload=payload)
            return _result(raw, endpoint="/api/simulation/prepare/status", task_id=task_id)

        if not simulation_id:
            return tool_error("simulation_id is required", success=False)

        if status_type == "env":
            raw = _request(
                "POST",
                "/api/simulation/env-status",
                payload={"simulation_id": simulation_id},
            )
            return _result(raw, endpoint="/api/simulation/env-status", simulation_id=simulation_id)

        if status_type == "run_detail":
            endpoint = f"/api/simulation/{urllib.parse.quote(str(simulation_id))}/run-status/detail"
        else:
            endpoint = f"/api/simulation/{urllib.parse.quote(str(simulation_id))}/run-status"
        raw = _request("GET", endpoint)
        return _result(raw, endpoint=endpoint, simulation_id=simulation_id)
    except Exception as exc:
        return _error_result(exc)


def mirofish_generate_report(args: dict[str, Any], **_: Any) -> str:
    """Start report generation for a simulation."""

    try:
        payload = _payload(args, ["simulation_id", "force_regenerate"])
        if not payload.get("simulation_id"):
            return tool_error("simulation_id is required", success=False)
        raw = _request("POST", "/api/report/generate", payload=payload)
        return _result(
            raw,
            endpoint="/api/report/generate",
            simulation_id=payload.get("simulation_id"),
            task_id=_find_id(raw, ("task_id",)),
            report_id=_find_id(raw, ("report_id",)),
        )
    except Exception as exc:
        return _error_result(exc)


def mirofish_get_report(args: dict[str, Any], **_: Any) -> str:
    """Fetch a report by report_id or simulation_id."""

    try:
        report_id = args.get("report_id")
        simulation_id = args.get("simulation_id")
        status_only = bool(args.get("status_only"))

        if status_only:
            payload = _payload(args, ["task_id", "simulation_id", "report_id"])
            raw = _request("POST", "/api/report/generate/status", payload=payload)
            return _result(raw, endpoint="/api/report/generate/status", task_id=args.get("task_id"))

        if report_id:
            endpoint = f"/api/report/{urllib.parse.quote(str(report_id))}"
        elif simulation_id:
            endpoint = f"/api/report/by-simulation/{urllib.parse.quote(str(simulation_id))}"
        else:
            return tool_error("report_id or simulation_id is required", success=False)

        raw = _request("GET", endpoint)
        return _result(raw, endpoint=endpoint, report_id=report_id, simulation_id=simulation_id)
    except Exception as exc:
        return _error_result(exc)


def mirofish_interview_agents(args: dict[str, Any], **_: Any) -> str:
    """Interview one, several, or all agents in a simulation."""

    try:
        mode = (args.get("mode") or "single").strip().lower()
        payload = _payload(
            args,
            [
                "simulation_id",
                "agent_id",
                "agent_ids",
                "interviews",
                "question",
                "prompt",
                "platform",
                "timeout",
            ],
        )
        if not payload.get("simulation_id"):
            return tool_error("simulation_id is required", success=False)
        if payload.get("question") and not payload.get("prompt"):
            payload["prompt"] = payload.pop("question")

        if mode == "all":
            endpoint = "/api/simulation/interview/all"
            if not payload.get("prompt"):
                return tool_error("prompt or question is required", success=False)
        elif mode == "batch":
            endpoint = "/api/simulation/interview/batch"
            if not payload.get("interviews") and payload.get("agent_ids"):
                if not payload.get("prompt"):
                    return tool_error("prompt or question is required for batch interviews", success=False)
                prompt = payload["prompt"]
                payload["interviews"] = [
                    {"agent_id": agent_id, "prompt": prompt}
                    for agent_id in payload["agent_ids"]
                ]
            if not payload.get("interviews"):
                return tool_error("interviews or agent_ids is required for batch interviews", success=False)
            payload.pop("agent_ids", None)
            payload.pop("agent_id", None)
            payload.pop("prompt", None)
        else:
            endpoint = "/api/simulation/interview"
            if not payload.get("agent_id"):
                return tool_error("agent_id is required for single-agent interviews", success=False)
            if not payload.get("prompt"):
                return tool_error("prompt or question is required", success=False)
            payload.pop("agent_ids", None)
            payload.pop("interviews", None)

        raw = _request("POST", endpoint, payload=payload)
        return _result(raw, endpoint=endpoint, simulation_id=payload.get("simulation_id"))
    except Exception as exc:
        return _error_result(exc)


def mirofish_close_env(args: dict[str, Any], **_: Any) -> str:
    """Close a MiroFish simulation environment."""

    try:
        payload = _payload(args, ["simulation_id", "timeout"])
        if not payload.get("simulation_id"):
            return tool_error("simulation_id is required", success=False)
        raw = _request("POST", "/api/simulation/close-env", payload=payload)
        return _result(raw, endpoint="/api/simulation/close-env", simulation_id=payload.get("simulation_id"))
    except Exception as exc:
        return _error_result(exc)


def _decision_seed_markdown(args: dict[str, Any]) -> str:
    decision = str(args.get("decision") or "").strip()
    options = args.get("options") if isinstance(args.get("options"), list) else []
    stakeholders = args.get("stakeholders") if isinstance(args.get("stakeholders"), list) else []
    context = str(args.get("context") or "").strip()
    success_criteria = str(args.get("success_criteria") or "").strip()
    risks = str(args.get("risks") or "").strip()

    sections = [
        "# Hermes Decision Simulation Seed",
        "",
        "## Decision",
        decision,
        "",
        "## Options",
    ]
    if options:
        sections.extend(f"- {option}" for option in options)
    else:
        sections.append("- Proceed with the proposed action")
        sections.append("- Modify the proposed action")
        sections.append("- Delay or avoid the proposed action")
    sections.extend(["", "## Stakeholders"])
    if stakeholders:
        sections.extend(f"- {stakeholder}" for stakeholder in stakeholders)
    else:
        sections.extend([
            "- User",
            "- Intended recipient or audience",
            "- Future self",
            "- Operational constraints",
        ])
    sections.extend([
        "",
        "## Context",
        context or "No additional context provided.",
        "",
        "## Success Criteria",
        success_criteria or "Maximize expected upside while reducing avoidable downside, regret, and relationship damage.",
        "",
        "## Known Risks",
        risks or "Unknown; agents should surface likely second-order consequences.",
        "",
        "## Simulation Requirement",
        (
            "Simulate likely stakeholder reactions, second-order effects, failure modes, "
            "and better alternatives. Produce evidence that helps Hermes decide whether "
            "to proceed, modify, delay, or ask the user for clarification."
        ),
    ])
    return "\n".join(sections).strip() + "\n"


def mirofish_simulate_decision(args: dict[str, Any], **_: Any) -> str:
    """High-level Hermes wrapper for low-cost MiroFish decision rehearsal."""

    try:
        phase = str(args.get("phase") or "bootstrap").strip().lower()
        if phase not in {"bootstrap", "simulate", "report", "full"}:
            return tool_error("phase must be one of: bootstrap, simulate, report, full", success=False)

        decision = str(args.get("decision") or "").strip()
        if not decision and phase in {"bootstrap", "full"}:
            return tool_error("decision is required for bootstrap/full decision simulation", success=False)

        project_id = str(args.get("project_id") or "").strip()
        graph_id = str(args.get("graph_id") or "").strip()
        simulation_id = str(args.get("simulation_id") or "").strip()
        project_name = str(args.get("project_name") or "").strip() or "Hermes decision simulation"
        platform = str(args.get("platform") or "parallel").strip().lower()
        max_rounds = int(args.get("max_rounds") or 3)
        parallel_profile_count = int(args.get("parallel_profile_count") or 3)
        enable_graph_memory_update = bool(args.get("enable_graph_memory_update", False))
        force = bool(args.get("force", False))

        steps: list[dict[str, Any]] = []

        if phase in {"bootstrap", "full"} and not project_id:
            seed = _decision_seed_markdown(args)
            requirement = (
                "Personal decision rehearsal for Hermes. "
                "Model stakeholder reactions, second-order effects, and recommended action path."
            )
            raw = _multipart_request(
                "/api/graph/ontology/generate",
                fields={
                    "simulation_requirement": requirement,
                    "project_name": project_name,
                    "additional_context": str(args.get("additional_context") or ""),
                },
                files=[("files", "hermes_decision_seed.md", seed.encode("utf-8"), "text/markdown")],
            )
            data = _data(raw)
            project_id = data.get("project_id") or project_id
            steps.append({
                "step": "ontology_generate",
                "endpoint": "/api/graph/ontology/generate",
                "success": _success(raw),
                "project_id": project_id,
                "mirofish": raw,
            })
            if not _success(raw):
                return tool_result(success=False, phase=phase, steps=steps, error=raw.get("error"))

        if phase in {"bootstrap", "full"} and project_id and not graph_id:
            raw = _request(
                "POST",
                "/api/graph/build",
                payload={
                    "project_id": project_id,
                    "graph_name": f"{project_name} graph",
                    "force": force,
                },
            )
            data = _data(raw)
            graph_id = data.get("graph_id") or graph_id
            steps.append({
                "step": "graph_build",
                "endpoint": "/api/graph/build",
                "success": _success(raw),
                "project_id": project_id,
                "graph_id": graph_id,
                "task_id": data.get("task_id") or raw.get("task_id"),
                "mirofish": raw,
            })
            if not _success(raw):
                return tool_result(success=False, phase=phase, steps=steps, error=raw.get("error"))

        if phase in {"simulate", "full"}:
            if not simulation_id:
                if not project_id:
                    return tool_error("project_id is required to create a simulation", success=False)
                create_payload = {
                    "project_id": project_id,
                    "enable_twitter": bool(args.get("enable_twitter", True)),
                    "enable_reddit": bool(args.get("enable_reddit", True)),
                }
                if graph_id:
                    create_payload["graph_id"] = graph_id
                raw = _request("POST", "/api/simulation/create", payload=create_payload)
                data = _data(raw)
                simulation_id = data.get("simulation_id") or simulation_id
                graph_id = data.get("graph_id") or graph_id
                steps.append({
                    "step": "simulation_create",
                    "endpoint": "/api/simulation/create",
                    "success": _success(raw),
                    "simulation_id": simulation_id,
                    "project_id": project_id,
                    "graph_id": graph_id,
                    "mirofish": raw,
                })
                if not _success(raw):
                    return tool_result(success=False, phase=phase, steps=steps, error=raw.get("error"))

            prepare_payload = {
                "simulation_id": simulation_id,
                "use_llm_for_profiles": bool(args.get("use_llm_for_profiles", True)),
                "parallel_profile_count": parallel_profile_count,
                "force_regenerate": bool(args.get("force_regenerate", False)),
            }
            if args.get("entity_types"):
                prepare_payload["entity_types"] = args.get("entity_types")
            raw = _request("POST", "/api/simulation/prepare", payload=prepare_payload)
            data = _data(raw)
            steps.append({
                "step": "simulation_prepare",
                "endpoint": "/api/simulation/prepare",
                "success": _success(raw),
                "simulation_id": simulation_id,
                "task_id": data.get("task_id"),
                "status": data.get("status"),
                "already_prepared": data.get("already_prepared"),
                "mirofish": raw,
            })
            if not _success(raw):
                return tool_result(success=False, phase=phase, steps=steps, error=raw.get("error"))

            if data.get("status") == "ready" or data.get("already_prepared"):
                raw = _request(
                    "POST",
                    "/api/simulation/start",
                    payload={
                        "simulation_id": simulation_id,
                        "platform": platform,
                        "max_rounds": max(1, max_rounds),
                        "enable_graph_memory_update": enable_graph_memory_update,
                        "force": force,
                    },
                )
                steps.append({
                    "step": "simulation_start",
                    "endpoint": "/api/simulation/start",
                    "success": _success(raw),
                    "simulation_id": simulation_id,
                    "max_rounds": max_rounds,
                    "platform": platform,
                    "mirofish": raw,
                })
            else:
                steps.append({
                    "step": "simulation_start",
                    "skipped": True,
                    "reason": "simulation preparation is async; poll prepare status before starting",
                    "next_tool": "mirofish_simulation_status",
                    "next_args": {
                        "status_type": "prepare",
                        "simulation_id": simulation_id,
                        "task_id": data.get("task_id"),
                    },
                })

        if phase == "report":
            if not simulation_id:
                return tool_error("simulation_id is required for report phase", success=False)
            raw = _request(
                "POST",
                "/api/report/generate",
                payload={
                    "simulation_id": simulation_id,
                    "force_regenerate": bool(args.get("force_regenerate", False)),
                },
            )
            data = _data(raw)
            steps.append({
                "step": "report_generate",
                "endpoint": "/api/report/generate",
                "success": _success(raw),
                "simulation_id": simulation_id,
                "task_id": data.get("task_id"),
                "report_id": data.get("report_id"),
                "status": data.get("status"),
                "mirofish": raw,
            })

        return tool_result(
            success=True,
            phase=phase,
            decision=decision,
            project_id=project_id,
            graph_id=graph_id,
            simulation_id=simulation_id,
            low_cost_defaults={
                "max_rounds": max_rounds,
                "parallel_profile_count": parallel_profile_count,
                "enable_graph_memory_update": enable_graph_memory_update,
            },
            steps=steps,
            recommendation_contract={
                "proceed": "Use when simulated upside is clear and risks are manageable.",
                "modify": "Use when a better variant dominates the original action.",
                "delay": "Use when information gaps or timing risks dominate.",
                "ask_user": "Use when missing private context changes the decision materially.",
            },
        )
    except Exception as exc:
        return _error_result(exc)


_PAYLOAD_SCHEMA = {
    "type": "object",
    "description": "Optional raw MiroFish request fields. Explicit top-level tool arguments override matching payload keys.",
    "additionalProperties": True,
}


registry.register(
    name="mirofish_simulate_decision",
    toolset="mirofish",
    schema={
        "name": "mirofish_simulate_decision",
        "description": (
            "High-level Hermes decision-rehearsal workflow over MiroFish. "
            "Use this to simulate likely stakeholder reactions, second-order effects, "
            "and action recommendations before Hermes acts. Defaults are low-cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["bootstrap", "simulate", "report", "full"],
                    "description": (
                        "bootstrap creates ontology/build task from decision text; "
                        "simulate creates/prepares/starts from an existing project/graph; "
                        "report starts report generation; full attempts bootstrap+simulate when IDs are available."
                    ),
                },
                "decision": {"type": "string", "description": "The action, plan, or decision Hermes should rehearse."},
                "context": {"type": "string", "description": "Relevant user memory, evolution genes, session context, or constraints."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decision options or variants to compare.",
                },
                "stakeholders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "People, groups, future selves, or systems affected by the decision.",
                },
                "success_criteria": {"type": "string", "description": "What a good outcome means for the user."},
                "risks": {"type": "string", "description": "Known concerns, failure modes, or downside scenarios."},
                "additional_context": {"type": "string", "description": "Extra context passed to MiroFish ontology generation."},
                "project_name": {"type": "string", "description": "Optional MiroFish project name."},
                "project_id": {"type": "string", "description": "Existing MiroFish project id."},
                "graph_id": {"type": "string", "description": "Existing MiroFish graph id."},
                "simulation_id": {"type": "string", "description": "Existing MiroFish simulation id."},
                "platform": {
                    "type": "string",
                    "enum": ["twitter", "reddit", "parallel"],
                    "description": "Simulation platform. Default: parallel.",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Maximum simulation rounds. Default: 3 for low-cost decision rehearsal.",
                },
                "parallel_profile_count": {
                    "type": "integer",
                    "description": "Parallel profile generation count. Default: 3 for low-cost decision rehearsal.",
                },
                "enable_twitter": {"type": "boolean", "description": "Enable Twitter-style channel when creating a simulation."},
                "enable_reddit": {"type": "boolean", "description": "Enable Reddit-style channel when creating a simulation."},
                "use_llm_for_profiles": {"type": "boolean", "description": "Whether MiroFish should generate LLM profiles. Default true."},
                "enable_graph_memory_update": {
                    "type": "boolean",
                    "description": "Whether MiroFish writes simulation activity back to graph memory. Default false.",
                },
                "force": {"type": "boolean", "description": "Force rebuild/restart where supported."},
                "force_regenerate": {"type": "boolean", "description": "Force regenerate prepared simulation/report where supported."},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional entity types to use for agent profiles.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    handler=mirofish_simulate_decision,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_build_graph",
    toolset="mirofish",
    schema={
        "name": "mirofish_build_graph",
        "description": "Start a MiroFish graph build for an existing project.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "MiroFish project id."},
                "graph_name": {"type": "string", "description": "Optional graph display name."},
                "chunk_size": {"type": "integer", "description": "Optional document chunk size."},
                "chunk_overlap": {"type": "integer", "description": "Optional document chunk overlap."},
                "force": {"type": "boolean", "description": "Force rebuild if a graph task already exists."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    handler=mirofish_build_graph,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_prepare_simulation",
    toolset="mirofish",
    schema={
        "name": "mirofish_prepare_simulation",
        "description": "Prepare a MiroFish simulation. If simulation_id is omitted, creates one from project_id first.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Existing simulation id."},
                "project_id": {"type": "string", "description": "Project id used when creating a simulation first."},
                "graph_id": {"type": "string", "description": "Graph id used when creating a simulation first."},
                "enable_twitter": {"type": "boolean", "description": "Enable Twitter-style simulation channel."},
                "enable_reddit": {"type": "boolean", "description": "Enable Reddit-style simulation channel."},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional entity types to use for agent profiles.",
                },
                "use_llm_for_profiles": {"type": "boolean", "description": "Whether MiroFish should generate LLM profiles."},
                "parallel_profile_count": {"type": "integer", "description": "Parallel profile generation count."},
                "force_regenerate": {"type": "boolean", "description": "Force regeneration of prepared simulation files."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    handler=mirofish_prepare_simulation,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_run_simulation",
    toolset="mirofish",
    schema={
        "name": "mirofish_run_simulation",
        "description": "Start a prepared MiroFish simulation run.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Simulation id to run."},
                "platform": {"type": "string", "description": "Simulation platform: twitter, reddit, or parallel."},
                "max_rounds": {"type": "integer", "description": "Optional maximum number of simulation rounds."},
                "enable_graph_memory_update": {"type": "boolean", "description": "Whether MiroFish should write activity back to its graph memory."},
                "force": {"type": "boolean", "description": "Force restart a running or completed simulation."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": ["simulation_id"],
            "additionalProperties": False,
        },
    },
    handler=mirofish_run_simulation,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_simulation_status",
    toolset="mirofish",
    schema={
        "name": "mirofish_simulation_status",
        "description": "Fetch MiroFish simulation run, run_detail, prepare, or environment status.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Simulation id."},
                "task_id": {"type": "string", "description": "Task id for prepare status."},
                "status_type": {
                    "type": "string",
                    "enum": ["run", "run_detail", "prepare", "env"],
                    "description": "Status endpoint to query.",
                },
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    handler=mirofish_simulation_status,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_generate_report",
    toolset="mirofish",
    schema={
        "name": "mirofish_generate_report",
        "description": "Start MiroFish report generation for a simulation.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Simulation id to analyze."},
                "force_regenerate": {"type": "boolean", "description": "Force report regeneration."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": ["simulation_id"],
            "additionalProperties": False,
        },
    },
    handler=mirofish_generate_report,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_get_report",
    toolset="mirofish",
    schema={
        "name": "mirofish_get_report",
        "description": "Fetch a MiroFish report or report-generation status.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string", "description": "Report id to fetch."},
                "simulation_id": {"type": "string", "description": "Simulation id whose report should be fetched."},
                "task_id": {"type": "string", "description": "Report generation task id for status checks."},
                "status_only": {"type": "boolean", "description": "Fetch report-generation status instead of report content."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    handler=mirofish_get_report,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_interview_agents",
    toolset="mirofish",
    schema={
        "name": "mirofish_interview_agents",
        "description": "Interview MiroFish simulation agents individually, in batch, or all at once.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Simulation id containing the agents."},
                "mode": {
                    "type": "string",
                    "enum": ["single", "batch", "all"],
                    "description": "Interview mode.",
                },
                "agent_id": {"type": "integer", "description": "Agent id for single mode."},
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Agent ids for batch mode.",
                },
                "interviews": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": "Raw MiroFish batch interview objects with agent_id, prompt, optional platform.",
                },
                "question": {"type": "string", "description": "Question for the simulated agent or agents."},
                "prompt": {"type": "string", "description": "Alternative prompt field for compatible MiroFish versions."},
                "platform": {"type": "string", "description": "Optional interview platform: twitter or reddit."},
                "timeout": {"type": "integer", "description": "Optional interview timeout in seconds."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": ["simulation_id"],
            "additionalProperties": False,
        },
    },
    handler=mirofish_interview_agents,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)


registry.register(
    name="mirofish_close_env",
    toolset="mirofish",
    schema={
        "name": "mirofish_close_env",
        "description": "Close a MiroFish simulation environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_id": {"type": "string", "description": "Simulation id whose environment should close."},
                "timeout": {"type": "integer", "description": "Optional close timeout in seconds."},
                "payload": _PAYLOAD_SCHEMA,
            },
            "required": ["simulation_id"],
            "additionalProperties": False,
        },
    },
    handler=mirofish_close_env,
    check_fn=check_mirofish_requirements,
    requires_env=["MIROFISH_BASE_URL"],
)
