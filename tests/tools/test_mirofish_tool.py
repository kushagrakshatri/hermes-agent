import io
import json
import urllib.error
from urllib.parse import urlparse

import tools.mirofish_tool as mirofish


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _decode_request(request):
    data = request.data.decode("utf-8") if request.data else None
    return {
        "method": request.get_method(),
        "url": request.full_url,
        "path": urlparse(request.full_url).path,
        "payload": json.loads(data) if data else None,
        "headers": dict(request.header_items()),
    }


def _decode_request_maybe_json(request):
    data = request.data.decode("utf-8") if request.data else None
    payload = None
    content_type = dict(request.header_items()).get("Content-type", "")
    if data and "application/json" in content_type:
        payload = json.loads(data)
    return {
        "method": request.get_method(),
        "url": request.full_url,
        "path": urlparse(request.full_url).path,
        "body": data,
        "payload": payload,
        "headers": dict(request.header_items()),
    }


def test_check_mirofish_requirements_accepts_env_or_twin_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MIROFISH_BASE_URL", raising=False)

    assert mirofish.check_mirofish_requirements() is False

    monkeypatch.setenv("MIROFISH_BASE_URL", "http://localhost:5001")
    assert mirofish.check_mirofish_requirements() is True

    monkeypatch.delenv("MIROFISH_BASE_URL", raising=False)
    (tmp_path / "twin.json").write_text(
        json.dumps({"mirofish": {"enabled": True, "base_url": "http://configured"}}),
        encoding="utf-8",
    )
    assert mirofish.check_mirofish_requirements() is True

    (tmp_path / "twin.json").write_text(
        json.dumps({"mirofish": {"enabled": False, "base_url": "http://configured"}}),
        encoding="utf-8",
    )
    assert mirofish.check_mirofish_requirements() is False


def test_run_simulation_posts_to_start(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local/")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((_decode_request(request), timeout))
        return FakeResponse({"success": True, "data": {"simulation_id": "sim_1"}})

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(
        mirofish.mirofish_run_simulation(
            {"simulation_id": "sim_1", "platform": "parallel", "max_rounds": 3}
        )
    )

    assert result["success"] is True
    assert result["endpoint"] == "/api/simulation/start"
    assert calls[0][0]["method"] == "POST"
    assert calls[0][0]["url"] == "http://mirofish.local/api/simulation/start"
    assert calls[0][0]["payload"] == {
        "simulation_id": "sim_1",
        "platform": "parallel",
        "max_rounds": 3,
    }
    assert calls[0][1] == 30.0


def test_prepare_simulation_creates_then_prepares(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")
    responses = [
        {"success": True, "data": {"simulation_id": "sim_created"}},
        {"success": True, "data": {"simulation_id": "sim_created", "task_id": "task_1"}},
    ]
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(_decode_request(request))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(
        mirofish.mirofish_prepare_simulation(
            {
                "project_id": "proj_1",
                "graph_id": "graph_1",
                "entity_types": ["Person"],
                "force_regenerate": True,
            }
        )
    )

    assert result["success"] is True
    assert result["simulation_id"] == "sim_created"
    assert result["task_id"] == "task_1"
    assert [call["path"] for call in calls] == [
        "/api/simulation/create",
        "/api/simulation/prepare",
    ]
    assert calls[0]["payload"] == {"project_id": "proj_1", "graph_id": "graph_1"}
    assert calls[1]["payload"] == {
        "entity_types": ["Person"],
        "force_regenerate": True,
        "simulation_id": "sim_created",
    }


def test_simulation_status_detail_uses_get_endpoint(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(_decode_request(request))
        return FakeResponse({"success": True, "data": {"runner_status": "running"}})

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(
        mirofish.mirofish_simulation_status(
            {"simulation_id": "sim 1", "status_type": "run_detail"}
        )
    )

    assert result["success"] is True
    assert calls[0]["method"] == "GET"
    assert calls[0]["payload"] is None
    assert calls[0]["path"] == "/api/simulation/sim%201/run-status/detail"


def test_batch_interview_builds_mirofish_interviews_payload(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(_decode_request(request))
        return FakeResponse({"success": True, "data": {"interviews_count": 2}})

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(
        mirofish.mirofish_interview_agents(
            {
                "simulation_id": "sim_1",
                "mode": "batch",
                "agent_ids": [0, 1],
                "question": "What changed?",
                "platform": "twitter",
            }
        )
    )

    assert result["success"] is True
    assert calls[0]["path"] == "/api/simulation/interview/batch"
    assert calls[0]["payload"] == {
        "simulation_id": "sim_1",
        "platform": "twitter",
        "interviews": [
            {"agent_id": 0, "prompt": "What changed?"},
            {"agent_id": 1, "prompt": "What changed?"},
        ],
    }


def test_http_error_returns_structured_tool_error(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "bad request"}'),
        )

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(mirofish.mirofish_build_graph({"project_id": "proj_1"}))

    assert result["success"] is False
    assert result["error"] == "bad request"
    assert result["status_code"] == 400
    assert result["body"] == {"error": "bad request"}


def test_simulate_decision_bootstrap_creates_seed_project_and_graph(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")
    calls = []

    def fake_urlopen(request, timeout):
        call = _decode_request_maybe_json(request)
        calls.append(call)
        if call["path"] == "/api/graph/ontology/generate":
            assert "multipart/form-data" in call["headers"]["Content-type"]
            assert "Should I ship this feature?" in call["body"]
            return FakeResponse({"success": True, "data": {"project_id": "proj_decision"}})
        if call["path"] == "/api/graph/build":
            return FakeResponse({"success": True, "data": {"task_id": "task_graph"}})
        raise AssertionError(call["path"])

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(mirofish.mirofish_simulate_decision({
        "phase": "bootstrap",
        "decision": "Should I ship this feature?",
        "options": ["ship", "delay"],
        "stakeholders": ["user", "customer"],
    }))

    assert result["success"] is True
    assert result["project_id"] == "proj_decision"
    assert [call["path"] for call in calls] == [
        "/api/graph/ontology/generate",
        "/api/graph/build",
    ]
    assert calls[1]["payload"]["project_id"] == "proj_decision"
    assert result["steps"][1]["task_id"] == "task_graph"


def test_simulate_decision_simulate_uses_low_cost_defaults(monkeypatch):
    monkeypatch.setenv("MIROFISH_BASE_URL", "http://mirofish.local")
    calls = []
    responses = {
        "/api/simulation/create": {"success": True, "data": {"simulation_id": "sim_1", "graph_id": "graph_1"}},
        "/api/simulation/prepare": {"success": True, "data": {"status": "ready", "already_prepared": True}},
        "/api/simulation/start": {"success": True, "data": {"runner_status": "running"}},
    }

    def fake_urlopen(request, timeout):
        call = _decode_request_maybe_json(request)
        calls.append(call)
        return FakeResponse(responses[call["path"]])

    monkeypatch.setattr(mirofish.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(mirofish.mirofish_simulate_decision({
        "phase": "simulate",
        "project_id": "proj_1",
        "graph_id": "graph_1",
    }))

    assert result["success"] is True
    assert result["simulation_id"] == "sim_1"
    assert [call["path"] for call in calls] == [
        "/api/simulation/create",
        "/api/simulation/prepare",
        "/api/simulation/start",
    ]
    assert calls[1]["payload"]["parallel_profile_count"] == 3
    assert calls[2]["payload"]["max_rounds"] == 3
    assert calls[2]["payload"]["enable_graph_memory_update"] is False
