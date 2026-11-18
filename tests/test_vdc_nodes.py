"""Node credential, reaping, availability and telemetry history.

These four together are what makes the fleet manageable rather than merely
visible: a node that dies must be noticed on a timer, a node that is alive must
be distinguishable from a node that is free, and only a node holding a credential
may be handed work.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meshweave.communication.auth import issue_node_token, verify_node_token
from meshweave.config import CoordinatorSettings
from meshweave.core import Heartbeat, NodeStatus, utcnow
from meshweave.coordinator import app as coordinator_app
from tests.test_deployments import QWEN_05B, node

SECRET_ENROLL = "test-enroll"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    async def fake_config(repo: str) -> str:
        return json.dumps(QWEN_05B)

    monkeypatch.setattr(coordinator_app, "fetch_model_config", fake_config)
    settings = CoordinatorSettings(
        port=8796, deployments_file=tmp_path / "deployments.json", enrollment_token=SECRET_ENROLL
    )
    return TestClient(coordinator_app.create_app(settings))


def _register(client: TestClient, record) -> str:
    """Register through HTTP so the node receives a real token."""
    response = client.post(
        "/v1/nodes/register",
        headers={"x-meshweave-enrollment": SECRET_ENROLL},
        json={
            "node_id": record.node_id,
            "hostname": record.node_id,
            "agent_version": "0.4.0",
            "provider": "manual",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["node_token"]


def _heartbeat(client: TestClient, record) -> None:
    payload = record.inventory.model_dump(mode="json")
    response = client.post(
        f"/v1/nodes/{record.node_id}/heartbeat",
        json={"node_id": record.node_id, "sent_at": utcnow().isoformat(), "inventory": payload},
    )
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# node tokens
# --------------------------------------------------------------------------- #


def test_a_node_token_names_exactly_one_node() -> None:
    secret = b"s" * 32
    token = issue_node_token("alpha", secret)
    assert verify_node_token(token, secret) == "alpha"
    assert verify_node_token(token, b"t" * 32) is None
    assert verify_node_token(token[:-4] + "AAAA", secret) is None
    assert verify_node_token("", secret) is None


def test_registration_hands_the_agent_a_working_credential(client: TestClient) -> None:
    token = _register(client, node("alpha"))
    work = client.get("/v1/agents/alpha/work", headers={"authorization": f"Bearer {token}"})
    assert work.status_code == 200


def test_work_is_refused_without_a_credential(client: TestClient) -> None:
    _register(client, node("alpha"))
    assert client.get("/v1/agents/alpha/work").status_code == 401
    bad = client.get("/v1/agents/alpha/work", headers={"authorization": "Bearer nonsense"})
    assert bad.status_code == 401
    assert "register" in bad.json()["detail"]


def test_one_node_cannot_ask_for_another_nodes_work(client: TestClient) -> None:
    _register(client, node("alpha"))
    beta_token = _register(client, node("beta"))
    stolen = client.get(
        "/v1/agents/alpha/work", headers={"authorization": f"Bearer {beta_token}"}
    )
    assert stolen.status_code == 401
    assert "beta" in stolen.json()["detail"]


def test_the_shared_enrollment_token_still_works_during_an_upgrade(
    client: TestClient,
) -> None:
    """Agents older than node tokens send the enrollment token. They must keep
    working until they are updated, or an upgrade takes the fleet down."""
    _register(client, node("alpha"))
    work = client.get(
        "/v1/agents/alpha/work", headers={"authorization": f"Bearer {SECRET_ENROLL}"}
    )
    assert work.status_code == 200


# --------------------------------------------------------------------------- #
# reaping
# --------------------------------------------------------------------------- #


def test_reaping_marks_a_silent_node_and_persists_it(client: TestClient) -> None:
    record = node("alpha")
    _register(client, record)
    _heartbeat(client, record)

    state = client.app.state.mw
    stored = state.registry.get("alpha")
    assert stored.status == NodeStatus.HEALTHY

    later = utcnow().timestamp() + state.settings.heartbeat_grace_s + 5
    changed = state.registry.reap(now=later)
    assert changed == ["alpha"]
    assert state.registry.get("alpha").status == NodeStatus.UNREACHABLE

    # The snapshot must carry it, or a restart resurrects a dead node as healthy.
    # Read the path the suite was pointed at, not the live coordinator's file.
    snapshot_path = os.environ["MESHWEAVE_REGISTRY_SNAPSHOT"]
    snapshot = json.loads(Path(snapshot_path).read_text())
    statuses = {n["node_id"]: n["status"] for n in snapshot["nodes"]}
    assert statuses["alpha"] == "UNREACHABLE"


def test_a_reaped_node_fails_the_deployment_it_was_hosting(client: TestClient) -> None:
    alpha = node("alpha", mapped={10100: 41000})
    _register(client, alpha)
    _heartbeat(client, alpha)

    created = client.post(
        "/v1/deployments", json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha"]}
    )
    assert created.status_code == 200, created.text
    deployment_id = created.json()["deployment_id"]

    state = client.app.state.mw
    state.deployments.get(deployment_id).node_status["alpha"] = {"phase": "running"}
    affected = state.deployments.mark_unreachable(["alpha"])

    assert affected == [deployment_id]
    summary = client.get(f"/v1/deployments/{deployment_id}").json()
    assert summary["status"] == "FAILED"
    assert "stopped responding" in summary["message"]


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #


def test_availability_separates_alive_from_free(client: TestClient) -> None:
    alpha = node("alpha", mapped={10100: 41000})
    _register(client, alpha)
    _heartbeat(client, alpha)

    def availability() -> str:
        rows = client.get("/v1/readiness").json()["nodes"]
        return next(r["availability"] for r in rows if r["node_id"] == "alpha")

    assert availability() == "AVAILABLE"

    client.post(
        "/v1/deployments", json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha"]}
    )
    state = client.app.state.mw
    deployment = state.deployments.all()[0]
    deployment.node_status["alpha"] = {"phase": "running"}
    assert availability() == "BUSY"

    state.registry.get("alpha").status = NodeStatus.UNREACHABLE
    assert availability() == "OFFLINE"

    state.registry.get("alpha").status = NodeStatus.DRAINING
    assert availability() == "MAINTENANCE"


# --------------------------------------------------------------------------- #
# telemetry history
# --------------------------------------------------------------------------- #


def test_each_heartbeat_appends_a_sample(client: TestClient) -> None:
    alpha = node("alpha", mapped={10100: 41000})
    _register(client, alpha)
    for _ in range(3):
        _heartbeat(client, alpha)

    series = client.get("/v1/nodes/alpha/series").json()
    assert series["count"] == 3
    sample = series["samples"][-1]
    assert sample["vram_total_bytes"] > 0
    assert "gpu_util_pct" in sample and "temperature_c" in sample
    assert "in memory only" in series["retention"]


def test_series_unknown_node_is_a_404(client: TestClient) -> None:
    assert client.get("/v1/nodes/nobody/series").status_code == 404
