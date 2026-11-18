"""Virtual clusters: what they accept, refuse, and report.

The important property is that a cluster is not a bigger GPU. Capacity is
reported as aggregate across separate machines, and validation says what it can
and cannot decide: memory is a deploy-time question, because a cluster cannot
know which model it will host.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from meshweave.config import CoordinatorSettings
from meshweave.core import Heartbeat, NodeStatus, utcnow
from meshweave.coordinator import app as coordinator_app
from tests.test_deployments import QWEN_05B, node

ENROLL = "test-enroll"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    async def fake_config(repo: str) -> str:
        return json.dumps(QWEN_05B)

    monkeypatch.setattr(coordinator_app, "fetch_model_config", fake_config)
    settings = CoordinatorSettings(
        port=8794,
        deployments_file=tmp_path / "deployments.json",
        clusters_file=tmp_path / "clusters.json",
        enrollment_token=ENROLL,
    )
    app = coordinator_app.create_app(settings)
    client = TestClient(app)
    state = client.app.state.mw
    for record in (
        node("alpha", mapped={10100: 41001}),
        node("beta", mapped={10100: 41002}),
    ):
        # A real host heartbeats, and a cluster may only include hosts that do.
        state.registry.register(record.registration)
        state.registry.record_heartbeat(
            Heartbeat(
                node_id=record.node_id,
                sent_at=utcnow(),
                inventory=record.inventory,
                active_workloads=[],
            )
        )
    return client


def _create(client: TestClient, name: str, nodes: list[str]):
    return client.post("/v1/clusters", json={"name": name, "node_ids": nodes})


# --------------------------------------------------------------------------- #
# creation
# --------------------------------------------------------------------------- #


def test_two_nodes_form_one_named_cluster(client: TestClient) -> None:
    response = _create(client, "VDC-001", ["alpha", "beta"])
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["cluster_id"] == "F16-VDC-001"
    assert body["members"] == ["alpha", "beta"]
    assert body["health"] == "HEALTHY"
    capacity = body["capacity"]
    assert capacity["gpus"] == 2
    assert capacity["vram_total_gib"] > 0
    assert capacity["kind"] == "aggregate distributed capacity"
    assert "not one pool" in capacity["note"]


def test_a_node_without_a_published_port_cannot_join_a_cluster(client: TestClient) -> None:
    state = client.app.state.mw
    record = node("lonely")
    state.registry.register(record.registration)
    state.registry.record_heartbeat(
        Heartbeat(node_id="lonely", sent_at=utcnow(), inventory=record.inventory, active_workloads=[])
    )

    response = _create(client, "VDC-BAD", ["alpha", "lonely"])
    assert response.status_code == 422
    assert any("lonely" in problem and "TCP port" in problem for problem in response.json()["detail"])


def test_a_node_with_an_old_agent_cannot_join(client: TestClient) -> None:
    state = client.app.state.mw
    record = node("old", mapped={10100: 41003}, agent_version="0.3.0")
    state.registry.register(record.registration)
    state.registry.record_heartbeat(
        Heartbeat(node_id="old", sent_at=utcnow(), inventory=record.inventory, active_workloads=[])
    )

    response = _create(client, "VDC-OLD", ["alpha", "old"])
    assert response.status_code == 422
    assert "0.3.0" in json.dumps(response.json()["detail"])


def test_an_unmeasured_link_is_a_warning_not_a_refusal(client: TestClient) -> None:
    response = _create(client, "VDC-WARN", ["alpha", "beta"])
    assert response.status_code == 200
    assert any("not been measured" in w for w in response.json()["warnings"])


def test_duplicate_names_and_unknown_nodes_are_refused(client: TestClient) -> None:
    assert _create(client, "VDC-001", ["alpha", "beta"]).status_code == 200
    again = _create(client, "VDC-001", ["alpha", "beta"])
    assert again.status_code == 409

    unknown = client.post("/v1/clusters", json={"name": "X", "node_ids": ["alpha", "nobody"]})
    assert unknown.status_code == 404


# --------------------------------------------------------------------------- #
# status follows the members
# --------------------------------------------------------------------------- #


def test_a_cluster_is_degraded_when_a_member_goes_silent(client: TestClient) -> None:
    _create(client, "VDC-001", ["alpha", "beta"])
    state = client.app.state.mw
    state.registry.get("beta").status = NodeStatus.UNREACHABLE

    status = client.get("/v1/clusters/F16-VDC-001").json()
    assert status["health"] == "DEGRADED"
    assert status["member_state"]["beta"] == "OFFLINE"
    # Capacity must shrink with the member, not stay optimistic.
    assert status["capacity"]["gpus"] == 1


def test_a_cluster_is_offline_when_no_member_is_reachable(client: TestClient) -> None:
    _create(client, "VDC-001", ["alpha", "beta"])
    state = client.app.state.mw
    for node_id in ("alpha", "beta"):
        state.registry.get(node_id).status = NodeStatus.UNREACHABLE

    assert client.get("/v1/clusters/F16-VDC-001").json()["health"] == "OFFLINE"


# --------------------------------------------------------------------------- #
# deployment through a cluster
# --------------------------------------------------------------------------- #


def test_a_model_can_be_deployed_to_a_cluster(client: TestClient) -> None:
    _create(client, "VDC-001", ["alpha", "beta"])

    response = client.post(
        "/v1/deployments",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "cluster_id": "F16-VDC-001"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["node_id"] for s in body["stages"]] == ["alpha", "beta"]
    assert [s["layers"] for s in body["stages"]] == ["[0,12)", "[12,24)"]


def test_node_ids_that_contradict_the_cluster_are_rejected(client: TestClient) -> None:
    _create(client, "VDC-001", ["alpha", "beta"])
    response = client.post(
        "/v1/deployments",
        json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "cluster_id": "F16-VDC-001",
            "node_ids": ["alpha"],
        },
    )
    assert response.status_code == 400
    assert "contradict" in response.json()["detail"]


def test_validate_reflects_what_members_report_now(client: TestClient) -> None:
    _create(client, "VDC-001", ["alpha", "beta"])
    state = client.app.state.mw
    state.registry.get("beta").status = NodeStatus.UNREACHABLE

    validated = client.post("/v1/clusters/F16-VDC-001/validate").json()
    assert validated["ok"] is False
    assert any("beta" in problem and "unreachable" in problem for problem in validated["errors"])