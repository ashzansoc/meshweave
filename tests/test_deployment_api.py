"""The deployment HTTP surface, exercised through the real app.

The model config is fetched from huggingface.co in production; here it is
patched so the routes can be tested without the network. Everything else -- node
lookup, planning, refusal, the work queue -- runs for real.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from meshweave.config import CoordinatorSettings
from meshweave.coordinator import app as coordinator_app
from tests.test_deployments import QWEN_05B, node


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    async def fake_config(repo: str) -> str:
        return json.dumps(QWEN_05B)

    monkeypatch.setattr(coordinator_app, "fetch_model_config", fake_config)
    settings = CoordinatorSettings(
        port=8799, deployments_file=tmp_path / "deployments.json", enrollment_token="test-enroll"
    )
    return TestClient(coordinator_app.create_app(settings))


def _add_nodes(client: TestClient, *records) -> None:
    """Put nodes in the registry directly; HTTP registration is tested elsewhere."""
    state = client.app.state.mw
    for record in records:
        state.registry.register(record.registration)
        state.registry.get(record.node_id).inventory = record.inventory


def test_deployment_is_refused_when_a_node_publishes_no_port(client: TestClient) -> None:
    _add_nodes(client, node("alpha", mapped={10100: 41000}), node("beta", mapped={}))

    response = client.post(
        "/v1/deployments",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha", "beta"]},
    )
    assert response.status_code == 409
    assert "beta" in response.json()["detail"]


def test_deployment_plans_layers_and_queues_a_stage(client: TestClient) -> None:
    _add_nodes(client, node("alpha", mapped={10100: 41000}), node("beta", mapped={10100: 42000}))

    response = client.post(
        "/v1/deployments",
        json={
            "model": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
            "node_ids": ["alpha", "beta"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["layers"] for s in body["stages"]] == ["[0,12)", "[12,24)"]
    assert body["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

    work = client.get("/v1/agents/alpha/work", headers={"authorization": "Bearer test-enroll"})
    assert work.status_code == 200
    commands = work.json()["deployment_commands"]
    assert [c["kind"] for c in commands] == ["run_stage"]
    assert commands[0]["rank"] == 0
    assert commands[0]["next"] == "203.0.113.9:42000"

    # Polling again before the stage reports running must not duplicate it: the
    # agent starts one process, not one per poll.
    again = client.get("/v1/agents/alpha/work", headers={"authorization": "Bearer test-enroll"})
    assert [c["kind"] for c in again.json()["deployment_commands"]] == ["run_stage"]


def test_prompts_wait_until_every_stage_is_running(client: TestClient) -> None:
    _add_nodes(client, node("alpha", mapped={10100: 41000}), node("beta", mapped={10100: 42000}))

    created = client.post(
        "/v1/deployments",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha", "beta"]},
    ).json()
    deployment_id = created["deployment_id"]

    early = client.post(f"/v1/deployments/{deployment_id}/prompt", json={"prompt": "hello"})
    assert early.status_code == 409

    for node_id in ("alpha", "beta"):
        client.post(
            f"/v1/deployments/{deployment_id}/nodes/{node_id}/status",
            json={"phase": "running"},
        )

    status = client.get(f"/v1/deployments/{deployment_id}").json()
    assert status["status"] == "RUNNING"

    sent = client.post(f"/v1/deployments/{deployment_id}/prompt", json={"prompt": "hello"})
    assert sent.status_code == 200

    work = client.get("/v1/agents/alpha/work", headers={"authorization": "Bearer test-enroll"}).json()
    prompts = [c for c in work["deployment_commands"] if c["kind"] == "run_prompt"]
    assert len(prompts) == 1
    assert prompts[0]["prompt"] == "hello"
    assert prompts[0]["control"].startswith("/run/meshweave")


def test_a_stage_that_never_serves_eventually_fails_the_deployment(client: TestClient) -> None:
    _add_nodes(client, node("solo"))

    created = client.post(
        "/v1/deployments", json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["solo"]}
    ).json()

    # A stage that dies before serving is retried a bounded number of times, then
    # the deployment fails with the reason rather than retrying forever.
    for attempt in range(3):
        client.post(
            f"/v1/deployments/{created['deployment_id']}/nodes/solo/status",
            json={"phase": "starting"},
        )
        client.post(
            f"/v1/deployments/{created['deployment_id']}/nodes/solo/status",
            json={"phase": "failed", "detail": "torch.cuda.OutOfMemoryError"},
        )
        status = client.get(f"/v1/deployments/{created['deployment_id']}").json()
        if attempt < 2:
            assert status["status"] != "FAILED"

    assert status["status"] == "FAILED"
    assert "OutOfMemory" in (status["message"] or "")


def test_an_unsupported_architecture_is_refused_by_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def llama_config(repo: str) -> str:
        return json.dumps({**QWEN_05B, "architectures": ["LlamaForCausalLM"]})

    monkeypatch.setattr(coordinator_app, "fetch_model_config", llama_config)
    _add_nodes(client, node("solo"))

    response = client.post(
        "/v1/deployments", json={"model": "meta-llama/Llama-3.2-1B", "node_ids": ["solo"]}
    )
    assert response.status_code == 422
    assert "LlamaForCausalLM" in response.json()["detail"]


def test_stop_then_forget(client: TestClient) -> None:
    _add_nodes(client, node("solo"))
    created = client.post(
        "/v1/deployments", json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["solo"]}
    ).json()
    deployment_id = created["deployment_id"]

    stopped = client.post(f"/v1/deployments/{deployment_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "STOPPED"

    work = client.get("/v1/agents/solo/work", headers={"authorization": "Bearer test-enroll"}).json()
    assert [c["kind"] for c in work["deployment_commands"]] == ["stop_stage"]

    removed = client.delete(f"/v1/deployments/{deployment_id}")
    assert removed.status_code == 200
    assert client.get(f"/v1/deployments/{deployment_id}").status_code == 404


def test_a_stage_that_dies_while_a_peer_downloads_is_retried(client: TestClient) -> None:
    """A 1 GiB checkpoint takes minutes on these links. The first stage must not
    fail the whole deployment because the second had not finished fetching."""
    state = client.app.state.mw
    for record in (node("alpha", mapped={10100: 41000}), node("beta", mapped={10100: 41002})):
        state.registry.register(record.registration)
        state.registry.get(record.node_id).inventory = record.inventory

    created = client.post(
        "/v1/deployments",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha", "beta"]},
    ).json()
    deployment_id = created["deployment_id"]

    report = lambda phase, detail=None: client.post(
        f"/v1/deployments/{deployment_id}/nodes/alpha/status",
        json={"phase": phase, "detail": detail},
    )

    report("starting")
    report("failed", "ConnectionRefusedError: Connection refused")
    status = client.get(f"/v1/deployments/{deployment_id}").json()
    assert status["status"] != "FAILED"
    assert status["stages"][0]["phase"] == "restarting"

    # Three launches, each failing, then it gives up with the reason attached.
    for _ in range(3):
        report("starting")
        report("failed", "ConnectionRefusedError: Connection refused")
    status = client.get(f"/v1/deployments/{deployment_id}").json()
    assert status["status"] == "FAILED"
    assert "Connection refused" in (status["stages"][0]["detail"] or "")


def test_stopping_a_deployment_is_not_undone_by_a_status_report(client: TestClient) -> None:
    """The agent reports 'stopped' after a stop; the roll-up used to read that as
    loading and re-queue the stage, restarting work that had been cancelled."""
    state = client.app.state.mw
    for record in (node("alpha", mapped={10100: 41000}), node("beta", mapped={10100: 41002})):
        state.registry.register(record.registration)
        state.registry.get(record.node_id).inventory = record.inventory

    created = client.post(
        "/v1/deployments",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["alpha", "beta"]},
    ).json()
    deployment_id = created["deployment_id"]

    client.post(f"/v1/deployments/{deployment_id}/stop")
    for node_id in ("alpha", "beta"):
        client.post(
            f"/v1/deployments/{deployment_id}/nodes/{node_id}/status",
            json={"phase": "stopped"},
        )

    status = client.get(f"/v1/deployments/{deployment_id}").json()
    assert status["status"] == "STOPPED"

    work = client.get(
        "/v1/agents/alpha/work", headers={"authorization": "Bearer test-enroll"}
    ).json()
    kinds = [c["kind"] for c in work["deployment_commands"]]
    assert "run_stage" not in kinds


def test_execution_evidence_survives_the_result_endpoint(client: TestClient) -> None:
    """The runner measures the pipeline round trip, the activation bytes and the
    GPU during generation. If the request model drops them, a distributed request
    reports nothing about what it actually did."""
    state = client.app.state.mw
    record = node("solo", mapped={10100: 41000})
    state.registry.register(record.registration)
    state.registry.get("solo").inventory = record.inventory

    created = client.post(
        "/v1/deployments", json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "node_ids": ["solo"]}
    ).json()

    response = client.post(
        f"/v1/deployments/{created['deployment_id']}/result",
        json={
            "prompt": "hi",
            "text": "hello",
            "output_tokens": 4,
            "prompt_id": "p-1",
            "stage_roundtrip_ms": 452.3,
            "activation_bytes_per_token": 1792.0,
            "activation_bytes_sent": 7168,
            "world": 2,
            "rank": 0,
            "gpu_during_generation": {"samples": 12, "util_pct_max": 97.0},
        },
    )
    assert response.status_code == 200

    detail = client.get(f"/v1/deployments/{created['deployment_id']}").json()
    result = detail["results"][-1]
    assert result["stage_roundtrip_ms"] == 452.3
    assert result["activation_bytes_per_token"] == 1792.0
    assert result["world"] == 2
    assert result["gpu_during_generation"]["util_pct_max"] == 97.0
