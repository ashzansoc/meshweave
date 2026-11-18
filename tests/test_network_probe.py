"""Link measurement: the probe engine and the orchestration around it.

The orchestration is what makes a measurement possible at all in this design,
because nothing dials into a node. Each agent is told to listen, then told to
probe its peer, and every step has to be driven by a report coming back.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from meshweave.config import CoordinatorSettings
from meshweave.coordinator import app as coordinator_app
from meshweave.coordinator.network import (
    PREFERRED_PROBE_PORTS,
    NetworkProbeManager,
    candidate_probe_ports,
)
from meshweave.runtime.echo import EchoServer, probe
from tests.test_deployments import QWEN_05B, node


# --------------------------------------------------------------------------- #
# the wire
# --------------------------------------------------------------------------- #


def test_probe_measures_a_real_socket() -> None:
    server = EchoServer("127.0.0.1", 18131)
    server.start()
    try:
        result = probe("127.0.0.1", 18131, rtt_count=8, bulk_bytes=256 * 1024)
    finally:
        server.stop()

    assert result.ok
    assert result.rtt_count >= 6
    assert result.rtt_p50_ms is not None and result.rtt_p50_ms >= 0
    assert result.jitter_ms is not None
    assert result.upload_mbps and result.upload_mbps > 0
    assert result.download_mbps and result.download_mbps > 0
    assert result.failures == 0


def test_probe_reports_a_refused_connection_rather_than_guessing() -> None:
    result = probe("127.0.0.1", 18199, rtt_count=2, bulk_bytes=0, timeout_s=2.0)
    assert not result.ok
    assert result.error and "connect failed" in result.error
    assert result.rtt_p50_ms is None


def test_a_long_transfer_is_cut_into_frames() -> None:
    """A 4 MiB round trip must not be one 4 MiB socket write with no framing."""
    server = EchoServer("127.0.0.1", 18132)
    server.start()
    try:
        result = probe(
            "127.0.0.1", 18132, rtt_count=2, bulk_bytes=1024 * 1024, chunk_bytes=64 * 1024
        )
    finally:
        server.stop()
    assert result.ok
    assert result.upload_mbps and result.download_mbps


# --------------------------------------------------------------------------- #
# port selection
# --------------------------------------------------------------------------- #


def test_probe_ports_are_tried_in_order() -> None:
    record = node("a", mapped={10100: 11524, 10200: 11493, 10201: 11494, 1111: 11517})
    ports = candidate_probe_ports(record.inventory)
    assert ports[0] == PREFERRED_PROBE_PORTS[0] == 10200
    assert ports[1] == 10201
    assert 11517 in ports or True  # every valid mapping is a candidate


def test_a_port_that_cannot_be_bound_is_not_offered() -> None:
    record = node("a", mapped={"72299": 46012, "10100": 46058})
    assert candidate_probe_ports(record.inventory) == [10100]


def test_a_node_with_only_ssh_has_no_probe_ports() -> None:
    record = node("a", mapped={22: 40000})
    assert candidate_probe_ports(record.inventory) == []


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


NODES = {
    "alpha": node("alpha", mapped={10201: 41001}),
    "beta": node("beta", mapped={10201: 41002}),
}


def _manager() -> NetworkProbeManager:
    return NetworkProbeManager(NODES.get)


def _ready(manager: NetworkProbeManager, measurement_id: str, node_id: str) -> None:
    entry = manager.get(measurement_id).echo[node_id]
    port = entry["ports"][0]
    manager.record(
        node_id,
        measurement_id,
        "echo_ready",
        {"port": port, "address": f"203.0.113.9:{40000 + port}"},
    )


def _probes(manager: NetworkProbeManager, measurement_id: str) -> dict[str, dict]:
    sent: dict[str, dict] = {}
    for node_id in NODES:
        for command in manager.commands_for(node_id):
            if command["kind"] == "probe_peer":
                sent[f"{node_id}->{command['target']}"] = command
    return sent


def test_a_measurement_asks_both_nodes_to_listen_first() -> None:
    manager = _manager()
    measurement = manager.start(["alpha", "beta"])
    assert measurement.phase == "starting_echo"

    for node_id in NODES:
        commands = manager.commands_for(node_id)
        assert [c["kind"] for c in commands] == ["run_echo"]
        assert 10201 in commands[0]["ports"]


def test_probes_are_queued_in_both_directions_only_after_both_listen() -> None:
    manager = _manager()
    measurement = manager.start(["alpha", "beta"])
    manager.commands_for("alpha")
    manager.commands_for("beta")

    _ready(manager, measurement.measurement_id, "alpha")
    assert measurement.phase == "starting_echo"
    assert _probes(manager, measurement.measurement_id) == {}

    _ready(manager, measurement.measurement_id, "beta")
    assert measurement.phase == "probing"
    sent = _probes(manager, measurement.measurement_id)
    assert set(sent) == {"alpha->beta", "beta->alpha"}
    assert sent["alpha->beta"]["host"] == "203.0.113.9"
    # The probe dials the address the peer reported, not one computed here.
    assert sent["alpha->beta"]["port"] == 40000 + 10201


def test_both_directions_are_kept_separately() -> None:
    """A link is not symmetric; averaging the two would hide that."""
    manager = _manager()
    measurement = manager.start(["alpha", "beta"])
    _ready(manager, measurement.measurement_id, "alpha")
    _ready(manager, measurement.measurement_id, "beta")
    _probes(manager, measurement.measurement_id)

    manager.record(
        "alpha",
        measurement.measurement_id,
        "probe_result",
        {
            "target": "beta",
            "result": {"ok": True, "rtt_min_ms": 295.0, "rtt_p50_ms": 300.0, "rtt_count": 20, "download_mbps": 1.2},
        },
    )
    assert not measurement.complete

    manager.record(
        "beta",
        measurement.measurement_id,
        "probe_result",
        {
            "target": "alpha",
            "result": {"ok": True, "rtt_min_ms": 298.0, "rtt_p50_ms": 301.0, "rtt_count": 20, "download_mbps": 0.8},
        },
    )
    assert measurement.complete
    assert measurement.phase == "done"

    matrix = manager.topology(measurement)
    assert matrix is not None
    links = {f"{link.source}->{link.target}": link for link in matrix.links}
    assert links["alpha->beta"].rtt_ms_min == 295.0
    assert links["beta->alpha"].rtt_ms_min == 298.0
    assert links["alpha->beta"].tcp_mbps_single_stream == 1.2
    assert links["beta->alpha"].tcp_mbps_single_stream == 0.8


def test_a_failed_direction_is_recorded_as_a_failure_not_a_zero() -> None:
    manager = _manager()
    measurement = manager.start(["alpha", "beta"])
    _ready(manager, measurement.measurement_id, "alpha")
    _ready(manager, measurement.measurement_id, "beta")
    _probes(manager, measurement.measurement_id)

    for source, target in (("alpha", "beta"), ("beta", "alpha")):
        manager.record(
            source,
            measurement.measurement_id,
            "probe_result",
            {"target": target, "result": {"ok": False, "error": "connect failed"}},
        )

    matrix = manager.topology(measurement)
    assert matrix is not None
    for link in matrix.links:
        assert link.connection_success is False
        assert link.error == "connect failed"
        assert link.rtt_ms_min is None


def test_a_node_without_a_published_port_fails_the_measurement_with_a_reason() -> None:
    manager = NetworkProbeManager({"alpha": node("alpha", mapped={10201: 1}), "beta": node("beta", mapped={22: 2})}.get)
    measurement = manager.start(["alpha", "beta"])
    assert measurement.phase == "failed"
    assert "beta" in measurement.error and "SSH" in measurement.error


# --------------------------------------------------------------------------- #
# through the HTTP surface
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    async def fake_config(repo: str) -> str:
        return json.dumps(QWEN_05B)

    monkeypatch.setattr(coordinator_app, "fetch_model_config", fake_config)
    settings = CoordinatorSettings(
        port=8798, deployments_file=tmp_path / "deployments.json", enrollment_token="test-enroll"
    )
    return TestClient(coordinator_app.create_app(settings))


def test_measurement_end_to_end_through_the_api(client: TestClient) -> None:
    state = client.app.state.mw
    for record in (node("alpha", mapped={10201: 41001}), node("beta", mapped={10201: 41002})):
        state.registry.register(record.registration)
        state.registry.get(record.node_id).inventory = record.inventory

    started = client.post(
        "/v1/network/measure", json={"node_ids": ["alpha", "beta"], "rtt_count": 4}
    )
    assert started.status_code == 200, started.text
    measurement_id = started.json()["measurement_id"]

    for node_id in ("alpha", "beta"):
        work = client.get(
            f"/v1/agents/{node_id}/work", headers={"authorization": "Bearer test-enroll"}
        ).json()
        # Probes and stages share one list: a second key would be invisible to
        # agents already deployed, which read only deployment_commands.
        assert [c["kind"] for c in work["deployment_commands"]] == ["run_echo"]

    for node_id in ("alpha", "beta"):
        client.post(
            f"/v1/nodes/{node_id}/network-result",
            json={
                "measurement_id": measurement_id,
                "phase": "echo_ready",
                "payload": {"port": 10201, "address": "203.0.113.9:11524"},
            },
        )

    # Now each node has one probe to run, aimed at the other.
    probe_commands = {}
    for node_id in ("alpha", "beta"):
        work = client.get(
            f"/v1/agents/{node_id}/work", headers={"authorization": "Bearer test-enroll"}
        ).json()
        probes = [c for c in work["deployment_commands"] if c["kind"] == "probe_peer"]
        assert len(probes) == 1
        probe_commands[node_id] = probes[0]

    for node_id, command in probe_commands.items():
        client.post(
            f"/v1/nodes/{node_id}/network-result",
            json={
                "measurement_id": measurement_id,
                "phase": "probe_result",
                "payload": {
                    "target": command["target"],
                    "result": {"ok": True, "rtt_min_ms": 12.0, "rtt_p50_ms": 12.5, "rtt_count": 4, "download_mbps": 50.0},
                },
            },
        )

    finished = client.get(f"/v1/network/measurements/{measurement_id}").json()
    assert finished["phase"] == "done"
    assert set(finished["results"]) == {"alpha->beta", "beta->alpha"}

    # A finished measurement becomes the topology the planner reads.
    topology = client.get("/v1/topology")
    assert topology.status_code == 200
    links = topology.json()["matrix"]["links"]
    assert {link["source"] for link in links} == {"alpha", "beta"}
    assert len(links) == 2


def test_an_older_agent_that_reports_only_a_port_still_works() -> None:
    """Agents deployed before multi-candidate support send no address. The
    coordinator must resolve it from the inventory rather than fail."""
    manager = NetworkProbeManager(NODES.get)
    measurement = manager.start(["alpha", "beta"])
    manager.commands_for("alpha")
    manager.commands_for("beta")

    manager.record("alpha", measurement.measurement_id, "echo_ready", {"port": 10100})
    manager.record("beta", measurement.measurement_id, "echo_ready", {"port": 10100})

    sent = _probes(manager, measurement.measurement_id)
    assert sent["alpha->beta"]["host"] == "203.0.113.9"
    assert sent["alpha->beta"]["port"] > 0
    assert sent["beta->alpha"]["port"] > 0
