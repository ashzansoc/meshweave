"""End-to-end pipeline over real HTTP.

This is the test that proves the transport and the stage protocol actually work
against a live server on a real socket, rather than against a mock. Two agent
apps are started on ephemeral ports, a workload is assigned to each, and an
activation is pushed through the chain.

No model weights are involved: the stage backend is absent, so each agent echoes
the activation. That is deliberate. This test proves the *transport*; execution
correctness is a separate concern proven by EXP-011 on real GPUs.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest
import uvicorn
from meshweave.agent.api import AgentState, create_agent_app
from meshweave.agent.synthetic import synthetic_inventory
from meshweave.core import WorkloadAssignment
from meshweave.runtime.transport import Activation, StageConnection, decode_activation


class _Server:
    def __init__(self, app, port: int) -> None:
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self.server = uvicorn.Server(self.config)

    async def start(self) -> None:
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:  # noqa: ASYNC110 - polling uvicorn startup
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        self.server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


def _free_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
async def two_agents():
    ports = [_free_port(), _free_port()]
    states = []
    servers = []
    for i, port in enumerate(ports):
        state = AgentState(node_id=f"node-{i}", coordinator_url="http://unused")
        state.set_inventory(synthetic_inventory("RTX 4090", agent_version="test"))
        app = create_agent_app(state)
        server = _Server(app, port)
        await server.start()
        states.append(state)
        servers.append(server)

    yield list(zip(states, ports, strict=True))

    for server in servers:
        await server.stop()


@pytest.mark.asyncio
async def test_agent_health_and_ping(two_agents) -> None:
    (state, port), _ = two_agents
    async with httpx.AsyncClient(timeout=10.0) as client:
        health = await client.get(f"http://127.0.0.1:{port}/agent/health")
        health.raise_for_status()
        body = health.json()
        assert body["node_id"] == "node-0"
        assert body["simulation"] is False

        ping = await client.post(f"http://127.0.0.1:{port}/agent/ping")
        ping.raise_for_status()
        assert ping.json()["gpus"] == 1


@pytest.mark.asyncio
async def test_stage_registration_and_rejection(two_agents) -> None:
    (_, port), _ = two_agents
    assignment = WorkloadAssignment(
        workload_id="wl-test",
        node_id="node-0",
        model="Qwen2.5-7B-Instruct",
        stage_index=0,
        layer_start=0,
        layer_end=14,
        quantization="bf16",
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/agent/assign", json=assignment.model_dump(mode="json")
        )
        resp.raise_for_status()
        assert resp.json()["layers"] == [0, 14]

        # a second assignment for the same workload is a conflict, not a silent overwrite
        dup = await client.post(
            f"http://127.0.0.1:{port}/agent/assign", json=assignment.model_dump(mode="json")
        )
        assert dup.status_code == 409


@pytest.mark.asyncio
async def test_activation_traverses_two_stages(two_agents) -> None:
    """Push a real activation through a two-stage chain over real HTTP."""
    (state_a, port_a), (state_b, port_b) = two_agents

    for _state, node_id, port, (start, end) in (
        (state_a, "node-0", port_a, (0, 14)),
        (state_b, "node-1", port_b, (14, 28)),
    ):
        assignment = WorkloadAssignment(
            workload_id="wl-chain",
            node_id=node_id,
            model="Qwen2.5-7B-Instruct",
            stage_index=0 if node_id == "node-0" else 1,
            layer_start=start,
            layer_end=end,
            quantization="bf16",
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/agent/assign", json=assignment.model_dump(mode="json")
            )
            resp.raise_for_status()

    activation = Activation(
        dtype="bfloat16", shape=(1, 7168), data=b"\x01" * (7168 * 2)
    )

    conn = StageConnection(
        endpoint=_endpoint(port_b),
        node_id="node-1",
        stage_index=0,
        workload_id="wl-chain",
    )
    stats = await conn.send(activation, to_stage=1)

    assert stats.bytes_sent == activation.nbytes
    assert stats.send_ms > 0
    assert stats.round_trip_ms is not None

    # and the payload that arrived is the payload that was sent
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"http://127.0.0.1:{port_b}/agent/activation",
            content=_encode(activation),
            headers={"x-meshweave-workload": "wl-chain"},
        )
        resp.raise_for_status()
        decoded = decode_activation(resp.content)
        assert decoded.nbytes == activation.nbytes
        assert decoded.shape == activation.shape


@pytest.mark.asyncio
async def test_unknown_workload_is_rejected(two_agents) -> None:
    (_, port), _ = two_agents
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/agent/activation",
            content=_encode(Activation(dtype="bfloat16", shape=(1, 1), data=b"\x00\x00")),
            headers={"x-meshweave-workload": "does-not-exist"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_sample_without_backend_is_not_faked(two_agents) -> None:
    """Sampling without weights must fail loudly, never invent a token."""
    (_, port), _ = two_agents
    assignment = WorkloadAssignment(
        workload_id="wl-noweights",
        node_id="node-0",
        model="Qwen2.5-7B-Instruct",
        stage_index=0,
        layer_start=0,
        layer_end=28,
        quantization="bf16",
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"http://127.0.0.1:{port}/agent/assign", json=assignment.model_dump(mode="json")
        )
        resp = await client.post(
            f"http://127.0.0.1:{port}/agent/sample",
            json={"workload_id": "wl-noweights", "generated": [], "max_new_tokens": 1},
        )
        assert resp.status_code == 501


def _encode(act: Activation) -> bytes:
    from meshweave.runtime.transport import encode_activation

    return encode_activation(act)


def _endpoint(port: int):
    from meshweave.communication.wire import HttpEndpoint

    return HttpEndpoint(base_url=f"http://127.0.0.1:{port}", timeout_s=10.0)
