"""Agent runtime: the process that runs on every GPU host.

Responsibilities:
  * discover GPUs and report real telemetry,
  * register with the coordinator and send heartbeats,
  * accept workload assignments and report execution metrics.

It does not load models itself. Loading is the backend's job (see
``meshweave.runtime.backends``), which keeps the agent small and the boundary
explicit: the agent knows about *stages*, not about *weights*.

Run with:
    meshweave agent --node-id node-a --coordinator https://10.0.0.1:8700
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import Any

from meshweave import __version__
from meshweave.agent import telemetry
from meshweave.agent.api import AgentState, create_agent_app
from meshweave.config import AgentSettings
from meshweave.core import GpuInventory, Heartbeat, NodeRegistration
from meshweave.errors import MeshweaveError

logger = logging.getLogger("meshweave.agent")


class AgentRuntime:
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        if not settings.node_id:
            raise MeshweaveError(
                "agent node_id is required. Pass --node-id or set MESHWEAVE_AGENT_NODE_ID."
            )
        self.node_id: str = settings.node_id
        self.state = AgentState(
            node_id=self.node_id,
            coordinator_url=settings.coordinator_url,
            simulation=settings.simulation,
        )
        self._inventory: GpuInventory | None = None
        self._stop = asyncio.Event()

    # ------------------------------------------------------------- startup

    def collect_inventory(self) -> GpuInventory:
        """Real NVML inventory. Raises on a host with no NVIDIA driver."""
        if self.settings.simulation:
            from meshweave.agent.synthetic import synthetic_inventory

            inv = synthetic_inventory(agent_version=__version__)
            self.state.set_inventory(inv)
            return inv

        inv = telemetry.collect_inventory(__version__)
        self.state.set_inventory(inv)
        return inv

    async def register(self) -> dict[str, Any]:
        inv = self.collect_inventory()
        registration = NodeRegistration(
            node_id=self.node_id,
            hostname=inv.network.hostname or self.node_id,
            agent_version=__version__,
            provider="manual",
            region=inv.network.geolocation,
        )
        import httpx

        async with httpx.AsyncClient(timeout=30.0, verify=self._verify()) as client:
            resp = await client.post(
                f"{self.settings.coordinator_url}/v1/nodes/register",
                json=registration.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return dict(resp.json())

    def _verify(self) -> str | bool:
        if self.settings.tls_ca:
            return str(self.settings.tls_ca)
        return True

    # ------------------------------------------------------------ main loop

    async def heartbeat_loop(self, interval_s: float) -> None:
        import httpx

        while not self._stop.is_set():
            if self._inventory is not None:
                beat = Heartbeat(
                    node_id=self.node_id, inventory=self._inventory
                )
                try:
                    async with httpx.AsyncClient(
                        timeout=15.0, verify=self._verify()
                    ) as client:
                        resp = await client.post(
                            f"{self.settings.coordinator_url}"
                            f"/v1/nodes/{self.node_id}/heartbeat",
                            json=beat.model_dump(mode="json"),
                        )
                        resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("heartbeat failed: %s", exc)
            await _sleep_or_stop(self._stop, interval_s)

    async def run(self) -> None:
        """Register, then serve the control API until stopped."""
        await self.register()
        logger.info("registered as %s", self.settings.node_id)

        app = create_agent_app(self.state)
        config = _uvicorn_config(app, self.settings)
        server = _uvicorn_server(config)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop.set)

        heartbeat_task = asyncio.create_task(
            self.heartbeat_loop(self.settings.heartbeat_interval_s)
        )
        try:
            await server.serve()
        finally:
            self._stop.set()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task


def _uvicorn_config(app: Any, settings: AgentSettings) -> Any:
    import uvicorn

    kwargs: dict[str, Any] = {
        "host": "0.0.0.0",  # noqa: S104 - agents listen on their mapped port
        "port": 8443,
        "log_level": "info",
    }
    if settings.tls_cert and settings.tls_key:
        kwargs["ssl_certfile"] = str(settings.tls_cert)
        kwargs["ssl_keyfile"] = str(settings.tls_key)
    return uvicorn.Config(app, **kwargs)


def _uvicorn_server(config: Any) -> Any:
    import uvicorn

    return uvicorn.Server(config)


async def _sleep_or_stop(event: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(event.wait(), timeout=seconds)
