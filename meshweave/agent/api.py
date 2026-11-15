"""Agent control API.

The closed set of things the coordinator may ask an agent to do:

  ping       -> liveness + current inventory
  assign     -> load a model stage and prepare to serve activations
  activation -> receive an activation, run this stage's layers, return the result
  sample     -> sample the next token from the final stage's logits
  memory     -> current GPU memory footprint of the workload
  cancel     -> drop a workload and free its memory
  drain      -> stop accepting new work

There is deliberately no shell, no file upload, no arbitrary code execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response

from meshweave import __version__
from meshweave.core import (
    GpuInventory,
    Heartbeat,
    MemorySample,
    WorkloadAssignment,
    WorkloadStatus,
    utcnow,
)

logger = logging.getLogger("meshweave.agent")


@dataclass
class LoadedStage:
    """A model stage resident on this node."""

    workload_id: str
    node_id: str
    model: str
    stage_index: int
    layer_start: int
    layer_end: int
    precision: str
    vram_bytes_used: int = 0
    loaded_at: float = field(default_factory=time.time)
    # Populated by the runtime backend when a real model is loaded.
    backend: Any = None
    status: WorkloadStatus = WorkloadStatus.LOADING

    def memory_sample(self, gpu_index: int, gpu_uuid: str, total: int, used: int) -> MemorySample:
        return MemorySample(
            node_id=self.node_id,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            vram_total_bytes=total,
            vram_used_bytes=used,
            vram_free_bytes=max(0, total - used),
            process_rss_bytes=self.vram_bytes_used or None,
        )


class AgentState:
    def __init__(self, node_id: str, coordinator_url: str, simulation: bool = False) -> None:
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.simulation = simulation
        self.started_at = time.time()
        self.stages: dict[str, LoadedStage] = {}
        self.draining = False
        self.inventory: GpuInventory | None = None
        self._lock = asyncio.Lock()

    def set_inventory(self, inventory: GpuInventory) -> None:
        self.inventory = inventory


def _require_token(provided: str | None, expected: str) -> None:
    import secrets as _s

    if not provided or not _s.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid workload token")


def create_agent_app(
    state: AgentState,
    token_for: Any = None,
) -> FastAPI:
    """Build the agent's FastAPI application.

    ``token_for(workload_id, node_id)`` must return the expected token string
    for that workload, or None when the workload is unknown to this node.
    """
    app = FastAPI(title="MESHWEAVE Agent", version=__version__)
    app.state.agent = state

    @app.get("/agent/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "draining" if state.draining else "ok",
            "node_id": state.node_id,
            "version": __version__,
            "uptime_s": round(time.time() - state.started_at, 1),
            "simulation": state.simulation,
            "stages": len(state.stages),
        }

    @app.post("/agent/ping")
    async def ping() -> dict[str, Any]:
        inv = state.inventory
        return {
            "node_id": state.node_id,
            "at": utcnow().isoformat(),
            "has_inventory": inv is not None,
            "gpus": len(inv.gpus) if inv else 0,
            "vram_free_gib": round(sum(g.vram_free_gib for g in inv.gpus), 2) if inv else None,
            "draining": state.draining,
        }

    @app.get("/agent/inventory")
    async def inventory() -> dict[str, Any]:
        if state.inventory is None:
            raise HTTPException(status_code=503, detail="no inventory collected yet")
        return state.inventory.model_dump(mode="json")

    @app.post("/agent/heartbeat")
    async def heartbeat() -> dict[str, Any]:
        """Build a heartbeat payload for the coordinator poll loop."""
        if state.inventory is None:
            raise HTTPException(status_code=503, detail="no inventory")
        return Heartbeat(
            node_id=state.node_id,
            inventory=state.inventory,
            active_workloads=list(state.stages),
        ).model_dump(mode="json")

    @app.post("/agent/assign")
    async def assign(assignment: WorkloadAssignment, request: Request) -> dict[str, Any]:
        if state.draining:
            raise HTTPException(status_code=503, detail="agent is draining")
        if assignment.node_id != state.node_id:
            raise HTTPException(status_code=403, detail="assignment is for another node")

        expected = token_for(assignment.workload_id, state.node_id) if token_for else None
        if expected is not None:
            _require_token(request.headers.get("authorization", "").removeprefix("Bearer "), expected)

        async with state._lock:
            if assignment.workload_id in state.stages:
                raise HTTPException(
                    status_code=409, detail=f"workload {assignment.workload_id} already loaded"
                )
            stage = LoadedStage(
                workload_id=assignment.workload_id,
                node_id=state.node_id,
                model=assignment.model,
                stage_index=assignment.stage_index or 0,
                layer_start=assignment.layer_start or 0,
                layer_end=assignment.layer_end or 0,
                precision=assignment.quantization,
                status=WorkloadStatus.LOADING,
            )
            state.stages[assignment.workload_id] = stage

        return {
            "ok": True,
            "workload_id": assignment.workload_id,
            "stage_index": stage.stage_index,
            "layers": [stage.layer_start, stage.layer_end],
            "note": (
                "stage registered; a real backend attaches weights in"
                " AgentRuntime.load_stage()"
            ),
        }

    @app.post("/agent/activation")
    async def activation(request: Request) -> Response:
        """Receive an activation, run this stage's layers, return the result.

        When no real backend is attached this echoes the payload back with the
        stage's shape, which lets the transport be exercised end to end.
        """
        from meshweave.runtime.transport import decode_activation, encode_activation

        frame = await request.body()
        try:
            act = decode_activation(frame)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad activation: {exc}") from exc

        stage = state.stages.get(request.headers.get("x-meshweave-workload", ""))
        if stage is None:
            raise HTTPException(status_code=401, detail="missing workload token")

        backend = stage.backend
        if backend is None:
            # No weights loaded: echo. This exercises transport, not execution.
            return Response(
                content=encode_activation(act), media_type="application/octet-stream"
            )

        try:
            out = backend.forward(act)
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise HTTPException(status_code=500, detail=f"stage forward failed: {exc}") from exc
        return Response(content=encode_activation(out), media_type="application/octet-stream")

    @app.post("/agent/sample")
    async def sample(body: dict[str, Any]) -> dict[str, Any]:
        workload_id = str(body.get("workload_id", ""))
        stage = state.stages.get(workload_id)
        if stage is None:
            raise HTTPException(status_code=404, detail="unknown workload")
        backend = stage.backend
        if backend is None:
            raise HTTPException(
                status_code=501,
                detail="no model backend loaded; sampling requires real weights",
            )
        try:
            token, finished = backend.sample(
                generated=list(body.get("generated", [])), max_new_tokens=1
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"sampling failed: {exc}") from exc
        return {"token_id": token, "finished": finished}

    @app.post("/agent/memory")
    async def memory() -> dict[str, Any]:
        inv = state.inventory
        samples = []
        if inv:
            for gpu in inv.gpus:
                for stage in state.stages.values():
                    if stage.node_id != state.node_id:
                        continue
                    samples.append(
                        stage.memory_sample(
                            gpu.index, gpu.uuid, gpu.vram_total_bytes, gpu.vram_used_bytes
                        ).model_dump(mode="json")
                    )
        return {"node_id": state.node_id, "samples": samples}

    @app.post("/agent/cancel")
    async def cancel(body: dict[str, Any]) -> dict[str, Any]:
        workload_id = str(body.get("workload_id", ""))
        async with state._lock:
            removed = state.stages.pop(workload_id, None)
        if removed is None:
            return {"ok": True, "cancelled": False, "reason": "not loaded here"}
        return {"ok": True, "cancelled": True, "workload_id": workload_id}

    @app.post("/agent/drain")
    async def drain() -> dict[str, Any]:
        state.draining = True
        return {"ok": True, "draining": True}

    return app
