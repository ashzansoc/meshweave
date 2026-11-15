"""Coordinator: the control plane.

Responsibilities per the MESHWEAVE architecture:
  node registration and authentication, resource discovery, GPU inventory,
  cluster formation, network topology, workload management, scheduling,
  execution coordination, failure detection, experiment management.

It is a modular monolith on purpose. Premature microservices would multiply the
failure modes before the underlying execution is proven.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status

from meshweave import __version__
from meshweave.communication.auth import (
    WorkloadToken,
    issue_workload_token,
    load_or_create_secret,
)
from meshweave.config import CoordinatorSettings
from meshweave.core import (
    ExecutionRecord,
    Heartbeat,
    NodeRegistration,
    PlanningRequest,
    TopologyMatrix,
    WorkloadAssignment,
    utcnow,
)
from meshweave.errors import (
    AuthError,
    PlanningError,
)
from meshweave.planner.intelligence import ExecutionPlanner, PlanningContext
from meshweave.registry.store import ResourceRegistry
from meshweave.topology.graph import TopologyGraph, build_graph

logger = logging.getLogger("meshweave.coordinator")

SECRET_PATH = "./reports/.coordinator_secret"


class WorkloadManager:
    """Tracks running workloads and the stages they occupy."""

    def __init__(self) -> None:
        self._workloads: dict[str, ExecutionRecord] = {}
        self._assignments: dict[str, list[WorkloadAssignment]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}

    def create(self, record: ExecutionRecord) -> None:
        self._workloads[record.workload_id] = record
        self._events[record.workload_id] = []

    def get(self, workload_id: str) -> ExecutionRecord | None:
        return self._workloads.get(workload_id)

    def all(self) -> list[ExecutionRecord]:
        return list(self._workloads.values())

    def active(self) -> list[ExecutionRecord]:
        return [
            w
            for w in self._workloads.values()
            if w.finished_at is None and w.success
        ]

    def record_assignment(self, assignment: WorkloadAssignment) -> None:
        self._assignments.setdefault(assignment.workload_id, []).append(assignment)

    def assignments_for(self, workload_id: str) -> list[WorkloadAssignment]:
        return self._assignments.get(workload_id, [])

    def finish(self, workload_id: str, record: ExecutionRecord) -> None:
        self._workloads[workload_id] = record

    def event(self, workload_id: str, kind: str, detail: dict[str, Any] | None = None) -> None:
        self._events.setdefault(workload_id, []).append(
            {"at": utcnow().isoformat(), "kind": kind, "detail": detail or {}}
        )

    def events(self, workload_id: str) -> list[dict[str, Any]]:
        return self._events.get(workload_id, [])


class CoordinatorState:
    def __init__(self, settings: CoordinatorSettings) -> None:
        self.settings = settings
        self.registry = ResourceRegistry(settings)
        self.workloads = WorkloadManager()
        self.planner = ExecutionPlanner()
        self.topology: TopologyMatrix | None = None
        self.graph: TopologyGraph | None = None
        self.secret = load_or_create_secret(SECRET_PATH)
        self._agent_clients: dict[str, Any] = {}
        self._reaper: asyncio.Task[None] | None = None

    def set_topology(self, matrix: TopologyMatrix) -> None:
        self.topology = matrix
        self.graph = build_graph(matrix)

    def node_endpoint(self, node_id: str) -> str | None:
        record = self.registry.get(node_id)
        if record is None or record.inventory is None:
            return None
        net = record.inventory.network
        if net.external_ip and net.mapped_ports:
            internal = min(net.mapped_ports)
            return f"https://{net.external_ip}:{net.mapped_ports[internal]}"
        if net.private_ip:
            return f"https://{net.private_ip}:8443"
        return None


def create_app(settings: CoordinatorSettings | None = None) -> FastAPI:
    settings = settings or CoordinatorSettings()
    state = CoordinatorState(settings)

    app = FastAPI(
        title="MESHWEAVE Coordinator",
        version=__version__,
        description=(
            "Control plane for MESHWEAVE. Schedules model execution across "
            "independently hosted GPUs and records what actually happened."
        ),
    )
    app.state.mw = state

    _register_routes(app, state)
    return app


# --------------------------------------------------------------------------- #
# auth dependencies
# --------------------------------------------------------------------------- #


def _bearer(header: str | None) -> str | None:
    if not header or not header.lower().startswith("bearer "):
        return None
    return header.split(" ", 1)[1].strip()


def require_admin(state: CoordinatorState, authorization: str | None) -> None:
    """Admin operations need the coordinator secret, never a node credential."""
    token = _bearer(authorization)
    if not token or not secrets.compare_digest(token, state.secret.hex()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token"
        )


def require_node_token(state: CoordinatorState, token_str: str, node_id: str) -> WorkloadToken:
    try:
        token = WorkloadToken.decode(token_str, state.secret)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    if token.node_id != node_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="token node mismatch"
        )
    return token


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


def _register_routes(app: FastAPI, state: CoordinatorState) -> None:
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "nodes": len(state.registry.all()),
            "healthy_nodes": len(state.registry.healthy()),
            "topology_measured": state.topology is not None,
        }

    # -- registration / telemetry ---------------------------------------

    @app.post("/v1/nodes/register")
    async def register(registration: NodeRegistration) -> dict[str, Any]:
        record = state.registry.register(registration)
        state.workloads.event(
            f"node:{registration.node_id}", "registered", {"node_id": registration.node_id}
        )
        # Hand the agent a node-scoped credential it uses for heartbeats.
        return {
            "node_id": record.node_id,
            "accepted": True,
            "heartbeat_interval_s": state.settings.heartbeat_timeout_s / 3.0,
        }

    @app.post("/v1/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, beat: Heartbeat) -> dict[str, Any]:
        if beat.node_id != node_id:
            raise HTTPException(status_code=400, detail="node_id mismatch")
        try:
            record = state.registry.record_heartbeat(beat)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unregistered node") from exc
        state.registry.recompute_reliability(node_id)
        return {
            "ok": True,
            "accepted_at": utcnow().isoformat(),
            "reliability": record.reliability_score,
            "status": record.status.value,
        }

    @app.get("/v1/nodes")
    async def list_nodes() -> dict[str, Any]:
        return {
            "nodes": [n.model_dump(mode="json") for n in state.registry.all()],
            "summary": {
                "total": len(state.registry.all()),
                "healthy": len(state.registry.healthy()),
                "total_gpus": state.registry.total_gpu_count(),
                "total_vram_gib": round(state.registry.total_vram_bytes() / 1024**3, 2),
                "free_vram_gib": round(state.registry.total_free_vram_bytes() / 1024**3, 2),
                "hourly_cost_usd": state.registry.hourly_cost_usd(),
            },
        }

    @app.post("/v1/nodes/{node_id}/cost")
    async def set_cost(node_id: str, body: dict[str, Any]) -> dict[str, Any]:
        usd = body.get("hourly_cost_usd")
        if usd is None:
            raise HTTPException(status_code=400, detail="hourly_cost_usd required")
        state.registry.set_hourly_cost(node_id, float(usd))
        return {"node_id": node_id, "hourly_cost_usd": float(usd)}

    # -- topology --------------------------------------------------------

    @app.post("/v1/topology")
    async def set_topology(matrix: TopologyMatrix) -> dict[str, Any]:
        state.set_topology(matrix)
        return {
            "nodes": len(matrix.nodes),
            "directed_links": len(matrix.links),
            "pairs": state.topology.pair_count() // 2 if state.topology else 0,
            "environments": (
                {k: v.value for k, v in state.graph.environment_by_node.items()}
                if state.graph
                else {}
            ),
        }

    @app.get("/v1/topology")
    async def get_topology() -> dict[str, Any]:
        if state.topology is None:
            raise HTTPException(status_code=404, detail="no topology measured yet")
        return {
            "matrix": state.topology.model_dump(mode="json"),
            "environments": (
                {k: v.value for k, v in state.graph.environment_by_node.items()}
                if state.graph
                else {}
            ),
        }

    # -- planning --------------------------------------------------------

    @app.post("/v1/plan")
    async def plan(request: PlanningRequest) -> dict[str, Any]:
        nodes = state.registry.available()
        if not nodes:
            raise HTTPException(
                status_code=503,
                detail="no healthy nodes with telemetry; run the network benchmark first",
            )
        ctx = PlanningContext(
            nodes=nodes,
            graph=state.graph,
            per_hop_rtt_ms=_measured_per_hop_rtt(state),
            tokens_per_gpu_sec=20.0,
        )
        try:
            result = state.planner.plan(request, ctx)
        except PlanningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.post("/v1/plan/compare")
    async def plan_compare(request: PlanningRequest) -> dict[str, Any]:
        nodes = state.registry.available()
        if not nodes:
            raise HTTPException(status_code=503, detail="no healthy nodes")
        ctx = PlanningContext(
            nodes=nodes,
            graph=state.graph,
            per_hop_rtt_ms=_measured_per_hop_rtt(state),
            tokens_per_gpu_sec=20.0,
        )
        return state.planner.compare(request, ctx)

    # -- workloads -------------------------------------------------------

    @app.post("/v1/workloads")
    async def submit(request: PlanningRequest) -> dict[str, Any]:
        nodes = state.registry.available()
        if not nodes:
            raise HTTPException(status_code=503, detail="no healthy nodes")
        ctx = PlanningContext(
            nodes=nodes,
            graph=state.graph,
            per_hop_rtt_ms=_measured_per_hop_rtt(state),
            tokens_per_gpu_sec=20.0,
        )
        try:
            result = state.planner.plan(request, ctx)
        except PlanningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        workload_id = f"wl-{secrets.token_hex(8)}"
        record = ExecutionRecord(
            workload_id=workload_id,
            model=request.model,
            strategy=result.strategy,
            nodes=list(
                result.replica_nodes
                or list(result.stage_assignment.values())
                or []
            ),
            environment=result.environment,
        )
        state.workloads.create(record)
        state.workloads.event(workload_id, "planned", result.model_dump(mode="json"))
        return {"workload_id": workload_id, "plan": result.model_dump(mode="json")}

    @app.get("/v1/workloads")
    async def list_workloads() -> dict[str, Any]:
        return {
            "workloads": [w.model_dump(mode="json") for w in state.workloads.all()],
            "active": len(state.workloads.active()),
        }

    @app.get("/v1/workloads/{workload_id}")
    async def get_workload(workload_id: str) -> dict[str, Any]:
        record = state.workloads.get(workload_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown workload")
        return {
            "workload": record.model_dump(mode="json"),
            "events": state.workloads.events(workload_id),
            "assignments": [
                a.model_dump(mode="json") for a in state.workloads.assignments_for(workload_id)
            ],
        }

    @app.post("/v1/workloads/{workload_id}/assign")
    async def assign(workload_id: str, assignment: WorkloadAssignment) -> dict[str, Any]:
        state.workloads.record_assignment(assignment)
        state.workloads.event(workload_id, "assigned", assignment.model_dump(mode="json"))
        record = state.workloads.get(workload_id)
        token: WorkloadToken | None = None
        if record is not None:
            token = issue_workload_token(workload_id, assignment.node_id, state.secret)
        return {"ok": True, "token": token.encode() if token else None}

    @app.post("/v1/workloads/{workload_id}/complete")
    async def complete(workload_id: str, record: ExecutionRecord) -> dict[str, Any]:
        state.workloads.finish(workload_id, record)
        state.workloads.event(
            workload_id,
            "completed" if record.success else "failed",
            {"output_tokens": record.output_tokens, "error": record.error},
        )
        return {"ok": True}

    # -- metrics / evidence ---------------------------------------------

    @app.get("/v1/metrics")
    async def metrics() -> dict[str, Any]:
        records = state.workloads.all()
        successful = [r for r in records if r.success and r.output_tokens > 0]
        return {
            "workloads_total": len(records),
            "workloads_successful": len(successful),
            "workloads_failed": len(records) - len(successful),
            "tokens_generated": sum(r.output_tokens for r in successful),
            "mean_tps": _mean([r.tokens_per_second for r in successful if r.tokens_per_second]),
            "mean_ttft_ms": _mean([r.ttft_ms for r in successful if r.ttft_ms]),
            "mean_cost_per_m_output_tokens_usd": _mean(
                [r.cost_per_m_output_tokens_usd for r in successful if r.cost_per_m_output_tokens_usd]
            ),
            "cluster_hourly_cost_usd": state.registry.hourly_cost_usd(),
            "total_vram_gib": round(state.registry.total_vram_bytes() / 1024**3, 2),
            "free_vram_gib": round(state.registry.total_free_vram_bytes() / 1024**3, 2),
            "measured_topology": state.topology is not None,
        }

    @app.post("/v1/admin/reap")
    async def reap(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(state, authorization)
        changed = state.registry.reap()
        return {"marked_unreachable": changed}

    return


def _measured_per_hop_rtt(state: CoordinatorState) -> float:
    """Mean measured RTT across the topology, or a documented default.

    The planner prefers a measured number. When no measurement exists the
    default is used and the plan is annotated as an estimate, never presented
    as a measurement.
    """
    if state.topology is None:
        return 50.0
    values = [
        link.rtt_ms_p50
        for link in state.topology.links
        if link.rtt_ms_p50 is not None and link.usable
    ]
    if not values:
        return 50.0
    return sum(values) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
