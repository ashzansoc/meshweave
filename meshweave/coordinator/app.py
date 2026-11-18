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
import contextlib
import logging
import secrets
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from meshweave import __version__
from meshweave.communication.auth import (
    WorkloadToken,
    issue_node_token,
    issue_workload_token,
    load_or_create_secret,
    verify_node_token,
)
from meshweave.config import CoordinatorSettings
from meshweave.coordinator.clusters import ClusterStore, cluster_status, validate_members
from meshweave.coordinator.network import NetworkProbeManager
from meshweave.coordinator.deployments import (
    GIB,
    Deployment,
    DeploymentRejected,
    ModelConfigUnavailable,
    fetch_model_config,
    DeploymentStore,
    effective_dtype,
    load_remote_config,
    new_deployment,
    normalise_repo,
    plan_deployment,
    plan_replicas,
    choose_replica,
    stage_key_for,
    best_single_node,
)
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SECRET_PATH = "./reports/.coordinator_secret"

# A stage that cannot start after this many attempts fails the deployment. The
# bound exists so a genuinely broken stage does not retry forever.
MAX_STAGE_ATTEMPTS = 3



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



class ClusterRequest(BaseModel):
    name: str
    node_ids: list[str] = Field(default_factory=list)


class NetworkMeasureRequest(BaseModel):
    node_ids: list[str] = Field(default_factory=list)
    rtt_count: int = 20
    bulk_bytes: int = 4 * 1024 * 1024


class NetworkProbeReport(BaseModel):
    measurement_id: str
    phase: str
    payload: dict[str, Any] = Field(default_factory=dict)



# --------------------------------------------------------------------------- #
# deployment request bodies
#
# Declared at module scope on purpose. Inside create_app() the annotations are
# strings (from __future__ import annotations) that FastAPI cannot resolve
# against module globals, so it silently treats the body as a query parameter
# and every request fails with a 422 about a missing "request" field.
# --------------------------------------------------------------------------- #


class DeploymentRequest(BaseModel):
    model: str
    node_ids: list[str] = Field(default_factory=list)
    cluster_id: str | None = None
    max_seq: int = 2048
    # "AUTO" lets the coordinator choose, "PIPELINE" splits one model across
    # nodes, "REPLICA" runs a full copy on each.
    mode: str = "AUTO"
    # Copies to place on each node. One is the default; more exploit idle GPU
    # memory when decode is CPU-bound.
    replicas_per_node: int = 1


class DeploymentStatusReport(BaseModel):
    phase: str
    detail: str | None = None
    pid: int | None = None
    # Which replica on this node. Defaults to replica 0 so a node running one
    # copy keeps working with an older agent.
    stage_key: str | None = None


class DeploymentResult(BaseModel):
    prompt: str = ""
    text: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    ms_per_token: float | None = None
    tokens_per_second: float | None = None
    total_s: float | None = None
    error: str | None = None
    node_id: str | None = None
    prompt_id: str | None = None

    # Evidence, not decoration. Without these fields the runner computes them and
    # the model silently drops them, so a distributed request reports nothing
    # about the activations it moved or the GPU that was busy.
    stage_roundtrip_ms: float | None = None
    activation_bytes_per_token: float | None = None
    activation_bytes_sent: int | None = None
    world: int | None = None
    rank: int | None = None
    gpu_during_generation: dict[str, Any] | None = None
    # Prefill and decode are separate workloads; per-step timings let a caller
    # compute inter-token latency percentiles rather than only a mean.
    prefill_ms: float | None = None
    step_ms: list[float] | None = None


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0


class CoordinatorState:
    def __init__(self, settings: CoordinatorSettings) -> None:
        self.settings = settings
        self.registry = ResourceRegistry(settings)
        self.workloads = WorkloadManager()
        self.planner = ExecutionPlanner()
        self.topology: TopologyMatrix | None = None
        self.graph: TopologyGraph | None = None
        self.secret = load_or_create_secret(SECRET_PATH)
        self.deployments = DeploymentStore(Path(settings.deployments_file))
        self.network = NetworkProbeManager(self.registry.get)
        self.clusters = ClusterStore(Path(settings.clusters_file))
        # Per-node telemetry history for the charts. Bounded and session-only:
        # it is what this coordinator observed while it was running, not a time
        # series database, and the API says so.
        self.series: dict[str, deque[dict[str, Any]]] = {}
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


def _availability(record: Any, deployments: Any) -> str:
    """The brief's node vocabulary, derived rather than stored.

    Internal status answers "is it alive", which is not the same question as
    "can I deploy to it now". BUSY and AVAILABLE come from whether this node is
    currently hosting a stage.
    """
    status = record.status.value
    if status in ("UNREACHABLE", "REGISTERING"):
        return "OFFLINE"
    if status == "DRAINING":
        return "MAINTENANCE"
    if status == "DEGRADED":
        return "DEGRADED"
    busy = False
    for deployment in deployments.all():
        # A stopped or failed deployment is not occupying the node, whatever its
        # last reported phase said.
        if deployment.status in ("STOPPED", "FAILED"):
            continue
        for stage in deployment.stages:
            if stage.node_id != record.node_id:
                continue
            phase = deployment.node_status.get(stage.node_id, {}).get("phase")
            if phase in ("running", "loading", "starting"):
                busy = True
    return "BUSY" if busy else "AVAILABLE"


def _series_sample(record: Any) -> dict[str, Any]:
    """One point from a heartbeat: enough to draw VRAM, utilisation and power."""
    inventory = getattr(record, "inventory", None)
    gpus = list(getattr(inventory, "gpus", []) or [])
    host = getattr(inventory, "host", None)

    def _max(values: list[Any]) -> float | None:
        present = [float(v) for v in values if v is not None]
        return max(present) if present else None

    return {
        "at": utcnow().isoformat(),
        "vram_total_bytes": sum(g.vram_total_bytes for g in gpus) if gpus else 0,
        "vram_free_bytes": sum(g.vram_free_bytes for g in gpus) if gpus else 0,
        "gpu_util_pct": _max([g.utilization_gpu_pct for g in gpus]),
        "temperature_c": _max([g.temperature_c for g in gpus]),
        "power_w": (
            round(sum(float(g.power_draw_w) for g in gpus if g.power_draw_w is not None), 2)
            if any(g.power_draw_w is not None for g in gpus)
            else None
        ),
        "ram_available_bytes": getattr(host, "ram_available_bytes", None),
    }


async def _reap_forever(state: "CoordinatorState") -> None:
    """Mark nodes that stopped heartbeating, on a timer rather than on request.

    Reaping used to require an admin POST, so a dead node stayed HEALTHY until
    somebody noticed. A node that dies mid-deployment now also fails that
    deployment, because a stage on a silent node is not running.
    """
    interval = max(2.0, state.settings.heartbeat_grace_s / 3.0)
    while True:
        await asyncio.sleep(interval)
        try:
            changed = state.registry.reap()
            if changed:
                affected = state.deployments.mark_unreachable([r for r in changed])
                if affected:
                    logger.warning(
                        "deployments failed after nodes went silent: %s", ", ".join(affected)
                    )
        except Exception:  # noqa: BLE001 - a reaper must not kill the loop
            logger.exception("reaper iteration failed")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state: CoordinatorState = app.state.mw
    task = asyncio.create_task(_reap_forever(state))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app(settings: CoordinatorSettings | None = None) -> FastAPI:
    settings = settings or CoordinatorSettings()
    state = CoordinatorState(settings)
    # Restore the cluster we knew before this process started, so a restart does
    # not silently drop every node.
    state.registry.load_snapshot()

    app = FastAPI(
        lifespan=_lifespan,
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
    async def register(registration: NodeRegistration, request: Request) -> dict[str, Any]:
        """Register an agent.

        Two paths:

          * **Open enrollment** -- the agent presents the shared enrollment token
            in ``x-meshweave-enrollment``. This is what lets a brand-new host join
            with a single command and no coordinator-issued credential.
          * **Closed** -- no enrollment token configured, or a wrong one. Then the
            node must already be known to the registry.

        Either way the node is identified by the node id the agent presents, never
        by its IP: two containers can share a public IP and differ by port.
        """
        supplied = request.headers.get("x-meshweave-enrollment", "")
        expected = state.settings.enrollment_token

        if expected:
            if not secrets.compare_digest(supplied, expected):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid enrollment token",
                )
        elif state.registry.get(registration.node_id) is None:
            # Enrollment closed and this node is unknown: refuse rather than
            # silently accepting an unauthenticated registration.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "open enrollment is disabled on this coordinator; set "
                    "MESHWEAVE_COORD_ENROLLMENT_TOKEN to enable it"
                ),
            )

        record = state.registry.register(registration)
        state.workloads.event(
            f"node:{registration.node_id}", "registered", {"node_id": registration.node_id}
        )
        # A credential the agent keeps and presents when it asks for work. Signed
        # with the coordinator secret, so verifying it needs no per-node state.
        node_token = issue_node_token(registration.node_id, state.secret)
        return {
            "node_token": node_token,
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
        buffer = state.series.setdefault(node_id, deque(maxlen=240))
        buffer.append(_series_sample(record))
        return {
            "ok": True,
            "accepted_at": utcnow().isoformat(),
            "reliability": record.reliability_score,
            "status": record.status.value,
        }

    @app.delete("/v1/nodes/{node_id}")
    async def forget_node(node_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Remove a node from the registry. Admin-only."""
        require_admin(state, authorization)
        removed = state.registry.forget(node_id)
        return {"node_id": node_id, "removed": removed}

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

    # -- agent work queue ------------------------------------------------

    @app.get("/v1/agents/{node_id}/work")
    async def agent_work(node_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """What this agent should do next.

        Returns an empty list when there is nothing to run. The closed set of
        commands is enforced by the agent, not by this endpoint: it may only
        execute a WorkloadAssignment, never an arbitrary string.
        """
        # The work queue is the one endpoint that can make a node start a
        # process, so it is the one that has to be authenticated. A node token
        # issued at registration proves which node is asking; the shared
        # enrollment token is still accepted so an agent that predates node
        # tokens keeps working during an upgrade.
        token = _bearer(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="missing agent token")

        presented_as = verify_node_token(token, state.secret)
        enrollment = state.settings.enrollment_token
        if presented_as != node_id and not (
            enrollment and secrets.compare_digest(token, enrollment)
        ):
            raise HTTPException(
                status_code=401,
                detail=(
                    "invalid node token; this agent must register to obtain one"
                    if presented_as is None
                    else f"token was issued for {presented_as}, not {node_id}"
                ),
            )

        record = state.registry.get(node_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown node")

        pending = [
            a
            for w in state.workloads.all()
            if w.finished_at is None
            for a in state.workloads.assignments_for(w.workload_id)
            if a.node_id == node_id
        ]
        return {
            "node_id": node_id,
            "commands": [a.model_dump(mode="json") for a in pending],
            # One command list, not two. Stages, prompts and link probes all
            # arrive here; a second key would have been invisible to the agents
            # already deployed, which read only this one.
            "deployment_commands": (
                state.deployments.commands_for(node_id)
                + state.network.commands_for(node_id)
            ),
            "heartbeat_interval_s": state.settings.heartbeat_timeout_s / 3.0,
        }

    # -- deployments -----------------------------------------------------
    #
    # A deployment is a model plus a chosen set of nodes, each running one stage
    # that holds a contiguous range of layers. The coordinator plans it, refuses
    # it with a reason when it cannot physically work, and then hands the stages
    # to the agents through the work queue. It never runs anything itself.

    async def _fetch_model_config(repo: str) -> str:
        """Read config.json from the hub. Planning needs the real geometry."""
        try:
            return await fetch_model_config(repo)
        except ModelConfigUnavailable as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    def _deployment_or_404(deployment_id: str) -> Deployment:
        deployment = state.deployments.get(deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="unknown deployment")
        return deployment

    def _summarise(deployment: Deployment) -> dict[str, Any]:
        nodes = [s.node_id for s in deployment.stages]
        phases = [deployment.node_status.get(n, {}).get("phase", "pending") for n in nodes]
        return {
            "deployment_id": deployment.deployment_id,
            "model": deployment.model,
            "status": deployment.status,
            "message": deployment.message,
            "created_at": deployment.created_at,
            "strategy": deployment.config.get("strategy"),
            "mode": deployment.mode,
            "replica_load": dict(deployment.replica_load),
            "dtype": deployment.config.get("dtype"),
            "layers": deployment.config.get("layers"),
            "stages": [
                {
                    "node_id": stage.node_id,
                    "rank": stage.rank,
                    "layers": stage.layers,
                    "count": stage.end - stage.start,
                    "resident_gib": round(stage.resident_bytes / GIB, 3),
                    "checkpoint_gib": round(stage.checkpoint_bytes / GIB, 2),
                    "address": stage.address,
                    "replica": stage.replica,
                    "phase": deployment.node_status.get(
                        stage_key_for(deployment.deployment_id, stage), {}
                    ).get("phase", "pending"),
                    "detail": deployment.node_status.get(
                        stage_key_for(deployment.deployment_id, stage), {}
                    ).get("detail"),
                }
                for stage in deployment.stages
            ],
            "phases": phases,
            "results": deployment.results[-10:],
            "pending_prompts": sum(1 for p in deployment.prompts if p.get("state") != "done"),
        }

    @app.post("/v1/deployments")
    async def create_deployment(request: DeploymentRequest) -> dict[str, Any]:
        try:
            repo = normalise_repo(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.cluster_id:
            cluster = state.clusters.get(request.cluster_id)
            if cluster is None:
                raise HTTPException(status_code=404, detail="unknown cluster")
            if request.node_ids and set(request.node_ids) != set(cluster.node_ids):
                raise HTTPException(
                    status_code=400,
                    detail="node_ids contradict the cluster membership; pass one or the other",
                )
            node_ids = list(cluster.node_ids)
        else:
            if not request.node_ids:
                raise HTTPException(
                    status_code=400,
                    detail="pass node_ids or a cluster_id naming the nodes to use",
                )
            node_ids = list(request.node_ids)

        records = []
        for node_id in node_ids:
            record = state.registry.get(node_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
            records.append(record)

        raw_config = await _fetch_model_config(repo)
        try:
            config = load_remote_config(raw_config)
        except Exception as exc:  # noqa: BLE001 - the reason matters more than the type
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        mode = request.mode.upper()
        if mode not in ("AUTO", "PIPELINE", "REPLICA"):
            raise HTTPException(
                status_code=400, detail="mode must be AUTO, PIPELINE or REPLICA"
            )

        chosen_reason = None
        if mode == "AUTO":
            # Never split what fits: a replica is faster per request and scales
            # with nodes, while a split pays a round trip per token.
            node_id, reason = best_single_node(
                records=records, config=config, max_seq=request.max_seq
            )
            mode = "REPLICA" if node_id else "PIPELINE"
            chosen_reason = reason

        try:
            if mode == "REPLICA":
                stages = plan_replicas(
                    records=records,
                    config=config,
                    max_seq=request.max_seq,
                    max_per_node=max(1, request.replicas_per_node),
                )
            else:
                stages = plan_deployment(
                    model=repo, records=records, config=config, max_seq=request.max_seq
                )
        except DeploymentRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        dtype = effective_dtype(config, records)
        deployment = new_deployment(repo, stages, config, request.max_seq, dtype=dtype)
        deployment.mode = mode
        if chosen_reason:
            deployment.message = chosen_reason
        if request.cluster_id:
            deployment.config["cluster_id"] = request.cluster_id
        state.deployments.add(deployment)
        state.workloads.event(
            f"dep:{deployment.deployment_id}",
            "created",
            {"model": repo, "nodes": [s.node_id for s in stages]},
        )
        return _summarise(deployment)

    @app.get("/v1/deployments")
    async def list_deployments() -> dict[str, Any]:
        return {"deployments": [_summarise(d) for d in state.deployments.all()]}

    @app.get("/v1/deployments/{deployment_id}")
    async def get_deployment(deployment_id: str) -> dict[str, Any]:
        return _summarise(_deployment_or_404(deployment_id))

    @app.post("/v1/deployments/{deployment_id}/nodes/{node_id}/status")
    async def report_node_status(
        deployment_id: str, node_id: str, report: DeploymentStatusReport
    ) -> dict[str, Any]:
        deployment = _deployment_or_404(deployment_id)
        if state.deployments.stage_for(deployment_id, node_id) is None:
            raise HTTPException(status_code=404, detail=f"{node_id} has no stage here")
        stage_key = report.stage_key or f"{deployment_id}#{node_id}#0"

        previous = deployment.node_status.get(stage_key, {})
        if previous.get("phase") == "failed" and report.phase == "stopped":
            # The stage already failed; the reason is the useful part. Keep it
            # rather than replacing it with the shutdown notice.
            return _summarise(deployment)

        phase = report.phase
        detail = report.detail or ""
        # The agent infers a stage's phase from its log, looking for the word
        # "ready". Runners before 0.4.2 do not print that marker for a
        # single-stage process, so a loaded and listening stage reported itself as
        # still loading forever. The listener line is the real evidence.
        if phase == "loading" and (
            "listening on" in detail or "control socket" in detail
        ):
            phase = "running"

        entry = {
            "phase": phase,
            "detail": report.detail,
            "pid": report.pid,
            "at": utcnow().isoformat(),
        }

        # Count launches, not failure reports: a stage that exits while its peer
        # is still downloading would otherwise be "retried" every heartbeat.
        previous = deployment.node_status.get(stage_key, {})
        attempts = int(previous.get("attempts") or 0)
        if report.phase == "starting":
            attempts += 1
        entry["attempts"] = attempts

        # A stage that dies before serving any request is a startup race, not a
        # verdict on the deployment: the first stage may be waiting on a peer that
        # is still fetching a checkpoint. Retry a bounded number of times, then
        # give up with the reason.
        serving = any(
            (
                deployment.node_status.get(
                    stage_key_for(deployment.deployment_id, s), {}
                ).get("phase")
                == "running"
            )
            for s in deployment.stages
        )
        if report.phase == "failed" and not serving and attempts < MAX_STAGE_ATTEMPTS:
            entry["phase"] = "restarting"
            entry["detail"] = (
                f"attempt {attempts} failed before this stage served a request; "
                "waiting for the other stages to come up"
            )

        deployment.node_status[stage_key] = entry

        # A stopped deployment stays stopped. Without this, the status roll-up
        # below recomputed STOPPED as LOADING, commands_for then re-queued the
        # stage, and the nodes restarted work that had just been cancelled.
        if deployment.stop_requested:
            state.deployments.touch(deployment)
            return _summarise(deployment)

        phases = [
            deployment.node_status.get(
                stage_key_for(deployment.deployment_id, stage), {}
            ).get("phase", "pending")
            for stage in deployment.stages
        ]
        if "failed" in phases:
            deployment.status = "FAILED"
            deployment.message = next(
                (
                    deployment.node_status[stage_key_for(deployment.deployment_id, st)].get("detail")
                    for st in deployment.stages
                    if deployment.node_status.get(
                        stage_key_for(deployment.deployment_id, st), {}
                    ).get("phase")
                    == "failed"
                ),
                None,
            )
        elif all(p == "running" for p in phases):
            deployment.status = "RUNNING"
        elif any(p in ("loading", "starting", "running", "restarting") for p in phases):
            deployment.status = "LOADING"
        state.deployments.touch(deployment)
        return _summarise(deployment)

    @app.post("/v1/deployments/{deployment_id}/result")
    async def report_result(deployment_id: str, result: DeploymentResult) -> dict[str, Any]:
        deployment = _deployment_or_404(deployment_id)
        payload = result.model_dump()
        payload["at"] = utcnow().isoformat()
        deployment.results.append(payload)
        if deployment.mode == "REPLICA":
            # Prefer the node that reported; fall back to the node the prompt
            # was assigned to, so a result that omits its origin still releases
            # the load it was counted against.
            key = payload.get("stage_key") or next(
                (
                    p.get("assigned_replica")
                    for p in deployment.prompts
                    if p.get("id") == payload.get("prompt_id")
                ),
                None,
            )
            if key and deployment.replica_load.get(key):
                deployment.replica_load[key] -= 1
        for prompt in deployment.prompts:
            if prompt.get("id") == result.prompt_id:
                prompt["state"] = "done"
        if result.error:
            deployment.message = result.error
        state.deployments.touch(deployment)
        return {"recorded": True, "results": len(deployment.results)}

    @app.post("/v1/deployments/{deployment_id}/prompt")
    async def submit_prompt(deployment_id: str, request: PromptRequest) -> dict[str, Any]:
        deployment = _deployment_or_404(deployment_id)
        if deployment.status not in ("RUNNING", "LOADING"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"deployment is {deployment.status}; wait until every stage reports "
                    "running before sending prompts"
                ),
            )
        prompt_id = f"p-{secrets.token_hex(4)}"
        assigned = None
        if deployment.mode == "REPLICA":
            def _healthy(stage_key: str) -> bool:
                info = deployment.node_status.get(stage_key, {})
                if info.get("phase") != "running":
                    return False
                # Keys are deployment#node#replica; the node is the middle field.
                parts = stage_key.split("#")
                node_id = parts[1] if len(parts) > 1 else stage_key
                record = state.registry.get(node_id)
                return bool(record and record.status.value == "HEALTHY")

            assigned = choose_replica(deployment, _healthy)
            if assigned is None:
                raise HTTPException(
                    status_code=503,
                    detail="no healthy replica is available to take this request",
                )
            deployment.replica_load[assigned] = deployment.replica_load.get(assigned, 0) + 1

        deployment.prompts.append(
            {
                "id": prompt_id,
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "state": "queued",
                "assigned_replica": assigned,
                "at": utcnow().isoformat(),
            }
        )
        state.deployments.touch(deployment)
        return {"prompt_id": prompt_id, "queued": True}

    @app.post("/v1/deployments/{deployment_id}/stop")
    async def stop_deployment(deployment_id: str) -> dict[str, Any]:
        deployment = _deployment_or_404(deployment_id)
        deployment.stop_requested = True
        deployment.status = "STOPPED"
        deployment.message = "stop requested"
        state.deployments.touch(deployment)
        return _summarise(deployment)

    @app.delete("/v1/deployments/{deployment_id}")
    async def forget_deployment(deployment_id: str) -> dict[str, Any]:
        deployment = _deployment_or_404(deployment_id)
        deployment.stop_requested = True
        state.deployments.touch(deployment)
        removed = state.deployments.remove(deployment_id)
        return {"deployment_id": deployment_id, "removed": removed}

    # -- virtual clusters -------------------------------------------------
    #
    # A cluster names a group of physical machines. It does not create a larger
    # GPU: aggregate capacity is reported as distributed, with a note that it is
    # not addressable as one pool.

    def _availability_of(record) -> str:
        """Availability takes a node record; cluster_status resolves members first."""
        if record is None:
            return "OFFLINE"
        return _availability(record, state.deployments)

    @app.post("/v1/clusters")
    async def create_cluster(request: ClusterRequest) -> dict[str, Any]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="cluster name is required")
        if state.clusters.by_name(name) is not None:
            raise HTTPException(status_code=409, detail=f"a cluster named {name!r} exists")

        records = []
        for node_id in request.node_ids:
            record = state.registry.get(node_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
            records.append(record)

        validation = validate_members(records, state.topology)
        if not validation.ok:
            raise HTTPException(status_code=422, detail=validation.errors)

        cluster = state.clusters.create(name, [r.node_id for r in records], validation.warnings)
        return cluster_status(cluster, state.registry.get, _availability_of)

    @app.get("/v1/clusters")
    async def list_clusters() -> dict[str, Any]:
        return {
            "clusters": [
                cluster_status(c, state.registry.get, _availability_of)
                for c in state.clusters.all()
            ]
        }

    @app.get("/v1/clusters/{cluster_id}")
    async def get_cluster(cluster_id: str) -> dict[str, Any]:
        cluster = state.clusters.get(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail="unknown cluster")
        return cluster_status(cluster, state.registry.get, _availability_of)

    @app.post("/v1/clusters/{cluster_id}/validate")
    async def validate_cluster(cluster_id: str) -> dict[str, Any]:
        """Re-run the static checks against what the members report right now."""
        cluster = state.clusters.get(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail="unknown cluster")
        records = [state.registry.get(n) for n in cluster.node_ids]
        present = [r for r in records if r is not None]
        missing = [n for n in cluster.node_ids if n not in {r.node_id for r in present}]
        validation = validate_members(present, state.topology)
        for node_id in missing:
            validation.errors.append(f"{node_id} is no longer registered")
        state.clusters.update_warnings(cluster, validation.warnings)
        return {
            "cluster_id": cluster_id,
            "ok": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "missing": missing,
        }

    @app.delete("/v1/clusters/{cluster_id}")
    async def delete_cluster(cluster_id: str) -> dict[str, Any]:
        removed = state.clusters.remove(cluster_id)
        return {"cluster_id": cluster_id, "removed": removed}

    # -- network intelligence --------------------------------------------

    @app.get("/v1/network/intelligence")
    async def network_intelligence() -> dict[str, Any]:
        """Score the latest measured links per strategy, against measured compute.

        Compute time comes from the fastest successful benchmark we have for the
        model being considered. When there is none, every verdict is reported as
        indeterminate rather than estimated, because the comparison is the point.
        """
        from meshweave.network.intelligence import LinkView, assess_strategy

        if state.topology is None:
            raise HTTPException(
                status_code=404,
                detail="no links measured yet; run POST /v1/network/measure first",
            )

        compute_ms = state.registry.fastest_compute_ms_per_token()
        assessed = []
        for link in state.topology.links:
            view = LinkView(
                source=link.source,
                target=link.target,
                rtt_p50_ms=link.rtt_ms_min or 0.0,
                jitter_ms=link.jitter_ms,
                download_mbps=link.tcp_mbps_single_stream,
                ok=bool(link.connection_success),
            )
            result = assess_strategy(
                view,
                compute_ms_per_token=compute_ms,
                hops=2,
                bytes_per_token=1792.0,
            )
            result["computed_from"] = {
                "compute_ms_per_token": compute_ms,
                "source": "fastest successful benchmark" if compute_ms else "unavailable",
            }
            assessed.append(result)

        return {
            "links": assessed,
            "note": (
                "Verdicts compare measured communication time against measured "
                "compute time. No verdict is produced without a compute time."
            ),
        }

    # -- network measurement ---------------------------------------------
    #
    # Measurement is orchestrated, not dialled: the coordinator tells each node
    # to listen, then tells each node to probe the others. Both directions are
    # measured separately, because a link is not symmetric.

    @app.post("/v1/network/measure")
    async def start_measurement(request: NetworkMeasureRequest) -> dict[str, Any]:
        for node_id in request.node_ids:
            if state.registry.get(node_id) is None:
                raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
        try:
            measurement = state.network.start(
                request.node_ids,
                {"rtt_count": request.rtt_count, "bulk_bytes": request.bulk_bytes},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if measurement.phase == "failed":
            raise HTTPException(status_code=409, detail=measurement.error)
        return measurement.to_dict()

    @app.get("/v1/network/measurements")
    async def list_measurements() -> dict[str, Any]:
        items = []
        for measurement in state.network.all():
            summary = measurement.to_dict()
            summary.pop("results", None)
            items.append(summary)
        return {"measurements": items}

    @app.get("/v1/network/measurements/{measurement_id}")
    async def get_measurement(measurement_id: str) -> dict[str, Any]:
        measurement = state.network.get(measurement_id)
        if measurement is None:
            raise HTTPException(status_code=404, detail="unknown measurement")
        return measurement.to_dict()

    @app.post("/v1/nodes/{node_id}/network-result")
    async def report_network(
        node_id: str, report: NetworkProbeReport
    ) -> dict[str, Any]:
        if state.registry.get(node_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
        measurement = state.network.record(
            node_id, report.measurement_id, report.phase, report.payload
        )
        if measurement is None:
            raise HTTPException(status_code=404, detail="unknown measurement")

        if measurement.phase == "done":
            matrix = state.network.topology(measurement)
            if matrix is not None:
                state.set_topology(matrix)
        return {
            "measurement_id": measurement.measurement_id,
            "phase": measurement.phase,
            "complete": measurement.complete,
        }

    # -- metrics / evidence ---------------------------------------------

    @app.get("/v1/readiness")
    async def readiness() -> dict[str, Any]:
        """What inference stack each node reports, and whether it can serve.

        Read from what agents reported. A node that has not reported a stack is
        listed as unknown rather than assumed capable.
        """
        nodes: list[dict[str, Any]] = []
        for record in state.registry.all():
            # inventory is a pydantic model, so read attributes rather than keys.
            inv = record.inventory
            gpus = inv.gpus if inv else []
            vram = sum(g.vram_total_bytes for g in gpus) / (1024**3) if gpus else 0.0
            host = inv.host if inv else None
            net = inv.network if inv else None

            def _gib(value: int | None) -> float | None:
                return round(value / (1024**3), 1) if value else None

            # Where another node could reach this node's stage listener. The
            # private address only works inside one network; the published
            # mappings are what make a container's port reachable across
            # providers. Empty when the provider declares none -- reported, not
            # guessed.
            mapped = {str(k): v for k, v in (net.mapped_ports if net else {}).items()}
            nodes.append(
                {
                    "node_id": record.node_id,
                    "status": record.status.value,
                    "gpus": len(gpus),
                    "gpu_name": gpus[0].name if gpus else None,
                    "vram_gib": round(vram, 1),
                    "can_host": bool(gpus) and record.status.value == "HEALTHY",
                    "availability": _availability(record, state.deployments),
                    "agent_version": inv.agent_version if inv else None,
                    "cpu_cores": host.cpu_cores if host else None,
                    "cpu_model": host.cpu_model if host else None,
                    "ram_total_gib": _gib(host.ram_total_bytes) if host else None,
                    "ram_available_gib": _gib(host.ram_available_bytes) if host else None,
                    "disk_total_gib": _gib(host.disk_total_bytes) if host else None,
                    "disk_available_gib": _gib(host.disk_available_bytes) if host else None,
                    "os_name": host.os_name if host else None,
                    "private_ip": net.private_ip if net else None,
                    "external_ip": net.external_ip if net else None,
                    "mapped_ports": mapped,
                }
            )
        return {
            "nodes": nodes,
            "ready": sum(1 for n in nodes if n["can_host"]),
            "total": len(nodes),
            "note": (
                "can_host means the node reported a GPU and is healthy. Whether an "
                "inference stack is installed is a separate question this endpoint "
                "does not answer."
            ),
        }

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

    @app.get("/v1/experiments")
    async def list_experiments() -> dict[str, Any]:
        """Index the experiment reports written to disk."""
        import json as _json

        raw_dir = REPO_ROOT / "reports" / "raw"
        entries: list[dict[str, Any]] = []
        if raw_dir.is_dir():
            for path in sorted(raw_dir.glob("*.json")):
                if path.name.endswith("-topology.json") or "-comparison" in path.name:
                    continue
                try:
                    with path.open() as fh:
                        data = _json.load(fh)
                except (OSError, _json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or "experiment_id" not in data:
                    continue
                entries.append(
                    {
                        "experiment_id": data.get("experiment_id"),
                        "title": data.get("title"),
                        "status": data.get("status"),
                        "synthetic": bool(
                            (data.get("provenance") or {}).get("synthetic", False)
                        ),
                        "measurement_count": len(data.get("measurements") or []),
                        "failure_count": len(
                            (data.get("failures") or {}).get("failures") or []
                        ),
                        "blocked_reason": data.get("blocked_reason"),
                        "path": str(path.relative_to(REPO_ROOT)),
                    }
                )
        entries.sort(key=lambda e: str(e.get("experiment_id") or ""))
        return {
            "experiments": entries,
            "total": len(entries),
            "completed": sum(1 for e in entries if e["status"] == "completed"),
            "blocked": sum(1 for e in entries if e["status"] == "blocked"),
        }

    @app.get("/v1/network")
    async def get_network() -> dict[str, Any]:
        """Phase 2A network measurements, read from the experiment evidence.

        Everything here comes from a file written by an instrumented transfer
        between real hosts. When a file is absent, the section is reported as
        unmeasured rather than filled with zeros.
        """
        import json as _json

        raw = REPO_ROOT / "experiments" / "network" / "raw"

        def _load(name: str) -> Any | None:
            path = raw / name
            if not path.exists():
                return None
            try:
                with path.open() as fh:
                    return _json.load(fh)
            except (OSError, _json.JSONDecodeError):
                return None

        tcp = _load("direct_tcp_matrix.json")
        rtt = _load("rtt_bandwidth.json")
        coloc = _load("colocation.json")
        coloc_net = _load("colocation_network.json")

        paths = []
        if rtt:
            for entry in rtt.get("paths", []):
                rtt_data = entry.get("rtt") or {}
                sustained = entry.get("sustained_15s") or {}
                paths.append(
                    {
                        "source": entry["source"],
                        "target": entry["target"],
                        "endpoint": entry.get("endpoint"),
                        "rtt_p50_ms": rtt_data.get("p50_ms"),
                        "rtt_p95_ms": rtt_data.get("p95_ms"),
                        "rtt_p99_ms": rtt_data.get("p99_ms"),
                        "jitter_ms": rtt_data.get("jitter_ms"),
                        "packet_loss_pct": (
                            rtt_data.get("failed_probes", 0) / rtt_data["probes"] * 100.0
                            if rtt_data.get("probes")
                            else None
                        ),
                        "sustained_mbps": sustained.get("throughput_mbps"),
                        "measured": rtt_data.get("ok", False),
                    }
                )

        return {
            "measured": bool(paths),
            "paths": paths,
            "topology": {"nodes": list((tcp or {}).get("endpoints", {}).keys())},
            "endpoints": (tcp or {}).get("endpoints", {}),
            "co-location": _colocation_summary(coloc, coloc_net),
            "source": "experiments/network/raw/",
        }

    @app.get("/v1/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict[str, Any]:
        """Return one full experiment report."""
        import json as _json

        raw_dir = REPO_ROOT / "reports" / "raw"
        target = raw_dir / f"{experiment_id.lower()}.json"
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"no report for {experiment_id}")
        try:
            with target.open() as fh:
                return dict(_json.load(fh))
        except (OSError, _json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"report unreadable: {exc}"
            ) from exc

    @app.get("/v1/nodes/{node_id}/series")
    async def node_series(node_id: str, limit: int = 120) -> dict[str, Any]:
        """Recent telemetry for one node.

        Session-only: this is what the coordinator saw while it was running, not
        a persisted time series, and the response says so rather than implying
        history it does not have.
        """
        if state.registry.get(node_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
        samples = list(state.series.get(node_id, deque()))[-max(1, min(limit, 240)):]
        return {
            "node_id": node_id,
            "samples": samples,
            "count": len(samples),
            "retention": (
                "in memory only; cleared when the coordinator restarts, and one "
                "sample per heartbeat"
            ),
        }

    @app.get("/v1/nodes/{node_id}")
    async def get_node(node_id: str) -> dict[str, Any]:
        """One node's full record, including every GPU it reports."""
        record = state.registry.get(node_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id}")

        environment: str | None = None
        if state.graph is not None:
            env = state.graph.environment_by_node.get(node_id)
            if env is not None:
                environment = env.value

        return {
            "node": record.model_dump(mode="json"),
            "degrade_reason": state.registry.degrade_reasons.get(node_id),
            "environment": environment,
        }

    return


def _colocation_summary(coloc: Any, coloc_net: Any) -> dict[str, Any]:
    """Condense the co-location evidence into a dashboard-friendly shape."""
    if not coloc:
        return {"available": False}

    pairs = coloc.get("pairs", []) if isinstance(coloc, dict) else []
    hosts = coloc.get("hosts", []) if isinstance(coloc, dict) else []
    adjacency = (coloc_net or {}).get("network_adjacency", {}) if isinstance(coloc_net, dict) else {}

    return {
        "available": True,
        "pairs": [
            {
                "a": p.get("a"),
                "b": p.get("b"),
                "shared_public_ip": p.get("shared_public_ip"),
                "verdict": p.get("verdict"),
            }
            for p in pairs
        ],
        "network_adjacency": adjacency,
        "host_count": len(hosts),
    }


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
