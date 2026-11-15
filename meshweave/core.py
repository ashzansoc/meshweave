"""Canonical data model for MESHWEAVE.

Everything on the wire and in the registry is expressed in these types. They are
deliberately serialisable (pydantic) so that the coordinator, the agents, the
planner and the report validator all agree on one schema.

Design rule: no field here is decorative. If a value is not measured or supplied
by a real source, it is ``None`` -- never zero, never a plausible-looking guess.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class Environment(enum.StrEnum):
    """The three infrastructure environments from the MESHWEAVE architecture.

    Assigned from *measured* link characteristics, never from what a provider
    advertises about its hardware.
    """

    A_DATACENTER = "A_DATACENTER"  # NVLink / InfiniBand / RoCE / high-speed Ethernet
    B_REGIONAL = "B_REGIONAL"  # reasonably fast Ethernet across machines
    C_INTERNET = "C_INTERNET"  # ordinary internet, NAT, variable bandwidth


class ParallelStrategy(enum.StrEnum):
    """Execution strategies the planner may select."""

    SINGLE_GPU = "SINGLE_GPU"
    REPLICATED = "REPLICATED"
    PIPELINE = "PIPELINE"
    TENSOR = "TENSOR"
    EXPERT = "EXPERT"
    DDP = "DDP"
    FSDP = "FSDP"
    DEPSPEED_ZERO = "DEEPSPEED_ZERO"
    ASYNC_DISTRIBUTED = "ASYNC_DISTRIBUTED"


class NodeStatus(enum.StrEnum):
    REGISTERING = "REGISTERING"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNREACHABLE = "UNREACHABLE"
    DRAINING = "DRAINING"
    RETIRED = "RETIRED"


class WorkloadStatus(enum.StrEnum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    LOADING = "LOADING"
    RUNNING = "RUNNING"
    STREAMING = "STREAMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DeviceKind(enum.StrEnum):
    DISCRETE_GPU = "DISCRETE_GPU"
    INTEGRATED_GPU = "INTEGRATED_GPU"


class LinkKind(enum.StrEnum):
    """Physical link between two GPUs or two nodes."""

    NVLINK = "NVLINK"
    INFINIBAND = "INFINIBAND"
    ROCE = "ROCE"
    PCIE = "PCIE"
    ETHERNET_10G = "ETHERNET_10G"
    ETHERNET_1G = "ETHERNET_1G"
    ETHERNET_100M = "ETHERNET_100M"
    INTERNET = "INTERNET"
    LOOPBACK = "LOOPBACK"
    OVERLAY = "OVERLAY"


# --------------------------------------------------------------------------- #
# GPU / node
# --------------------------------------------------------------------------- #


class GpuDevice(BaseModel):
    """A single physical GPU as reported by NVML on the agent host."""

    index: int
    uuid: str
    name: str
    manufacturer: str = "NVIDIA"
    architecture: str | None = None  # e.g. "Ada Lovelace", "Ampere"
    compute_capability: tuple[int, int] | None = None  # (major, minor)
    cuda_version: str | None = None
    driver_version: str | None = None

    vram_total_bytes: int
    vram_free_bytes: int
    vram_used_bytes: int

    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    utilization_gpu_pct: float | None = None
    utilization_memory_pct: float | None = None

    # Sustained figures matter far more than nameplate for planning.
    memory_bandwidth_gbps: float | None = None
    kind: DeviceKind = DeviceKind.DISCRETE_GPU
    pci_bus_id: str | None = None

    @property
    def vram_total_gib(self) -> float:
        return self.vram_total_bytes / (1024**3)

    @property
    def vram_free_gib(self) -> float:
        return self.vram_free_bytes / (1024**3)


class HostResources(BaseModel):
    cpu_model: str | None = None
    cpu_cores: int | None = None
    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None
    disk_total_bytes: int | None = None
    disk_available_bytes: int | None = None
    os_name: str | None = None
    kernel_version: str | None = None


class NetworkInfo(BaseModel):
    """What the agent knows about its own reachability.

    ``external_ip``/``mapped_ports`` are read from the container environment
    (Vast exposes ``VAST_TCP_PORT_x``); they are not guessed.
    """

    hostname: str | None = None
    private_ip: str | None = None
    external_ip: str | None = None
    isp: str | None = None
    geolocation: str | None = None
    nat: bool | None = None
    mapped_ports: dict[int, int] = Field(default_factory=dict)  # internal -> external TCP
    overlay_interface: str | None = None
    advertised_down_mbps: float | None = None
    advertised_up_mbps: float | None = None


class GpuInventory(BaseModel):
    """Everything one agent reports about its host in a single heartbeat."""

    gpus: list[GpuDevice]
    host: HostResources = Field(default_factory=HostResources)
    network: NetworkInfo = Field(default_factory=NetworkInfo)
    agent_version: str
    collected_at: datetime = Field(default_factory=utcnow)


class NodeRegistration(BaseModel):
    """First contact from an agent."""

    node_id: str
    hostname: str
    agent_version: str
    provider: str  # "vast" | "manual" | "local" ...
    provider_instance_id: str | None = None
    region: str | None = None
    registered_at: datetime = Field(default_factory=utcnow)


class NodeRecord(BaseModel):
    """Live registry entry. Reliability is earned, not configured."""

    node_id: str
    registration: NodeRegistration
    inventory: GpuInventory | None = None
    status: NodeStatus = NodeStatus.REGISTERING

    # Earned metrics (0.0 - 1.0); None until enough observations exist.
    reliability_score: float | None = None
    uptime_fraction: float | None = None
    observations: int = 0
    missed_heartbeats: int = 0
    last_heartbeat_at: datetime | None = None

    # Cost accounting. None means "we do not know what this costs".
    hourly_cost_usd: float | None = None

    @property
    def gpus(self) -> list[GpuDevice]:
        return self.inventory.gpus if self.inventory else []

    @property
    def vram_total_bytes(self) -> int:
        return sum(g.vram_total_bytes for g in self.gpus)

    @property
    def vram_free_bytes(self) -> int:
        return sum(g.vram_free_bytes for g in self.gpus)

    def gpu_uuid_set(self) -> set[str]:
        return {g.uuid for g in self.gpus}


# --------------------------------------------------------------------------- #
# Network measurement
# --------------------------------------------------------------------------- #


class LinkMeasurement(BaseModel):
    """One measured directed link between two endpoints.

    Every field is a measurement. ``error`` is set when the probe failed, and in
    that case the numeric fields are ``None`` rather than zero.
    """

    source: str
    target: str
    measured_at: datetime = Field(default_factory=utcnow)

    # Latency
    rtt_ms_p50: float | None = None
    rtt_ms_p95: float | None = None
    rtt_ms_min: float | None = None
    one_way_ms_estimate: float | None = None

    # Stability
    jitter_ms: float | None = None
    packet_loss_pct: float | None = None

    # Throughput (TCP)
    tcp_mbps_single_stream: float | None = None
    tcp_mbps_multi_stream: float | None = None
    sustained_mbps: float | None = None  # long-running, after ramp-up
    transfer_time_s: float | None = None  # for a fixed reference payload

    # Connection behaviour
    connection_success: bool | None = None
    connection_stability_pct: float | None = None
    udp_supported: bool | None = None

    link_kind: LinkKind = LinkKind.INTERNET
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.rtt_ms_p50 is not None


class TopologyMatrix(BaseModel):
    """The full N x N measured matrix the planner consumes."""

    generated_at: datetime = Field(default_factory=utcnow)
    nodes: list[str]
    links: list[LinkMeasurement] = Field(default_factory=list)

    def get(self, source: str, target: str) -> LinkMeasurement | None:
        for link in self.links:
            if link.source == source and link.target == target:
                return link
        return None

    def pair_count(self) -> int:
        return len(self.nodes) * (len(self.nodes) - 1)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


class ExecutionPlan(BaseModel):
    """A concrete proposal for running one workload.

    ``rejected_reason`` is mandatory when ``viable`` is false. The planner must
    be able to say no, and must say why.
    """

    plan_id: str
    model: str
    strategy: ParallelStrategy
    environment: Environment

    # stage_index -> node_id (ordered pipeline stages, for PIPELINE)
    stage_assignment: dict[int, str] = Field(default_factory=dict)
    # node_id -> layer range [start, end) (for PIPELINE)
    layer_assignment: dict[str, list[int]] = Field(default_factory=dict)
    replica_nodes: list[str] = Field(default_factory=list)

    estimated_ttft_ms: float | None = None
    estimated_itl_ms: float | None = None
    estimated_throughput_tps: float | None = None
    estimated_cost_per_m_output_tokens: float | None = None

    viable: bool = True
    rejected_reason: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Why the planner chose this plan, and what it rejected. A plan without a
    # rationale is not auditable.
    rationale: list[str] = Field(default_factory=list)
    # What happens to a running request if a node in this plan disappears.
    disconnect_impact: str | None = None

    def check(self) -> None:
        if not self.viable and not self.rejected_reason:
            raise ValueError("an unviable plan must state a rejected_reason")


class PlanningRequest(BaseModel):
    model: str
    prompt_tokens: int = 0
    max_new_tokens: int = 256
    concurrency: int = 1
    quantization: str = "fp16"  # fp16 | int8 | int4
    preferred_nodes: list[str] = Field(default_factory=list)
    forbid_strategies: list[ParallelStrategy] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #


class WorkloadRequest(BaseModel):
    workload_id: str
    model: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    strategy: ParallelStrategy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTransfer(BaseModel):
    """One activation transfer between adjacent pipeline stages."""

    from_stage: int
    to_stage: int
    bytes_sent: int
    send_ms: float
    wire_rtt_ms: float | None = None
    direction: str = "forward"


class MemorySample(BaseModel):
    node_id: str
    gpu_index: int
    gpu_uuid: str
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    process_rss_bytes: int | None = None
    sampled_at: datetime = Field(default_factory=utcnow)


class ExecutionRecord(BaseModel):
    """Per-request execution facts. The unit of evidence for every claim."""

    workload_id: str
    model: str
    strategy: ParallelStrategy
    nodes: list[str]
    environment: Environment

    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    inter_token_latency_ms_p50: float | None = None
    inter_token_latency_ms_p95: float | None = None
    total_generation_s: float | None = None
    tokens_per_second: float | None = None

    model_load_s: float | None = None
    transfers: list[StageTransfer] = Field(default_factory=list)
    memory_samples: list[MemorySample] = Field(default_factory=list)
    node_failure: str | None = None

    # Economic. Populated from registry hourly_cost_usd; None if unknown.
    cost_usd: float | None = None
    cost_per_m_output_tokens_usd: float | None = None

    output_text: str | None = None
    output_hash: str | None = None
    success: bool = True
    error: str | None = None


# --------------------------------------------------------------------------- #
# Agent protocol
# --------------------------------------------------------------------------- #


class Heartbeat(BaseModel):
    node_id: str
    sent_at: datetime = Field(default_factory=utcnow)
    inventory: GpuInventory
    active_workloads: list[str] = Field(default_factory=list)


class WorkloadAssignment(BaseModel):
    workload_id: str
    node_id: str
    model: str
    stage_index: int | None = None
    layer_start: int | None = None
    layer_end: int | None = None
    peer_endpoints: list[str] = Field(default_factory=list)  # adjacent stage URLs
    is_first_stage: bool = False
    is_last_stage: bool = False
    quantization: str = "fp16"


class CommandResult(BaseModel):
    workload_id: str
    node_id: str
    ok: bool
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
