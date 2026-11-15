"""Topology engine.

Turns raw per-pair measurements into a graph the planner can reason about. The
environment for a link is decided from measured latency, not from what a host
advertises. A host saying "10 Gbps" proves nothing about what an activation
transfer will actually cost.

Thresholds live in one place so they can be challenged in DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meshweave.core import Environment, LinkKind, LinkMeasurement, TopologyMatrix

# Measured RTT (p50) boundaries, milliseconds. Chosen to reflect what each
# parallelism strategy can tolerate, not hardware marketing.
DATACENTER_RTT_MAX_MS = 2.0
REGIONAL_RTT_MAX_MS = 20.0

# Minimum sustained bandwidth (Mbps) for a link to be considered usable at all.
MIN_USABLE_MBPS = 5.0


def classify_link(link: LinkMeasurement) -> Environment:
    """Classify a link from its measurement.

    Ordering matters: a loopback link is always datacenter-grade.
    """
    if link.error is not None or not link.usable:
        # A failed probe is not evidence of a fast link. Treat as internet-grade
        # unknown so the planner does not over-promise.
        return Environment.C_INTERNET

    if link.link_kind in (LinkKind.NVLINK, LinkKind.INFINIBAND, LinkKind.ROCE):
        return Environment.A_DATACENTER
    if link.link_kind == LinkKind.LOOPBACK:
        return Environment.A_DATACENTER
    if link.link_kind == LinkKind.OVERLAY:
        # Overlay still runs over the underlying physical link. Classify by
        # measurement, not by the fact that it is an overlay.
        pass

    rtt = link.rtt_ms_p50
    if rtt is None:
        return Environment.C_INTERNET
    if rtt <= DATACENTER_RTT_MAX_MS:
        return Environment.A_DATACENTER
    if rtt <= REGIONAL_RTT_MAX_MS:
        return Environment.B_REGIONAL
    return Environment.C_INTERNET


@dataclass
class NodeProfile:
    """What the topology engine knows about one node, for grouping."""

    node_id: str
    total_vram_gib: float
    free_vram_gib: float
    gpu_name: str
    compute_capability: tuple[int, int] | None
    bandwidth_gbps: float | None
    location: str | None = None
    environment: Environment = Environment.C_INTERNET
    reliability: float | None = None
    hourly_cost_usd: float | None = None


@dataclass
class TopologyGraph:
    nodes: list[str]
    links: list[LinkMeasurement]
    matrix: TopologyMatrix
    environment_by_node: dict[str, Environment] = field(default_factory=dict)

    def link(self, a: str, b: str) -> LinkMeasurement | None:
        return self.matrix.get(a, b)

    def environment(self, a: str, b: str) -> Environment:
        link = self.link(a, b)
        if link is None:
            return Environment.C_INTERNET
        return classify_link(link)

    def usable_pairs(self) -> list[tuple[str, str, LinkMeasurement]]:
        out = []
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1 :]:
                link = self.link(a, b)
                if link and link.usable:
                    out.append((a, b, link))
        return out

    def bottleneck(self, node_ids: list[str]) -> float | None:
        """Slowest measured sustained bandwidth among the given nodes, Mbps."""
        values: list[float] = []
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1 :]:
                link = self.link(a, b)
                if link and link.sustained_mbps is not None:
                    values.append(link.sustained_mbps)
        return min(values) if values else None

    def max_rtt_ms(self, node_ids: list[str]) -> float | None:
        values: list[float] = []
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1 :]:
                link = self.link(a, b)
                if link and link.rtt_ms_p50 is not None:
                    values.append(link.rtt_ms_p50)
        return max(values) if values else None


def build_graph(matrix: TopologyMatrix) -> TopologyGraph:
    env_by_node: dict[str, Environment] = {}
    for node in matrix.nodes:
        # A node's environment is the *worst* environment among its links, so a
        # single far-away peer prevents us treating the node as datacenter-grade.
        worst = Environment.A_DATACENTER
        for other in matrix.nodes:
            if other == node:
                continue
            link = matrix.get(node, other)
            if link is None:
                continue
            env = classify_link(link)
            if env.value > worst.value:
                worst = env
        env_by_node[node] = worst
    return TopologyGraph(nodes=list(matrix.nodes), links=list(matrix.links), matrix=matrix, environment_by_node=env_by_node)


def require_datacenter(graph: TopologyGraph, node_ids: list[str], min_sustained_mbps: float) -> str | None:
    """Return a rejection reason if these nodes cannot support tight coupling.

    Used to gate tensor parallelism. Returns ``None`` when the group qualifies.
    """
    if len(node_ids) < 2:
        return "tensor parallelism requires at least two nodes"
    worst_env = Environment.A_DATACENTER
    for a, b in _pairs(node_ids):
        env = graph.environment(a, b)
        if env.value > worst_env.value:
            worst_env = env
    if worst_env is not Environment.A_DATACENTER:
        return (
            f"tensor parallelism requires {Environment.A_DATACENTER.value} links; "
            f"measured {worst_env.value}"
        )
    bn = graph.bottleneck(node_ids)
    if bn is None:
        return "no sustained-bandwidth measurement for the candidate group"
    if bn < min_sustained_mbps:
        return f"sustained bandwidth {bn:.0f} Mbps below the {min_sustained_mbps:.0f} Mbps floor"
    return None


def _pairs(node_ids: list[str]) -> list[tuple[str, str]]:
    return [(node_ids[i], node_ids[j]) for i in range(len(node_ids)) for j in range(i + 1, len(node_ids))]


def pair_count(n: int) -> int:
    """Unique unordered pairs for n nodes (8 nodes -> 28 pairs)."""
    return n * (n - 1) // 2
