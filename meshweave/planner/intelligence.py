"""Execution Intelligence: the deterministic cost-based planner.

This is the component that answers MESHWEAVE's nine questions:

  1. One GPU or several?
  2. Replicate or partition?
  3. Which nodes form a group?
  4. Which parallelism strategy?
  5. How are layers distributed?
  6. Expected latency?
  7. Expected cost?
  8. What happens if a node disconnects?

It is deterministic on purpose. A learned scheduler is a later concern; a
scheduler nobody can audit is worse than a simple one everybody can.

The planner's most important property is that it can say *no*. Forcing
distributed execution when one GPU is faster and cheaper is a failure mode, not a
conservative default.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from meshweave.bench.economics import ConfigurationCost, compare_configurations
from meshweave.core import (
    Environment,
    ExecutionPlan,
    NodeRecord,
    ParallelStrategy,
    PlanningRequest,
)
from meshweave.errors import PlanningError
from meshweave.models.base import ModelSpec
from meshweave.models.qwen2_5 import Qwen25Adapter, precision_bytes_for
from meshweave.topology.graph import (
    REGIONAL_RTT_MAX_MS,
    TopologyGraph,
    require_datacenter,
)

# A pipeline boundary costs roughly one network round trip per token. Used to
# *estimate* latency before measurement; every experiment overwrites this with
# measured per-hop latency.
DEFAULT_PER_HOP_RTT_MS = 50.0

# Rough decode throughput per GPU for a 7B-class model at bf16, tokens/sec.
# Overwritten by measurement in every real experiment.
DEFAULT_TOKENS_PER_GPU_SEC = 25.0

# Minimum sustained bandwidth (Mbps) before tensor parallelism is considered.
TP_MIN_SUSTAINED_MBPS = 1000.0

# Replicas do not scale perfectly. Requests queue for a router, batching is
# imperfect, and the slowest node in a heterogeneous fleet gates the tail. This
# efficiency factor is a planning assumption; every experiment overwrites it with
# the measured aggregate throughput.
DEFAULT_REPLICATION_EFFICIENCY = 0.85


@dataclass
class PlanningContext:
    """Everything the planner is allowed to look at."""

    nodes: list[NodeRecord]
    graph: TopologyGraph | None
    model_path: str | None = None
    model_spec: ModelSpec | None = None
    per_hop_rtt_ms: float = DEFAULT_PER_HOP_RTT_MS
    tokens_per_gpu_sec: float = DEFAULT_TOKENS_PER_GPU_SEC
    replication_efficiency: float = DEFAULT_REPLICATION_EFFICIENCY
    notes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """The planner's answer, in the shape of the nine questions."""

    use_single_gpu: bool
    use_multiple_gpus: bool
    replicate: bool
    partition: bool
    node_group: list[str]
    strategy: ParallelStrategy
    layer_assignment: dict[str, list[int]]
    expected_ttft_ms: float | None
    expected_itl_ms: float | None
    expected_throughput_tps: float | None
    expected_cost_per_m_output_tokens_usd: float | None
    disconnect_impact: str
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected_alternatives: list[dict[str, Any]] = field(default_factory=list)
    viable: bool = True


class ExecutionPlanner:
    """Plans workloads against measured infrastructure."""

    def __init__(self, adapter: Qwen25Adapter | None = None) -> None:
        self.adapter = adapter or Qwen25Adapter()

    # ------------------------------------------------------------------ API

    def plan(self, request: PlanningRequest, ctx: PlanningContext) -> ExecutionPlan:
        spec = self._spec_for(request, ctx)
        self.adapter._spec = spec  # noqa: SLF001 - the adapter is our memory model
        candidates = self._candidate_nodes(request, ctx)

        if not candidates:
            raise PlanningError(
                f"no healthy node with {request.model} capacity available; "
                f"{len(ctx.nodes)} nodes in registry"
            )

        decisions: list[Decision] = []
        for decision in self._enumerate_decisions(request, ctx, candidates, spec):
            decisions.append(decision)

        if not decisions:
            raise PlanningError(
                "every strategy was rejected; see the alternatives recorded on the request"
            )

        best = min(
            decisions,
            key=lambda d: (
                d.expected_cost_per_m_output_tokens_usd
                if d.expected_cost_per_m_output_tokens_usd is not None
                else float("inf")
            ),
        )
        return self._to_plan(best, request, spec)

    # ------------------------------------------------------------ internals

    def _spec_for(self, request: PlanningRequest, ctx: PlanningContext) -> ModelSpec:
        """Resolve the model's shape facts.

        Order: an explicit spec on the context (a caller that already loaded the
        config), then a local config.json on disk, then the built-in registry.
        Planning never downloads weights.
        """
        if ctx.model_spec is not None:
            return ctx.model_spec

        if ctx.model_path:
            adapter = Qwen25Adapter(ctx.model_path)
            try:
                return adapter.load_spec(ctx.model_path)
            except Exception:  # noqa: BLE001 - fall through to the registry
                pass

        from meshweave.models.registry import resolve

        return resolve(request.model)

    def _candidate_nodes(
        self, request: PlanningRequest, ctx: PlanningContext
    ) -> list[NodeRecord]:
        nodes = [n for n in ctx.nodes if n.gpus and n.status.value == "HEALTHY"]
        if request.preferred_nodes:
            preferred = set(request.preferred_nodes)
            filtered = [n for n in nodes if n.node_id in preferred]
            if filtered:
                nodes = filtered
        return nodes

    def _enumerate_decisions(
        self,
        request: PlanningRequest,
        ctx: PlanningContext,
        candidates: list[NodeRecord],
        spec: ModelSpec,
    ) -> list[Decision]:
        out: list[Decision] = []
        p_bytes = precision_bytes_for(request.quantization)

        for node in candidates:
            out.extend(self._single_gpu_decisions(node, request, ctx, spec, p_bytes))

        if len(candidates) >= 2:
            out.extend(self._replicated_decisions(candidates, ctx, spec, p_bytes))
            out.extend(self._pipeline_decisions(candidates, request, ctx, spec, p_bytes))
            out.extend(self._tensor_decisions(candidates, request, ctx, spec))

        return out

    # -- single GPU --------------------------------------------------------

    def _single_gpu_decisions(
        self,
        node: NodeRecord,
        request: PlanningRequest,
        ctx: PlanningContext,
        spec: ModelSpec,
        p_bytes: float,
    ) -> list[Decision]:
        weight_bytes = self.adapter.weight_bytes_total(p_bytes)
        headroom = max(1, len(node.gpus)) and node.gpus[0].vram_free_bytes
        fits = weight_bytes <= headroom

        throughput = ctx.tokens_per_gpu_sec
        cost_per_m = _cost_per_m_tokens(node, throughput)
        reasons: list[str] = []
        if not fits:
            reasons.append(
                f"weights need {weight_bytes / 1024**3:.1f} GiB, "
                f"{node.node_id} has {headroom / 1024**3:.1f} GiB free"
            )
            return [
                Decision(
                    use_single_gpu=True,
                    use_multiple_gpus=False,
                    replicate=False,
                    partition=False,
                    node_group=[node.node_id],
                    strategy=ParallelStrategy.SINGLE_GPU,
                    layer_assignment={node.node_id: [0, spec.num_hidden_layers]},
                    expected_ttft_ms=None,
                    expected_itl_ms=1000.0 / throughput,
                    expected_throughput_tps=throughput,
                    expected_cost_per_m_output_tokens_usd=None,
                    disconnect_impact="single point of failure: request fails",
                    rationale=reasons,
                )
            ]

        reasons.append(
            f"{request.model} fits on {node.node_id} "
            f"({weight_bytes / 1024**3:.1f} of {headroom / 1024**3:.1f} GiB)"
        )
        return [
            Decision(
                use_single_gpu=True,
                use_multiple_gpus=False,
                replicate=False,
                partition=False,
                node_group=[node.node_id],
                strategy=ParallelStrategy.SINGLE_GPU,
                layer_assignment={node.node_id: [0, spec.num_hidden_layers]},
                expected_ttft_ms=None,
                expected_itl_ms=1000.0 / throughput,
                expected_throughput_tps=throughput,
                expected_cost_per_m_output_tokens_usd=cost_per_m,
                disconnect_impact="single point of failure: request fails",
                rationale=reasons,
            )
        ]

    # -- replicated --------------------------------------------------------

    def _replicated_decisions(
        self,
        candidates: list[NodeRecord],
        ctx: PlanningContext,
        spec: ModelSpec,
        p_bytes: float,
    ) -> list[Decision]:
        """N independent replicas. The network is off the hot path."""
        weight_bytes = self.adapter.weight_bytes_total(p_bytes)
        fitting = [n for n in candidates if n.gpus and n.gpus[0].vram_free_bytes >= weight_bytes]
        if not fitting:
            return []

        total_tps = (
            ctx.tokens_per_gpu_sec * len(fitting) * ctx.replication_efficiency
        )
        total_cost = sum(n.hourly_cost_usd or 0.0 for n in fitting) or None
        cost_per_m = _cost_per_m_tokens_for_cost(total_cost, total_tps)
        return [
            Decision(
                use_single_gpu=False,
                use_multiple_gpus=True,
                replicate=True,
                partition=False,
                node_group=[n.node_id for n in fitting],
                strategy=ParallelStrategy.REPLICATED,
                layer_assignment={
                    n.node_id: [0, spec.num_hidden_layers] for n in fitting
                },
                expected_ttft_ms=None,
                expected_itl_ms=1000.0 / ctx.tokens_per_gpu_sec,
                expected_throughput_tps=total_tps,
                expected_cost_per_m_output_tokens_usd=cost_per_m,
                disconnect_impact=(
                    f"losing one of {len(fitting)} replicas removes "
                    f"{100.0 / len(fitting):.0f}% of throughput; remaining replicas serve"
                ),
                rationale=[
                    f"{len(fitting)} nodes can each hold the full model; "
                    "requests are distributed, no activation crosses the network"
                ],
            )
        ]

    # -- pipeline ----------------------------------------------------------

    def _pipeline_decisions(
        self,
        candidates: list[NodeRecord],
        request: PlanningRequest,
        ctx: PlanningContext,
        spec: ModelSpec,
        p_bytes: float,
    ) -> Iterator[Decision]:
        """Split layers across nodes. Buys capacity; costs one hop per boundary."""
        for size in range(len(candidates), 1, -1):
            group = candidates[:size]
            capacities = [n.gpus[0].vram_free_bytes for n in group if n.gpus]
            if len(capacities) != size:
                continue
            try:
                partition = self.adapter.partition_for_memory(capacities, p_bytes)
            except Exception:  # noqa: BLE001 - ModelLoadError or capacity rejection
                continue

            stages = partition.stage_count
            if stages < 2:
                continue

            # One hop per stage boundary, per token. RTT is measured, not assumed.
            hops = stages - 1
            itl = (1000.0 / ctx.tokens_per_gpu_sec) + hops * ctx.per_hop_rtt_ms
            throughput = 1000.0 / itl

            hourly = _group_hourly_cost(group)
            cost_per_m = _cost_per_m_tokens_for_cost(hourly, throughput)

            ttft = None
            if request.prompt_tokens:
                # Prefill is compute-bound and pipelined across stages.
                ttft = (
                    request.prompt_tokens / ctx.tokens_per_gpu_sec * 1000.0
                    + hops * ctx.per_hop_rtt_ms
                )

            worst = _worst_rtt(group, ctx)
            warnings: list[str] = []
            if worst is not None and worst > REGIONAL_RTT_MAX_MS:
                warnings.append(
                    f"measured RTT between group members is {worst:.0f} ms; "
                    "per-token latency will be dominated by the network"
                )

            yield Decision(
                use_single_gpu=False,
                use_multiple_gpus=True,
                replicate=False,
                partition=True,
                node_group=[n.node_id for n in group],
                strategy=ParallelStrategy.PIPELINE,
                layer_assignment={
                    n.node_id: [partition.layers[s].start, partition.layers[s].end]
                    for s, n in enumerate(group)
                    if s in partition.layers
                },
                expected_ttft_ms=ttft,
                expected_itl_ms=itl,
                expected_throughput_tps=throughput,
                expected_cost_per_m_output_tokens_usd=cost_per_m,
                disconnect_impact=(
                    f"losing any one of the {stages} stages fails the request; "
                    "the whole chain must be rebuilt"
                ),
                rationale=[
                    f"{spec.num_hidden_layers} layers split as "
                    + ", ".join(
                        f"{n.node_id}:{partition.layers[s].start}-{partition.layers[s].end}"
                        for s, n in enumerate(group)
                        if s in partition.layers
                    ),
                    f"{hops} pipeline boundaries x ~{ctx.per_hop_rtt_ms:.0f} ms measured RTT",
                ],
                warnings=warnings,
            )

    # -- tensor ------------------------------------------------------------

    def _tensor_decisions(
        self,
        candidates: list[NodeRecord],
        request: PlanningRequest,
        ctx: PlanningContext,
        spec: ModelSpec,
    ) -> Iterator[Decision]:
        """Tensor parallelism. Refused unless the measured link qualifies."""
        if request.forbid_strategies and ParallelStrategy.TENSOR in request.forbid_strategies:
            return

        graph = ctx.graph
        for size in range(2, min(len(candidates), 5) + 1):
            group = candidates[:size]
            if graph is None:
                yield Decision(
                    use_single_gpu=False,
                    use_multiple_gpus=True,
                    replicate=False,
                    partition=True,
                    node_group=[n.node_id for n in group],
                    strategy=ParallelStrategy.TENSOR,
                    layer_assignment={},
                    expected_ttft_ms=None,
                    expected_itl_ms=None,
                    expected_throughput_tps=None,
                    expected_cost_per_m_output_tokens_usd=None,
                    disconnect_impact="collective fails; request fails",
                    rationale=["no topology measurements available"],
                    rejected_alternatives=[
                        {"reason": "topology matrix missing; tensor parallelism cannot be evaluated"}
                    ],
                    viable=False,
                )
                continue

            reason = require_datacenter(graph, [n.node_id for n in group], TP_MIN_SUSTAINED_MBPS)
            if reason is not None:
                yield Decision(
                    use_single_gpu=False,
                    use_multiple_gpus=True,
                    replicate=False,
                    partition=True,
                    node_group=[n.node_id for n in group],
                    strategy=ParallelStrategy.TENSOR,
                    layer_assignment={},
                    expected_ttft_ms=None,
                    expected_itl_ms=None,
                    expected_throughput_tps=None,
                    expected_cost_per_m_output_tokens_usd=None,
                    disconnect_impact="collective fails; request fails",
                    rationale=[],
                    rejected_alternatives=[{"reason": reason}],
                    viable=False,
                )
                continue

            # Qualifies. Estimate: 2 collectives per layer, each ~1 RTT.
            collectives = 2 * spec.num_hidden_layers
            itl = collectives * ctx.per_hop_rtt_ms
            yield Decision(
                use_single_gpu=False,
                use_multiple_gpus=True,
                replicate=False,
                partition=True,
                node_group=[n.node_id for n in group],
                strategy=ParallelStrategy.TENSOR,
                layer_assignment={n.node_id: [0, spec.num_hidden_layers] for n in group},
                expected_ttft_ms=None,
                expected_itl_ms=itl,
                expected_throughput_tps=1000.0 / itl if itl else None,
                expected_cost_per_m_output_tokens_usd=None,
                disconnect_impact="collective fails; request fails",
                rationale=[
                    f"{collectives} collectives per token across {size} nodes; "
                    "links measured as datacenter-grade"
                ],
            )

    # ----------------------------------------------------------- conversion

    def _to_plan(
        self,
        decision: Decision,
        request: PlanningRequest,
        spec: ModelSpec,
    ) -> ExecutionPlan:
        stage_assignment: dict[int, str] = {}
        if decision.strategy == ParallelStrategy.PIPELINE:
            for idx, node_id in enumerate(decision.node_group):
                stage_assignment[idx] = node_id

        return ExecutionPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:12]}",
            model=request.model,
            strategy=decision.strategy,
            environment=_environment_for(decision),
            stage_assignment=stage_assignment,
            layer_assignment=decision.layer_assignment,
            replica_nodes=(
                decision.node_group
                if decision.strategy == ParallelStrategy.REPLICATED
                else []
            ),
            estimated_ttft_ms=decision.expected_ttft_ms,
            estimated_itl_ms=decision.expected_itl_ms,
            estimated_throughput_tps=decision.expected_throughput_tps,
            estimated_cost_per_m_output_tokens=decision.expected_cost_per_m_output_tokens_usd,
            viable=True,
            assumptions=[
                f"{spec.num_hidden_layers} layers, {spec.hidden_size} hidden, "
                f"quantization {request.quantization}"
            ],
            warnings=list(decision.warnings),
            rationale=list(decision.rationale),
            disconnect_impact=decision.disconnect_impact,
        )

    # ------------------------------------------------------------ comparison

    def compare(
        self,
        request: PlanningRequest,
        ctx: PlanningContext,
    ) -> dict[str, Any]:
        """Cost comparison across candidate deployments.

        Produces the crossover analysis: at what node count does distributing
        stop being cheaper than adding a replica?
        """
        candidates = self._candidate_nodes(request, ctx)
        if not candidates:
            return {"error": "no healthy nodes"}

        self.adapter._spec = self._spec_for(request, ctx)  # noqa: SLF001
        p_bytes = precision_bytes_for(request.quantization)
        weight_bytes = self.adapter.weight_bytes_total(p_bytes)
        fitting = [n for n in candidates if n.gpus and n.gpus[0].vram_free_bytes >= weight_bytes]

        configs: list[ConfigurationCost] = []
        if len(fitting) >= 1:
            n = fitting[0]
            configs.append(
                ConfigurationCost(
                    label="1x single GPU",
                    strategy=ParallelStrategy.SINGLE_GPU,
                    node_count=1,
                    gpu_count=1,
                    hourly_cost_usd=n.hourly_cost_usd,
                    measured_throughput_tps=ctx.tokens_per_gpu_sec,
                )
            )

        for count in (2, 4, 8):
            if len(fitting) < count:
                continue
            group = fitting[:count]
            configs.append(
                ConfigurationCost(
                    label=f"{count}x replicated",
                    strategy=ParallelStrategy.REPLICATED,
                    node_count=count,
                    gpu_count=count,
                    hourly_cost_usd=_group_hourly_cost(group),
                    measured_throughput_tps=(
                        ctx.tokens_per_gpu_sec * count * ctx.replication_efficiency
                    ),
                )
            )

        for count in (2, 4, 8):
            group = candidates[:count]
            if len(group) < count:
                continue
            try:
                partition = self.adapter.partition_for_memory(
                    [n.gpus[0].vram_free_bytes for n in group if n.gpus], p_bytes
                )
            except Exception:  # noqa: BLE001
                continue
            stages = partition.stage_count
            if stages < 2:
                continue
            itl = (1000.0 / ctx.tokens_per_gpu_sec) + (stages - 1) * ctx.per_hop_rtt_ms
            configs.append(
                ConfigurationCost(
                    label=f"{count}x pipeline",
                    strategy=ParallelStrategy.PIPELINE,
                    node_count=count,
                    gpu_count=count,
                    hourly_cost_usd=_group_hourly_cost(group),
                    measured_throughput_tps=1000.0 / itl,
                )
            )

        return compare_configurations(configs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _group_hourly_cost(nodes: list[NodeRecord]) -> float | None:
    costs = [n.hourly_cost_usd for n in nodes]
    if not costs or any(c is None for c in costs):
        return None
    return float(sum(c for c in costs if c is not None))


def _cost_per_m_tokens_for_cost(
    hourly_usd: float | None,
    throughput_tps: float,
) -> float | None:
    if hourly_usd is None or throughput_tps <= 0:
        return None
    tokens_per_hour = throughput_tps * 3600.0
    return hourly_usd / tokens_per_hour * 1e6


def _cost_per_m_tokens(
    node: NodeRecord,
    throughput_tps: float,
) -> float | None:
    return _cost_per_m_tokens_for_cost(node.hourly_cost_usd, throughput_tps)


def _worst_rtt(nodes: list[NodeRecord], ctx: PlanningContext) -> float | None:
    if ctx.graph is None:
        return None
    ids = [n.node_id for n in nodes]
    return ctx.graph.max_rtt_ms(ids)


def _environment_for(decision: Decision) -> Environment:
    if decision.strategy == ParallelStrategy.SINGLE_GPU:
        return Environment.A_DATACENTER
    if decision.strategy == ParallelStrategy.TENSOR:
        return Environment.A_DATACENTER
    if decision.strategy == ParallelStrategy.PIPELINE:
        return Environment.B_REGIONAL
    return Environment.C_INTERNET


def worst_rtt_for_group(nodes: list[NodeRecord], graph: TopologyGraph | None) -> float | None:
    if graph is None:
        return None
    return graph.max_rtt_ms([n.node_id for n in nodes])
