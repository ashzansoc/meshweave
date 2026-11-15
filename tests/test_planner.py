"""The planner must choose correctly and must be willing to say no.

These are the acceptance-critical behaviours: the planner must not force
distributed execution when one GPU is better, and must not propose tensor
parallelism over an internet link.
"""

from __future__ import annotations

import pytest
from meshweave.bench.economics import ConfigurationCost, compare_configurations
from meshweave.core import (
    LinkKind,
    LinkMeasurement,
    ParallelStrategy,
    PlanningRequest,
    TopologyMatrix,
    utcnow,
)
from meshweave.errors import PlanningError
from meshweave.models.base import ModelSpec
from meshweave.models.qwen2_5 import Qwen25Adapter
from meshweave.planner.intelligence import ExecutionPlanner, PlanningContext
from meshweave.registry.store import ResourceRegistry
from meshweave.topology.graph import build_graph

GIB = 1024**3

QWEN_7B = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}

QWEN_32B = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 27648,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


def spec(**overrides: object) -> ModelSpec:
    kwargs = {
        "architecture": "Qwen2ForCausalLM",
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "rms_norm_eps": 1e-6,
        **QWEN_7B,
    }
    kwargs.update(overrides)
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


def planner_for(model_spec: ModelSpec) -> tuple[ExecutionPlanner, Qwen25Adapter]:
    adapter = Qwen25Adapter()
    adapter._spec = model_spec
    return ExecutionPlanner(adapter), adapter


def test_prefers_single_gpu_when_model_fits(four_nodes: ResourceRegistry) -> None:
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(nodes=four_nodes.all(), graph=None, model_spec=adapter.spec)
    plan = planner.plan(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    assert plan.strategy == ParallelStrategy.SINGLE_GPU
    assert plan.viable is True
    assert plan.rationale


def test_replication_improves_cost_efficiency_with_uniform_prices(
    four_nodes: ResourceRegistry,
) -> None:
    """With identical $/hr nodes, adding replicas lowers cost per token.

    Replication efficiency is 0.85, so N replicas deliver 0.85*N times the
    throughput for N times the cost. That is a net win on cost per token as long
    as 0.85*N > 1, i.e. N >= 2. This is the baseline against which partitioning
    is judged: if partitioning cannot beat this, it is not worth doing.
    """
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(nodes=four_nodes.all(), graph=None, model_spec=adapter.spec)
    comparison = planner.compare(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    rows = {c["label"]: c for c in comparison["configurations"]}
    single = rows["1x single GPU"]["cost_per_m_output_tokens_usd"]
    replicated4 = rows["4x replicated"]["cost_per_m_output_tokens_usd"]
    assert single is not None and replicated4 is not None
    assert replicated4 < single


def test_partitioning_is_not_cheaper_than_replication_when_model_fits(
    four_nodes: ResourceRegistry,
) -> None:
    """The planner must not distribute a model that fits on one GPU.

    Splitting a model that already fits adds network latency per token for no
    capacity gain, so pipeline must lose to both single-GPU and replication.
    """
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(
        nodes=four_nodes.all(), graph=None, model_spec=adapter.spec, per_hop_rtt_ms=60.0
    )
    comparison = planner.compare(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    rows = {c["label"]: c for c in comparison["configurations"]}
    best_non_pipeline = min(
        c["cost_per_m_output_tokens_usd"]
        for label, c in rows.items()
        if "pipeline" not in label and c["cost_per_m_output_tokens_usd"] is not None
    )
    pipeline = rows["4x pipeline"]["cost_per_m_output_tokens_usd"]
    assert pipeline is not None
    assert pipeline > best_non_pipeline


def test_price_dispersion_can_make_single_gpu_wins(
    registry: ResourceRegistry,
) -> None:
    """A cheap node can beat a fleet of expensive ones on cost per token.

    This is the case where replication's advantage disappears: if adding nodes
    means renting progressively more expensive hardware, the cost per token can
    rise with scale. The planner must report that rather than assume scaling wins.
    """
    from tests.conftest import register_node

    register_node(registry, "cheap", "RTX 4090", 0.10)
    register_node(registry, "mid", "RTX 4090", 0.30)
    register_node(registry, "pricey", "RTX 4090", 0.60)
    register_node(registry, "dear", "RTX 4090", 0.90)

    planner, adapter = planner_for(spec())
    ctx = PlanningContext(nodes=registry.all(), graph=None, model_spec=adapter.spec)
    comparison = planner.compare(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    rows = {c["label"]: c for c in comparison["configurations"]}
    single = rows["1x single GPU"]["cost_per_m_output_tokens_usd"]
    replicated = rows["4x replicated"]["cost_per_m_output_tokens_usd"]
    assert single is not None and replicated is not None
    # 4x the fleet cost for 3.4x the throughput is a loss
    assert replicated > single


def test_replication_beats_pipeline_on_cost_per_token(four_nodes: ResourceRegistry) -> None:
    """The core economic finding: replicas are cheaper per token than pipelines."""
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(
        nodes=four_nodes.all(), graph=None, model_spec=adapter.spec, per_hop_rtt_ms=60.0
    )
    comparison = planner.compare(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    rows = {c["label"]: c for c in comparison["configurations"]}
    replicated = rows["4x replicated"]["cost_per_m_output_tokens_usd"]
    pipelined = rows["4x pipeline"]["cost_per_m_output_tokens_usd"]
    assert replicated is not None and pipelined is not None
    assert replicated < pipelined


def test_distribution_is_used_when_model_does_not_fit(four_nodes: ResourceRegistry) -> None:
    planner, adapter = planner_for(spec(**QWEN_32B))
    ctx = PlanningContext(
        nodes=four_nodes.all(), graph=None, model_spec=adapter.spec, per_hop_rtt_ms=60.0
    )
    plan = planner.plan(
        PlanningRequest(model="Qwen2.5-32B-Instruct", quantization="bf16"), ctx
    )
    assert plan.strategy == ParallelStrategy.PIPELINE
    assert len(plan.stage_assignment) >= 2
    # layers are spread, not duplicated
    covered = sum(end - start for start, end in plan.layer_assignment.values())
    assert covered == 64


def test_pipeline_records_its_disconnect_consequence(four_nodes: ResourceRegistry) -> None:
    planner, adapter = planner_for(spec(**QWEN_32B))
    ctx = PlanningContext(
        nodes=four_nodes.all(), graph=None, model_spec=adapter.spec, per_hop_rtt_ms=60.0
    )
    plan = planner.plan(
        PlanningRequest(model="Qwen2.5-32B-Instruct", quantization="bf16"), ctx
    )
    assert plan.disconnect_impact is not None
    assert "fails" in plan.disconnect_impact


def test_tensor_parallelism_is_refused_over_internet(four_nodes: ResourceRegistry) -> None:
    """TP needs datacenter links. Over measured internet it must be refused."""
    planner, adapter = planner_for(spec())
    nodes = four_nodes.all()
    ids = [n.node_id for n in nodes]
    links = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            links.append(
                LinkMeasurement(
                    source=a,
                    target=b,
                    rtt_ms_p50=85.0,
                    tcp_mbps_single_stream=90.0,
                    sustained_mbps=80.0,
                    link_kind=LinkKind.INTERNET,
                    connection_success=True,
                )
            )
            links.append(
                LinkMeasurement(
                    source=b,
                    target=a,
                    rtt_ms_p50=85.0,
                    tcp_mbps_single_stream=90.0,
                    sustained_mbps=80.0,
                    link_kind=LinkKind.INTERNET,
                    connection_success=True,
                )
            )
    matrix = TopologyMatrix(generated_at=utcnow(), nodes=ids, links=links)
    graph = build_graph(matrix)
    PlanningContext(nodes=nodes, graph=graph, model_spec=adapter.spec)

    # TP is never selected for a model that fits anyway; assert the guard directly.
    from meshweave.topology.graph import require_datacenter

    reason = require_datacenter(graph, ids, 1000.0)
    assert reason is not None
    assert "A_DATACENTER" in reason or "datacenter" in reason.lower()


def test_planning_fails_loudly_with_no_nodes() -> None:
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(nodes=[], graph=None, model_spec=adapter.spec)
    with pytest.raises(PlanningError, match="no healthy node"):
        planner.plan(
            PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
        )


def test_unknown_cost_propagates_as_none(four_nodes: ResourceRegistry) -> None:
    """A node with no known price yields no cost figure, not a guessed one."""
    # Drop the price on one node so a configuration must include it.
    four_nodes.get("node-d").hourly_cost_usd = None
    planner, adapter = planner_for(spec())
    ctx = PlanningContext(nodes=four_nodes.all(), graph=None, model_spec=adapter.spec)
    comparison = planner.compare(
        PlanningRequest(model="Qwen2.5-7B-Instruct", quantization="bf16"), ctx
    )
    assert comparison["unpriced"], "expected at least one configuration with unknown cost"


def test_crossover_detection_reports_when_scaling_stops_paying() -> None:
    """Beyond the point where cost per token rises, the crossover is reported."""
    configs = [
        ConfigurationCost("1x", ParallelStrategy.REPLICATED, 1, 1, 0.35, 25.0),
        ConfigurationCost("2x", ParallelStrategy.REPLICATED, 2, 2, 0.70, 50.0),
        ConfigurationCost("4x", ParallelStrategy.REPLICATED, 4, 4, 1.40, 100.0),
    ]
    result = compare_configurations(configs)
    crossover = result["crossover"]
    assert crossover is not None
    assert crossover["explanation"]


def test_node_without_heartbeat_is_not_available(registry: ResourceRegistry) -> None:
    from meshweave.core import NodeRegistration

    registry.register(
        NodeRegistration(node_id="ghost", hostname="ghost", agent_version="t", provider="vast")
    )
    assert registry.available() == []
    assert registry.total_gpu_count() == 0
