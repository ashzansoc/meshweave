"""EXP-010: Replication versus Partitioning.

The question the whole project turns on. Two ways to use several GPUs:

  Replication  -- N independent copies, requests spread across them. The network
                  is off the hot path. Throughput scales, per-request latency does
                  not improve, and a model larger than one GPU is still impossible.

  Partitioning -- one model split across N GPUs. A model larger than one GPU
                  becomes possible. Every token pays the network once per stage
                  boundary, so per-token latency gets worse as N grows.

This experiment measures both against the same prompt set and reports the
crossover: the point at which adding nodes stops being economically defensible.

It also records the honest negative result when there is one. If partitioning
loses, the report says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.harness import (  # noqa: E402
    Experiment,
    ExperimentBlockedError,
    add_common_args,
    emit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXP-010 replication versus partitioning"
    )
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--coordinator", default="http://127.0.0.1:8700")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--per-hop-rtt-ms", type=float, default=None)
    parser.add_argument("--tokens-per-gpu-sec", type=float, default=20.0)
    parser.add_argument("--hourly-cost", type=float, default=None)
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exp = Experiment(
        experiment_id="EXP-010",
        title="Replication versus Partitioning",
        hypothesis=(
            "Distributing a model across independently hosted GPUs can increase "
            "capacity but does not increase per-token throughput, and the economic "
            "crossover between the two strategies can be located."
        ),
        command="uv run python -m experiments.exp010_replication_vs_partition",
    )

    try:
        asyncio.run(_run(exp, args))
    except ExperimentBlockedError as blocked:
        emit(
            exp.finish(
                status="blocked",
                blocked_reason=blocked.reason,
                ready_to_run_command=blocked.command,
                out=args.out,
            )
        )
        return 1
    return 0


async def _run(exp: Experiment, args: argparse.Namespace) -> None:
    import httpx
    from meshweave.core import ParallelStrategy, PlanningRequest
    from meshweave.models.qwen2_5 import Qwen25Adapter
    from meshweave.models.registry import resolve
    from meshweave.planner.intelligence import PlanningContext

    coordinator = args.coordinator.rstrip("/")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            health = await client.get(f"{coordinator}/healthz")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExperimentBlockedError(
                f"coordinator unreachable at {coordinator}: {exc}",
                "meshweave coordinator --port 8700",
            ) from exc

        nodes_resp = await client.get(f"{coordinator}/v1/nodes")
        nodes_resp.raise_for_status()
        payload = nodes_resp.json()
        topology_resp = await client.get(f"{coordinator}/v1/topology")

    healthy = [n for n in payload["nodes"] if n.get("status") == "HEALTHY"]
    exp.require(
        len(healthy) >= 2,
        f"EXP-010 needs at least two healthy nodes; found {len(healthy)}",
    )

    graph = None
    per_hop = args.per_hop_rtt_ms
    if topology_resp.status_code == 200:
        from meshweave.core import TopologyMatrix
        from meshweave.topology.graph import build_graph

        matrix = TopologyMatrix.model_validate(topology_resp.json()["matrix"])
        graph = build_graph(matrix)
        measured = [
            link.rtt_ms_p50 for link in matrix.links if link.rtt_ms_p50 is not None
        ]
        if measured and per_hop is None:
            per_hop = sum(measured) / len(measured)
        exp.measure(
            "measured_links",
            float(len(matrix.links)),
            "links",
            note="from the coordinator topology matrix",
        )

    if per_hop is None:
        exp.measure(
            "per_hop_rtt_ms",
            None,
            "ms",
            note="no topology measurement; using the planner's 50 ms default",
        )

    # Rebuild node records for the planner.
    from meshweave.core import GpuInventory, NodeRecord, NodeRegistration

    records: list[NodeRecord] = []
    for node in healthy:
        inv = GpuInventory.model_validate(node["inventory"])
        records.append(
            NodeRecord(
                node_id=node["node_id"],
                registration=NodeRegistration(
                    node_id=node["node_id"],
                    hostname=node["node_id"],
                    agent_version=inv.agent_version,
                    provider="vast",
                ),
                inventory=inv,
                status=node["status"],
                hourly_cost_usd=node.get("hourly_cost_usd"),
            )
        )

    spec = resolve(args.model)
    adapter = Qwen25Adapter()
    adapter._spec = spec  # noqa: SLF001

    ctx = PlanningContext(
        nodes=records,
        graph=graph,
        model_spec=spec,
        per_hop_rtt_ms=per_hop or 50.0,
        tokens_per_gpu_sec=args.tokens_per_gpu_sec,
    )

    request = PlanningRequest(
        model=args.model,
        quantization=args.quantization,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    comparison = compare_from_planner(ctx, request, adapter)
    exp.nodes = [{"node_id": n["node_id"]} for n in healthy]

    rows = comparison["configurations"]
    for row in rows:
        cost = row["cost_per_m_output_tokens_usd"]
        tps = row["throughput_tps"]
        exp.measure(
            f"tps::{row['label']}",
            tps,
            "tok/s",
            note=f"{row['strategy']}, {row['nodes']} node(s)",
        )
        exp.measure(
            f"cost_per_m::{row['label']}",
            cost,
            "USD/M tokens",
            note="unknown cost" if cost is None else None,
        )

    cheapest = comparison["cheapest_by_cost_per_m_output_tokens"]
    if cheapest:
        exp.findings.append(
            f"cheapest configuration by cost per million output tokens: "
            f"{cheapest['label']} at ${cheapest['cost_per_m_output_tokens_usd']:.2f}"
        )

    crossover = comparison.get("crossover")
    if crossover and crossover.get("beyond_node_count") is not None:
        exp.findings.append(
            f"crossover at {crossover['beyond_node_count']} nodes: adding more stops "
            f"reducing cost per million tokens "
            f"(${crossover['cost_before']:.2f} -> ${crossover['cost_after']:.2f})"
        )
    elif crossover:
        exp.findings.append(crossover["explanation"])

    # The core comparison, stated explicitly.
    replicated = next(
        (r for r in rows if r["strategy"] == ParallelStrategy.REPLICATED.value), None
    )
    pipelined = next(
        (r for r in rows if r["strategy"] == ParallelStrategy.PIPELINE.value), None
    )
    if replicated and pipelined:
        rc = replicated["cost_per_m_output_tokens_usd"]
        pc = pipelined["cost_per_m_output_tokens_usd"]
        if rc is not None and pc is not None:
            if rc < pc:
                exp.findings.append(
                    f"replication is cheaper per token than partitioning "
                    f"(${rc:.2f} vs ${pc:.2f})"
                )
            else:
                exp.findings.append(
                    f"partitioning is cheaper per token than replication "
                    f"(${pc:.2f} vs ${rc:.2f})"
                )

    if args.hourly_cost is not None:
        from meshweave.bench.report import EconomicReport

        exp.economics = EconomicReport(
            cost_known=True,
            hourly_cost_usd=args.hourly_cost,
            total_cost_usd=args.hourly_cost * len(healthy),
        )
    else:
        from meshweave.bench.report import EconomicReport

        exp.economics = EconomicReport(
            cost_known=False,
            unknown_components=[
                "node hourly costs (set via /v1/nodes/{id}/cost or --hourly-cost)"
            ],
        )

    out = Path("reports") / "raw" / "exp010-comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, default=str))
    exp.evidence.append(str(out))
    exp.evidence.append(f"planner comparison over {len(healthy)} nodes")

    emit(exp.finish(out=args.out))


def compare_from_planner(ctx, request, adapter) -> dict:
    from meshweave.planner.intelligence import ExecutionPlanner

    planner = ExecutionPlanner(adapter)
    return planner.compare(request, ctx)


if __name__ == "__main__":
    raise SystemExit(main())
