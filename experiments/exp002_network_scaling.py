"""EXP-002/003/004/005: network measurement and scaling experiments.

These share one harness because they differ only in the node set and the metric
highlighted. Splitting them into four near-identical files would make the
comparison harder to audit, not easier.

EXP-002  two nodes, same network      -> what does a fast link buy?
EXP-003  two independent Vast hosts   -> what does the internet cost?
EXP-004  four independent GPUs        -> does scaling hold?
EXP-005  eight independent GPUs       -> where does it break?

Each measures the N x N link matrix first. No model is loaded until the links are
characterised, because the planner needs them and because "the network was slow"
is only a finding if you measured it.
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

EXPERIMENTS = {
    "exp002": {
        "id": "EXP-002",
        "title": "Two Nodes, Same Network",
        "hypothesis": (
            "Two GPUs on a fast shared link can cooperate with acceptable overhead, "
            "establishing the upper bound for what distribution can achieve."
        ),
        "nodes": 2,
        "expect": "low RTT, high sustained bandwidth",
    },
    "exp003": {
        "id": "EXP-003",
        "title": "Two Independent Vast Hosts",
        "hypothesis": (
            "Two GPUs on separate hosts across the public internet can cooperate, and "
            "the measured overhead quantifies what geographic distribution costs."
        ),
        "nodes": 2,
        "expect": "high RTT, variable bandwidth, possible packet loss",
    },
    "exp004": {
        "id": "EXP-004",
        "title": "Four Independent GPUs",
        "hypothesis": (
            "Scaling from two to four independently hosted GPUs continues to add "
            "usable capacity, and the scaling efficiency is measurable."
        ),
        "nodes": 4,
        "expect": "scaling efficiency below linear as RTT accumulates",
    },
    "exp005": {
        "id": "EXP-005",
        "title": "Eight Independent GPUs",
        "hypothesis": (
            "Eight independently hosted GPUs can execute a model that fits on none of "
            "them, and the cost of the required coordination is quantified."
        ),
        "nodes": 8,
        "expect": "largest capacity, worst per-token latency",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MESHWEAVE network scaling experiments")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--coordinator", default="http://127.0.0.1:8700")
    parser.add_argument("--nodes", default="", help="comma-separated host:port probe endpoints")
    parser.add_argument("--ids", default="", help="comma-separated node ids matching --nodes")
    parser.add_argument("--payload-mb", type=int, default=32)
    parser.add_argument("--rtt-samples", type=int, default=15)
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = EXPERIMENTS[args.experiment]

    exp = Experiment(
        experiment_id=spec["id"],
        title=spec["title"],
        hypothesis=spec["hypothesis"],
        command=f"uv run python -m experiments.exp002_network_scaling {args.experiment}",
    )

    try:
        asyncio.run(_run(exp, args, spec))
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


async def _run(exp: Experiment, args: argparse.Namespace, spec: dict) -> None:
    from meshweave.bench.network import (
        ProbeConfig,
        build_matrix,
        endpoint_from_url,
    )
    from meshweave.topology.graph import build_graph

    targets = [t.strip() for t in args.nodes.split(",") if t.strip()]
    if len(targets) < 2:
        raise ExperimentBlockedError(
            f"{spec['id']} needs at least two probe endpoints to measure links; "
            "none were supplied",
            "meshweave bench network --nodes host1:port,host2:port --ids node-a,node-b",
        )

    ids = (
        [i.strip() for i in args.ids.split(",")]
        if args.ids
        else [t.split(":")[0] for t in targets]
    )
    if len(ids) != len(targets):
        raise ExperimentBlockedError(
            f"--ids ({len(ids)}) must match --nodes ({len(targets)}) in count"
        )

    exp.require(
        len(ids) >= spec["nodes"],
        f"{spec['id']} expects {spec['nodes']} nodes; {len(ids)} supplied",
    )

    cfg = ProbeConfig(
        reference_payload_bytes=args.payload_mb * 1024 * 1024,
        rtt_samples=args.rtt_samples,
        probes=["rtt", "tcp", "stability"],
        run_sustained=True,
    )
    endpoints = {
        node_id: endpoint_from_url(
            target if "://" in target else f"http://{target}", node_id
        )
        for node_id, target in zip(ids, targets, strict=True)
    }

    exp.measure("nodes_measured", float(len(ids)), "count")
    exp.measure("directed_links", float(len(ids) * (len(ids) - 1)), "count")
    exp.measure(
        "unordered_pairs",
        float(len(ids) * (len(ids) - 1) // 2),
        "count",
        note="each pair is probed in both directions",
    )

    matrix = await build_matrix(ids, endpoints, cfg)
    graph = build_graph(matrix)

    rtts = [link.rtt_ms_p50 for link in matrix.links if link.rtt_ms_p50 is not None]
    bandwidths = [
        link.sustained_mbps or link.tcp_mbps_single_stream
        for link in matrix.links
        if (link.sustained_mbps or link.tcp_mbps_single_stream) is not None
    ]
    losses = [link.packet_loss_pct for link in matrix.links if link.packet_loss_pct is not None]

    if rtts:
        exp.measure("rtt_p50_ms", _median(rtts), "ms", samples=len(rtts))
        exp.measure("rtt_min_ms", min(rtts), "ms")
        exp.measure("rtt_max_ms", max(rtts), "ms")
    else:
        exp.measure(
            "rtt_p50_ms", None, "ms", note="no link returned an RTT; all probes failed"
        )

    if bandwidths:
        exp.measure("bandwidth_mbps", _median(bandwidths), "Mbps", samples=len(bandwidths))
        exp.measure("bandwidth_min_mbps", min(bandwidths), "Mbps")
    if losses:
        exp.measure("packet_loss_pct", _mean(losses), "%")

    environments = {k: v.value for k, v in graph.environment_by_node.items()}
    exp.measure(
        "environments_classified",
        float(len(environments)),
        "nodes",
        note=json.dumps(environments),
    )

    exp.nodes = [{"node_id": n, "endpoint": endpoints[n].base_url} for n in ids]

    failed = [link for link in matrix.links if link.error]
    if failed:
        exp.findings.append(
            f"{len(failed)} of {len(matrix.links)} directed links reported errors"
        )

    exp.findings.append(
        f"measured {len(ids)} nodes: median RTT {_median(rtts) if rtts else float('nan'):.1f} ms, "
        f"median sustained bandwidth {_median(bandwidths) if bandwidths else float('nan'):.0f} Mbps"
    )
    exp.findings.append(f"link environments: {environments or 'none classified'}")

    # The planner's verdict on this topology, recorded as evidence.
    exp.measure(
        "expectation",
        None,
        "text",
        note=spec["expect"],
    )

    topology_path = Path("reports") / "raw" / f"{spec['id'].lower()}-topology.json"
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(json.dumps(json.loads(matrix.model_dump_json()), indent=2))
    exp.evidence.append(str(topology_path))

    from meshweave.bench.report import EconomicReport

    exp.economics = EconomicReport(
        cost_known=False,
        unknown_components=["node hourly costs (set per node via /v1/nodes/{id}/cost)"],
    )
    emit(exp.finish(out=args.out))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
