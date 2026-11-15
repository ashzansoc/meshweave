"""MESHWEAVE command line interface.

    meshweave agent           run an agent on this GPU host
    meshweave coordinator     run the control plane
    meshweave bench           network benchmark between nodes
    meshweave plan            ask the planner what it would do
    meshweave certs           generate a local CA and node certificates
    meshweave nodes           list what the coordinator knows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from meshweave import __version__
from meshweave.config import AgentSettings, CoordinatorSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meshweave",
        description="MESHWEAVE — Many Machines. One Intelligence.",
    )
    parser.add_argument("--version", action="version", version=f"meshweave {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- agent ---------------------------------------------------------
    agent = sub.add_parser("agent", help="run an agent on this GPU host")
    agent.add_argument("--node-id", required=True)
    agent.add_argument("--coordinator", default="https://127.0.0.1:8700")
    agent.add_argument("--port", type=int, default=8443)
    agent.add_argument("--tls-cert", type=Path, default=None)
    agent.add_argument("--tls-key", type=Path, default=None)
    agent.add_argument("--tls-ca", type=Path, default=None)
    agent.add_argument(
        "--simulation",
        action="store_true",
        help="use synthetic GPUs. FOR CONTROL-PLANE TESTING ONLY.",
    )

    # -- coordinator ---------------------------------------------------
    coord = sub.add_parser("coordinator", help="run the control plane")
    coord.add_argument("--host", default="127.0.0.1")
    coord.add_argument("--port", type=int, default=8700)
    coord.add_argument("--tls-cert", type=Path, default=None)
    coord.add_argument("--tls-key", type=Path, default=None)

    # -- bench ---------------------------------------------------------
    bench = sub.add_parser("bench", help="network benchmark")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)

    net = bench_sub.add_parser("network", help="measure the N x N link matrix")
    net.add_argument("--nodes", required=True, help="comma-separated host:port list")
    net.add_argument("--ids", default=None, help="comma-separated node ids (same order)")
    net.add_argument("--payload-mb", type=int, default=64)
    net.add_argument("--sustained-mb", type=int, default=256)
    net.add_argument("--rtt-samples", type=int, default=20)
    net.add_argument("--skip-udp", action="store_true")
    net.add_argument("--out", default="reports/topology.json")

    # -- plan ----------------------------------------------------------
    plan = sub.add_parser("plan", help="ask the planner what it would do")
    plan.add_argument("--model", default="Qwen2.5-7B-Instruct")
    plan.add_argument("--quantization", default="bf16")
    plan.add_argument("--prompt-tokens", type=int, default=128)
    plan.add_argument("--max-new-tokens", type=int, default=256)
    plan.add_argument("--coordinator", default="http://127.0.0.1:8700")
    plan.add_argument("--compare", action="store_true", help="compare configurations")

    # -- certs ---------------------------------------------------------
    certs = sub.add_parser("certs", help="generate a local CA and node certificate")
    certs.add_argument("--out", type=Path, default=Path("./secrets"))
    certs.add_argument("--node-id", required=True)
    certs.add_argument("--days", type=int, default=365)

    # -- nodes ---------------------------------------------------------
    nodes = sub.add_parser("nodes", help="list nodes known to the coordinator")
    nodes.add_argument("--coordinator", default="http://127.0.0.1:8700")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "agent":
        return _cmd_agent(args)
    if args.command == "coordinator":
        return _cmd_coordinator(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "certs":
        return _cmd_certs(args)
    if args.command == "nodes":
        return _cmd_nodes(args)
    return 2


def _cmd_agent(args: argparse.Namespace) -> int:
    from meshweave.agent.runtime import AgentRuntime

    settings = AgentSettings(
        node_id=args.node_id,
        coordinator_url=args.coordinator,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        tls_ca=args.tls_ca,
        simulation=args.simulation,
    )
    if args.simulation:
        print(
            "WARNING: --simulation uses synthetic GPUs. "
            "Results are not evidence of distributed execution.",
            file=sys.stderr,
        )
    runtime = AgentRuntime(settings)
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        return 130
    return 0


def _cmd_coordinator(args: argparse.Namespace) -> int:
  # fmt: skip
    import uvicorn

    from meshweave.coordinator.app import create_app

    settings = CoordinatorSettings(host=args.host, port=args.port)
    app = create_app(settings)
    kwargs: dict[str, Any] = {"host": args.host, "port": args.port, "log_level": "info"}
    if args.tls_cert and args.tls_key:
        kwargs["ssl_certfile"] = str(args.tls_cert)
        kwargs["ssl_keyfile"] = str(args.tls_key)
    uvicorn.run(app, **kwargs)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.bench_command == "network":
        return _bench_network(args)
    return 2


def _bench_network(args: argparse.Namespace) -> int:
    from meshweave.bench.network import ProbeConfig, build_matrix, endpoint_from_url

    targets = [t.strip() for t in args.nodes.split(",") if t.strip()]
    ids = (
        [i.strip() for i in args.ids.split(",")]
        if args.ids
        else [t.split(":")[0] for t in targets]
    )
    if len(ids) != len(targets):
        print("error: --ids must match --nodes in count", file=sys.stderr)
        return 2

    probes = ["rtt", "tcp", "stability"] + ([] if args.skip_udp else ["udp"])
    cfg = ProbeConfig(
        reference_payload_bytes=args.payload_mb * 1024 * 1024,
        sustained_payload_bytes=args.sustained_mb * 1024 * 1024,
        rtt_samples=args.rtt_samples,
        probes=probes,
    )
    endpoints = {
        node_id: endpoint_from_url(target if "://" in target else f"http://{target}", node_id)
        for node_id, target in zip(ids, targets, strict=True)
    }

    print(f"probing {len(ids)} nodes -> {len(ids) * (len(ids) - 1)} directed links")
    matrix = asyncio.run(build_matrix(ids, endpoints, cfg))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json.loads(matrix.model_dump_json()), indent=2))
    print(f"wrote {out}")
    for link in matrix.links:
        rtt = f"{link.rtt_ms_p50:.2f} ms" if link.rtt_ms_p50 else "n/a"
        bw = (
            f"{link.sustained_mbps:.0f} Mbps"
            if link.sustained_mbps
            else (f"{link.tcp_mbps_single_stream:.0f} Mbps" if link.tcp_mbps_single_stream else "n/a")
        )
        loss = f"{link.packet_loss_pct:.1f}%" if link.packet_loss_pct is not None else "n/a"
        print(f"  {link.source} -> {link.target}: rtt={rtt} bw={bw} loss={loss}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    import httpx

    body = {
        "model": args.model,
        "quantization": args.quantization,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    path = "/v1/plan/compare" if args.compare else "/v1/plan"
    try:
        resp = httpx.post(f"{args.coordinator}{path}", json=body, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0


def _cmd_nodes(args: argparse.Namespace) -> int:
    import httpx

    try:
        resp = httpx.get(f"{args.coordinator}/v1/nodes", timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0


def _cmd_certs(args: argparse.Namespace) -> int:
    from meshweave.security.certs import issue_node_certificate

    out = issue_node_certificate(
        out_dir=args.out, node_id=args.node_id, days=args.days
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
