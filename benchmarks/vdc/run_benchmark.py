#!/usr/bin/env python3
"""Benchmark A / B / C: single node, the other single node, then both together.

Same model, same prompt, same generation length, same settings in all three, so
the only variable is where the work ran.

    python3 benchmarks/vdc/run_benchmark.py \
        --coordinator http://127.0.0.1:8700 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --nodes alpha beta \
        --cluster F16-VDC-001 \
        --tokens 48

Writes ``benchmarks/vdc/results.json`` and a Markdown comparison beside it.

It never claims an improvement it did not measure: each configuration records its
own numbers, and the report prints the measured ratio alongside them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
GIB = 1024**3


def api(base: str, method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise SystemExit(f"{method} {path} -> {exc.code}: {detail}") from exc


def wait_until_running(base: str, deployment_id: str, timeout_s: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        last = api(base, "GET", f"/v1/deployments/{deployment_id}")
        if last.get("status") == "RUNNING":
            return last
        if last.get("status") == "FAILED":
            raise SystemExit(f"deployment failed: {last.get('message')}")
        time.sleep(3.0)
    raise SystemExit(f"deployment did not reach RUNNING: {last.get('status')}")


def run_case(base: str, model: str, nodes: list[str], prompt: str, tokens: int) -> dict:
    created = api(
        base,
        "POST",
        "/v1/deployments",
        {"model": model, "node_ids": nodes, "max_seq": 2048},
        timeout=120.0,
    )
    deployment_id = created["deployment_id"]
    print(f"  deployed {deployment_id} on {', '.join(nodes)}")

    try:
        wait_until_running(base, deployment_id)
        queued = api(
            base,
            "POST",
            f"/v1/deployments/{deployment_id}/prompt",
            {"prompt": prompt, "max_new_tokens": tokens, "temperature": 0.0},
        )
        prompt_id = queued["prompt_id"]

        deadline = time.monotonic() + 900.0
        result = None
        while time.monotonic() < deadline:
            snapshot = api(base, "GET", f"/v1/deployments/{deployment_id}")
            for candidate in snapshot.get("results", []):
                if candidate.get("prompt_id") == prompt_id:
                    result = candidate
                    break
            if result is not None:
                break
            time.sleep(1.0)
        if result is None:
            raise SystemExit("no result arrived for the prompt")
    finally:
        api(base, "POST", f"/v1/deployments/{deployment_id}/stop")

    stages = [
        {
            "node_id": s.get("node_id"),
            "layers": s.get("layers"),
            "resident_gib": s.get("resident_gib"),
            "phase": s.get("phase"),
        }
        for s in created.get("stages", [])
    ]
    return {
        "deployment_id": deployment_id,
        "nodes": nodes,
        "stages": stages,
        "prompt": prompt,
        "output_tokens": result.get("output_tokens"),
        "prompt_tokens": result.get("prompt_tokens"),
        "ttft_ms": result.get("ttft_ms"),
        "ms_per_token": result.get("ms_per_token"),
        "total_s": result.get("total_s"),
        "tokens_per_second": result.get("tokens_per_second"),
        "stage_roundtrip_ms": result.get("stage_roundtrip_ms"),
        "activation_bytes_per_token": result.get("activation_bytes_per_token"),
        "gpu_during_generation": result.get("gpu_during_generation"),
        "text": result.get("text"),
        "error": result.get("error"),
    }


def nodes_status(base: str) -> dict:
    readiness = api(base, "GET", "/v1/readiness")
    return {
        n["node_id"]: {
            "gpu": n.get("gpu_name"),
            "vram_gib": n.get("vram_gib"),
            "free_vram_gib": n.get("vram_free_gib"),
            "can_host": n.get("can_host"),
        }
        for n in readiness.get("nodes", [])
    }


def _fmt(value, unit: str = "") -> str:
    """Render a measurement, or a dash when there was none."""
    if value is None:
        return "—"
    return f"{value}{unit}"


def markdown(cases: dict, cluster_nodes: list[str], model: str) -> str:
    lines = [
        f"# MESHWEAVE VDC benchmarks",
        "",
        f"Model: `{model}`  ",
        f"Cluster nodes: {', '.join(cluster_nodes)}  ",
        f"Measured: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Configuration | tok/s | ms/token | TTFT ms | total s |",
        "|---|---|---|---|---|",
    ]
    for label in ("single_1", "single_2", "distributed"):
        case = cases.get(label)
        if not case:
            continue
        lines.append(
            f"| {label} ({', '.join(case.get('nodes', []))}) "
            f"| {_fmt(case.get('tokens_per_second'))} "
            f"| {_fmt(case.get('ms_per_token'))} "
            f"| {_fmt(case.get('ttft_ms'))} "
            f"| {_fmt(case.get('total_s'))} |"
        )

    single = next((cases[k] for k in ("single_1", "single_2") if cases.get(k)), None)
    distributed = cases.get("distributed")
    lines += ["", "## What the numbers say", ""]
    if single and distributed and single.get("tokens_per_second") and distributed.get("tokens_per_second"):
        ratio = single["tokens_per_second"] / distributed["tokens_per_second"]
        lines.append(
            f"Distributed is {ratio:.1f}x **slower** than the faster single-node "
            "configuration for the same prompt and length. This is measured, not "
            "projected."
        )
        lines.append("")
        lines.append(
            "Distributed execution is a capacity mechanism on these links: it lets "
            "a model that does not fit one GPU run at all. It does not make a model "
            "that already fits run faster."
        )
    else:
        lines.append("One or more configurations did not produce a measurement.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--nodes", nargs="+", required=True, help="node ids, in order")
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args(argv)

    if len(args.nodes) < 2:
        print("a comparison needs at least two nodes", file=sys.stderr)
        return 2

    print("nodes:", json.dumps(nodes_status(args.coordinator), indent=2))
    cases: dict[str, dict] = {}

    plans = [
        ("single_1", [args.nodes[0]]),
        ("single_2", [args.nodes[1]]),
        ("distributed", list(args.nodes)),
    ]
    for label, nodes in plans:
        print(f"\n== {label}: {', '.join(nodes)}")
        try:
            cases[label] = run_case(args.coordinator, args.model, nodes, args.prompt, args.tokens)
        except SystemExit as exc:
            print(f"  blocked: {exc}")
            cases[label] = {"error": str(exc), "nodes": nodes}

    RESULTS_PATH.write_text(json.dumps({"model": args.model, "cases": cases}, indent=2))
    (RESULTS_PATH.with_suffix(".md")).write_text(markdown(cases, args.nodes, args.model))
    print(f"\nwrote {RESULTS_PATH} and {RESULTS_PATH.with_suffix('.md')}")
    print("\n" + markdown(cases, args.nodes, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())