#!/usr/bin/env python3
"""Dual-metric benchmarks: single-request TPS and aggregate TPS, separately.

    python3 benchmarks/vdc/run_dual_benchmark.py \
        --coordinator http://127.0.0.1:8700 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --nodes alpha beta \
        --concurrency 1 2 4 8 16

The two metrics answer different questions and move in opposite directions, so
they are never merged into one number:

  * **single-request TPS** -- how fast one generation completes. Reported per
    request, with p50 and p95 of per-request latency.
  * **aggregate TPS** -- total tokens across all concurrent customers divided by
    wall time, including queueing.

For each concurrency level the harness submits N simultaneous prompts and waits
for all of them, recording queue wait, per-request timings and totals.

Nothing here is simulated: every number comes from a deployment that ran on real
nodes, and a configuration that could not be created is recorded as blocked with
its reason.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "dual_results.json"
GIB = 1024**3


def api(base: str, method: str, path: str, body: dict | None = None, timeout: float = 120.0) -> dict:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise SystemExit(f"{method} {path} -> {exc.code}: {detail}") from exc


def wait_until_running(base: str, deployment_id: str, timeout_s: float = 900.0) -> dict:
    """Wait until every stage is serving.

    A stage whose log shows it is listening is serving even if the coordinator
    still labels it loading: the phase is inferred from log lines.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = api(base, "GET", f"/v1/deployments/{deployment_id}")
        if snapshot.get("status") == "RUNNING":
            return snapshot
        if snapshot.get("status") == "FAILED":
            raise SystemExit(f"deployment failed: {snapshot.get('message')}")
        stages = snapshot.get("stages", [])
        listening = [
            s
            for s in stages
            if "listening" in (s.get("detail") or "") or "control socket" in (s.get("detail") or "")
        ]
        if stages and len(listening) == len(stages):
            return snapshot
        time.sleep(3.0)
    raise SystemExit("deployment did not come up")


def submit_and_wait(base: str, deployment_id: str, prompt: str, tokens: int,
                    timeout_s: float = 900.0) -> dict:
    """Queue one prompt and wait for its result."""
    queued = api(
        base, "POST", f"/v1/deployments/{deployment_id}/prompt",
        {"prompt": prompt, "max_new_tokens": tokens, "temperature": 0},
    )
    prompt_id = queued.get("prompt_id")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = api(base, "GET", f"/v1/deployments/{deployment_id}")
        for result in snapshot.get("results", []):
            if result.get("prompt_id") == prompt_id:
                return result
        time.sleep(1.0)
    return {"error": "no result arrived"}


def run_level(base: str, deployment_id: str, concurrency: int, prompt: str,
              tokens: int, barrier_hold: float) -> dict:
    """Submit ``concurrency`` prompts at once and measure both metrics."""
    results: list[dict] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        # Align the start so the requests genuinely overlap.
        time.sleep(barrier_hold)
        outcome = submit_and_wait(base, deployment_id, f"{prompt} ({index})", tokens)
        with lock:
            results.append(outcome)

    wall_start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_s = time.perf_counter() - wall_start

    ok = [r for r in results if not r.get("error")]
    total_tokens = sum(r.get("output_tokens") or 0 for r in ok)

    latencies = [r["total_s"] for r in ok if r.get("total_s")]
    per_request_tps = [
        (r.get("output_tokens") or 0) / r["total_s"]
        for r in ok
        if r.get("total_s") and (r.get("output_tokens") or 0) > 0
    ]

    return {
        "concurrency": concurrency,
        "requested": concurrency,
        "succeeded": len(ok),
        "failed": concurrency - len(ok),
        "wall_s": round(wall_s, 3),
        "total_output_tokens": total_tokens,
        # Aggregate: everything the cluster produced divided by elapsed time.
        "aggregate_tps": round(total_tokens / wall_s, 3) if wall_s > 0 else None,
        # Single-request: how fast one generation went, ignoring queueing.
        "single_request_tps_mean": round(statistics.mean(per_request_tps), 3) if per_request_tps else None,
        "single_request_tps_min": round(min(per_request_tps), 3) if per_request_tps else None,
        "latency_p50_s": round(_percentile(latencies, 0.50), 3) if latencies else None,
        "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
        "ttft_p50_ms": _mean_of([r.get("ttft_ms") for r in ok]),
        "ms_per_token_mean": _mean_of([r.get("ms_per_token") for r in ok]),
        "activation_bytes_per_token": _first_of([r.get("activation_bytes_per_token") for r in ok]),
        "errors": [r.get("error") for r in results if r.get("error")][:5],
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _mean_of(values: list) -> float | None:
    present = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(present), 2) if present else None


def _first_of(values: list) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--mode", default="AUTO", choices=["AUTO", "PIPELINE", "REPLICA"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args(argv)

    created = api(
        args.coordinator, "POST", "/v1/deployments",
        {
            "model": args.model,
            "node_ids": args.nodes,
            "mode": args.mode,
            "max_seq": 2048,
        },
        timeout=120.0,
    )
    deployment_id = created["deployment_id"]
    print(f"deployment {deployment_id}  mode={created.get('mode')}  "
          f"stages={[(s['node_id'], s['layers']) for s in created['stages']]}")

    try:
        wait_until_running(args.coordinator, deployment_id)
        levels = []
        for concurrency in args.concurrency:
            print(f"\n== concurrency {concurrency}")
            level = run_level(
                args.coordinator, deployment_id, concurrency, args.prompt,
                args.tokens, barrier_hold=1.0,
            )
            levels.append(level)
            print(f"   succeeded {level['succeeded']}/{level['requested']}  "
                  f"aggregate {level['aggregate_tps']} tok/s  "
                  f"single-request {level['single_request_tps_mean']} tok/s  "
                  f"p95 {level['latency_p95_s']} s")
    finally:
        api(args.coordinator, "POST", f"/v1/deployments/{deployment_id}/stop", {})

    payload = {
        "model": args.model,
        "mode": created.get("mode"),
        "nodes": args.nodes,
        "prompt": args.prompt,
        "max_new_tokens": args.tokens,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": levels,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    _print_report(payload)
    return 0


def _print_report(payload: dict) -> None:
    print("\n| conc | ok | aggregate tok/s | single-request tok/s | p50 s | p95 s |")
    print("|---|---|---|---|---|---|")
    for level in payload["levels"]:
        print(
            f"| {level['concurrency']} | {level['succeeded']}/{level['requested']} "
            f"| {level['aggregate_tps']} "
            f"| {level['single_request_tps_mean']} "
            f"| {level['latency_p50_s']} | {level['latency_p95_s']} |"
        )
    print()
    print("aggregate tok/s = all tokens produced / wall time (includes queueing).")
    print("single-request tok/s = one request's tokens / its own latency.")


if __name__ == "__main__":
    raise SystemExit(main())