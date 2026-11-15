"""EXP-NET-005/006: RTT percentiles, jitter, and bandwidth across payload sizes.

Runs over the paths that EXP-NET-002 established. Three measurement classes:

  RTT      many small exchanges, reported as percentiles plus jitter. Local
           timing uses a monotonic clock. No cross-machine one-way latency is
           computed, because the clocks are not synchronised.
  Payload  a sweep of nine sizes from 1 KiB to 100 MiB, measuring throughput and
           transfer time for each.
  Sustained  a longer transfer used to expose throttling after the initial burst.

Design constraints taken from the real hosts:

  * One path (A<->B, B<->C) sustains under 2 Mbps. A 100 MiB transfer there
    would take over a minute, so every payload test has a wall-clock cap and
    reports ``truncated`` when it hits. A partial measurement with an honest
    label beats a timeout with no data.
  * Latency is measured on the *sending* host with ``perf_counter``, so it is a
    real monotonic measurement rather than a subtraction of two wall clocks.

Usage:
    uv run python -m experiments.network.exp_net_005 --pairs all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshweave.remote import HOSTS, RAW_DIR, RemoteError  # noqa: E402
from meshweave.remote.ssh import run_python  # noqa: E402

PAYLOAD_SIZES = [1024, 4096, 8192, 16384, 65536, 262144, 1048576, 10 * 1048576, 100 * 1048576]
RTT_PROBES = 60
MAX_SECONDS_PER_TEST = 45.0
EXPERIMENT_ID = "EXP-NET-005"

# A single client that performs one measurement class and prints JSON.
BENCH_CLIENT = r'''
import socket, time, json, statistics, sys

role = sys.argv[1]
ip = sys.argv[2]
port = int(sys.argv[3])
now = time.perf_counter

def connect():
    return socket.create_connection((ip, port), timeout=15)

def rtt_test():
    samples = []
    failures = 0
    try:
        s = connect()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"P")
        s.recv(1)
        for _ in range(NPROBES):
            t0 = now()
            try:
                s.sendall(b"P")
                s.recv(1)
                samples.append((now() - t0) * 1000.0)
            except OSError:
                failures += 1
        s.close()
    except OSError:
        failures = NPROBES
    if not samples:
        return {"ok": False, "error": "no successful probes"}
    ordered = sorted(samples)
    def pct(p):
        idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
        return ordered[idx]
    jitter = (
        statistics.mean(abs(samples[i] - samples[i - 1]) for i in range(1, len(samples)))
        if len(samples) > 1 else 0.0
    )
    return {
        "ok": True, "probes": len(samples), "failed_probes": failures,
        "min_ms": ordered[0], "p50_ms": pct(50), "p90_ms": pct(90),
        "p95_ms": pct(95), "p99_ms": pct(99), "max_ms": ordered[-1],
        "mean_ms": statistics.mean(samples), "jitter_ms": jitter,
    }

def payload_test(nbytes):
    chunk = b"x" * 65536
    try:
        s = connect()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"S")
        s.recv(1)
        sent = 0
        t0 = now()
        truncated = False
        while sent < nbytes:
            s.sendall(chunk[:min(len(chunk), nbytes - sent)])
            sent += len(chunk[:min(len(chunk), nbytes - sent)])
            if now() - t0 > MAXSECONDS:
                truncated = True
                break
        got = 0
        while got < sent:
            data = s.recv(262144)
            if not data:
                break
            got += len(data)
        elapsed = now() - t0
        s.close()
        if elapsed <= 0 or got == 0:
            return {"ok": False, "error": "no data transferred"}
        return {
            "ok": True, "bytes": sent, "received": got, "truncated": truncated,
            "seconds": elapsed,
            "throughput_mbps": (got * 8 / elapsed / 1e6),
            "throughput_mib_s": (got / elapsed / 1048576),
        }
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

def sustained_test(seconds):
    chunk = b"y" * 65536
    try:
        s = connect()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"L")
        s.recv(1)
        total = 0
        t0 = now()
        while now() - t0 < seconds:
            s.sendall(chunk)
            total += len(chunk)
            got = 0
            while got < len(chunk):
                d = s.recv(262144)
                if not d:
                    raise OSError("closed")
                got += len(d)
        elapsed = now() - t0
        s.close()
        return {"ok": True, "bytes": total, "seconds": elapsed,
                "throughput_mbps": (total * 8 / elapsed / 1e6)}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

if role == "rtt":
    print(json.dumps(rtt_test()))
elif role == "payload":
    print(json.dumps(payload_test(int(sys.argv[4]))))
elif role == "sustained":
    print(json.dumps(sustained_test(int(sys.argv[4]))))
'''


def _script() -> str:
    return BENCH_CLIENT.replace("NPROBES", str(RTT_PROBES)).replace(
        "MAXSECONDS", str(MAX_SECONDS_PER_TEST)
    )


def _args(role: str, ip: str, port: int, extra: str = "") -> str:
    parts = [role, ip, str(port)]
    if extra:
        parts.append(extra)
    return " ".join(parts)


async def run_client(host, role: str, ip: str, port: int, extra: str = "") -> dict:
    """Run one measurement class. Args are passed positionally to the script."""
    try:
        result = await run_python(
            host, _script(), args=_args(role, ip, port, extra), timeout_s=180.0
        )
    except RemoteError as exc:
        return {"ok": False, "error": str(exc)}

    lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {
            "ok": False,
            "error": f"no result (exit {result.exit_code}) {result.stderr.strip()[:140]}",
        }
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-NET-005 RTT and bandwidth")
    parser.add_argument("--pairs", default="all", help="'all' or comma-separated a>b")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--skip-large", action="store_true", help="skip payload sizes above 1 MiB"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


async def _run(args) -> int:
    matrix_path = RAW_DIR / "direct_tcp_matrix.json"
    if not matrix_path.exists():
        print("direct_tcp_matrix.json missing; run exp_net_002 first", file=sys.stderr)
        return 2
    matrix = json.loads(matrix_path.read_text())
    endpoints = {k: tuple(v) for k, v in matrix["endpoints"].items()}

    # Only measure directions that EXP-NET-002 proved carry real data.
    working = {
        (r["source"], r["target"])
        for r in matrix["results"]
        if r.get("ok")
    }

    wanted: list[tuple[str, str]] = []
    ids = list(endpoints)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            for src, dst in ((a, b), (b, a)):
                if args.pairs != "all" and f"{src}>{dst}" not in args.pairs.split(","):
                    continue
                if (src, dst) not in working:
                    continue
                wanted.append((src, dst))

    if not wanted:
        print("no working directed paths to measure", file=sys.stderr)
        return 1

    print(f"\n{'=' * 78}")
    print("EXP-NET-005: RTT, jitter, and bandwidth")
    print(f"{'=' * 78}")
    print(f"measuring {len(wanted)} directed paths\n")

    report: dict = {"experiment": EXPERIMENT_ID, "paths": []}

    for src, dst in wanted:
        ip, port = endpoints[dst]
        by_id = {h.node_id: h for h in HOSTS}
        source_host = by_id[src]

        print(f"--- {src} -> {dst} ({ip}:{port}) ---")
        entry: dict = {"source": src, "target": dst, "endpoint": f"{ip}:{port}"}

        rtt = await run_client(source_host, "rtt", ip, port)
        entry["rtt"] = rtt
        if rtt.get("ok"):
            print(
                f"  rtt  p50={rtt['p50_ms']:.2f}ms p95={rtt['p95_ms']:.2f}ms "
                f"p99={rtt['p99_ms']:.2f}ms jitter={rtt['jitter_ms']:.2f}ms "
                f"loss={rtt['failed_probes']}/{rtt['probes']}"
            )
        else:
            print(f"  rtt  FAILED: {rtt.get('error')}")

        sizes = [s for s in PAYLOAD_SIZES if not (args.skip_large and s > 1048576)]
        payloads = []
        for size in sizes:
            res = await run_client(source_host, "payload", ip, port, str(size))
            res["requested_bytes"] = size
            payloads.append(res)
            if res.get("ok"):
                label = f"{size/1024:.0f} KiB" if size < 1048576 else f"{size/1048576:.0f} MiB"
                trunc = " (truncated)" if res.get("truncated") else ""
                print(
                    f"  {label:>9} {res['throughput_mbps']:8.2f} Mbps  "
                    f"{res['seconds']:6.2f}s{trunc}"
                )
            else:
                label = f"{size/1024:.0f} KiB" if size < 1048576 else f"{size/1048576:.0f} MiB"
                print(f"  {label:>9} FAILED: {res.get('error')}")
        entry["payloads"] = payloads

        sustained = await run_client(source_host, "sustained", ip, port, "15")
        entry["sustained_15s"] = sustained
        if sustained.get("ok"):
            print(f"  sustained 15s: {sustained['throughput_mbps']:.2f} Mbps")
        else:
            print(f"  sustained 15s: FAILED {sustained.get('error')}")

        report["paths"].append(entry)
        print()

    out = args.out or (RAW_DIR / "rtt_bandwidth.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    print(f"{'=' * 78}")
    print("summary")
    print(f"{'=' * 78}")
    print(f"  {'path':<40}{'rtt p50':>10}{'p99':>10}{'jitter':>10}{'1MiB':>10}")
    for entry in report["paths"]:
        path = f"{entry['source']} -> {entry['target']}"
        rtt = entry.get("rtt") or {}
        one_mib = next(
            (p for p in entry.get("payloads", []) if p.get("requested_bytes") == 1048576),
            None,
        )
        rtt_p50 = f"{rtt['p50_ms']:.1f}ms" if rtt.get("ok") else "n/a"
        rtt_p99 = f"{rtt['p99_ms']:.1f}ms" if rtt.get("ok") else "n/a"
        jitter = f"{rtt['jitter_ms']:.1f}ms" if rtt.get("ok") else "n/a"
        mib = f"{one_mib['throughput_mbps']:.1f}M" if one_mib and one_mib.get("ok") else "n/a"
        print(f"  {path:<40}{rtt_p50:>10}{rtt_p99:>10}{jitter:>10}{mib:>10}")


if __name__ == "__main__":
    raise SystemExit(main())
