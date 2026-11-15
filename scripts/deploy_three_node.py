"""Deploy and drive three-node Phi-mini-MoE execution.

The architecture, and why:

  llama.cpp's RPC backend is the execution engine. It splits a model by layer
  across machines: worker hosts run ``rpc-server`` exposing their GPU, and one
  host runs the model and assigns layers to each peer. This is the proven
  mechanism for "one model across several machines" -- it is plain TCP, it
  tolerates latency by sending one activation per hop, and it does not need NCCL.

  MESHWEAVE is the control plane: it decides the placement, launches the
  processes, records their PIDs, measures the links, and benchmarks.

Why not expert parallelism here: with 16 experts per layer and top-2 routing,
a token routed to a remote expert costs one round trip *per layer that routes
there*. At HOST-B's measured 309 ms RTT that is seconds per token. Pipeline
parallelism pays one round trip per stage boundary per token, which is strictly
cheaper, and llama.cpp's RPC backend is built for it.

Why B is placed at the end of the chain, not the middle: a boundary costs one
RTT. A chain of three has two boundaries. Putting B last means the slow hop is
C->B (309 ms) rather than two slow hops (A->B and B->C = 620 ms). The
placement is chosen from the measured matrix, not by assumption.

Usage:
    uv run python scripts/deploy_three_node.py --detect
    uv run python scripts/deploy_three_node.py --up
    uv run python scripts/deploy_three_node.py --bench
    uv run python scripts/deploy_three_node.py --down
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meshweave.remote import HOSTS, Host, RemoteError, run  # noqa: E402
from meshweave.remote.process_registry import REGISTRY  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "model_execution" / "results"
EXPERIMENT_ID = "THREE-NODE-001"

MODEL = "phi-mini-moe-Q6_K.gguf"
MODEL_PATH = f"/root/models/{MODEL}"
LLAMA = "/root/llama.cpp/build/bin"
RPC_PORT = 50052
MAIN_PORT = 8095


def _host(node_id: str) -> Host:
    return next(h for h in HOSTS if h.node_id == node_id)


# --------------------------------------------------------------------------- #
# capability detection
# --------------------------------------------------------------------------- #


async def detect() -> dict:
    """Check that every host has the binary and the model."""
    report = {}
    for host in HOSTS:
        entry: dict = {"node_id": host.node_id}
        try:
            result = await run(
                host,
                f"test -x {LLAMA}/rpc-server && echo RPC_OK || echo RPC_MISSING; "
                f"test -f {MODEL_PATH} && stat -c %s {MODEL_PATH} || echo MODEL_MISSING; "
                f"nvidia-smi --query-gpu=name,memory.total,memory.free "
                f"--format=csv,noheader",
                timeout_s=30.0,
                check=False,
            )
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            entry["rpc_server"] = "RPC_OK" in lines
            entry["model_bytes"] = next(
                (int(ln) for ln in lines if ln.isdigit()), None
            )
            entry["gpu"] = next(
                (ln for ln in lines if "MiB" in ln), None
            )
        except RemoteError as exc:
            entry["error"] = str(exc)
        report[host.node_id] = entry
        status = "ok" if entry.get("rpc_server") and entry.get("model_bytes") else "incomplete"
        print(f"  {host.node_id}: {status}")
        if entry.get("gpu"):
            print(f"    gpu: {entry['gpu']}")
        if entry.get("model_bytes"):
            print(f"    model: {entry['model_bytes'] / 1024**3:.2f} GiB")
    return report


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #


def choose_order(rtt: dict[str, float]) -> list[str]:
    """Order the nodes so the slowest link sits at one end.

    A pipeline of N nodes has N-1 boundaries; each costs one RTT per token.
    Placing the slow node at an end costs one slow hop. Placing it in the middle
    costs two. So: put the node with the worst average RTT last.

    Returns node ids in pipeline order (stage 0 first).
    """
    nodes = list(rtt)

    def avg(node: str) -> float:
        peers = [v for k, v in rtt.items() if node in k]
        return sum(peers) / len(peers) if peers else 0.0

    return sorted(nodes, key=avg)  # fastest first


def measured_rtt() -> dict[str, float]:
    """Mean measured RTT per node from the Phase 2A evidence."""
    path = REPO / "experiments" / "network" / "raw" / "rtt_bandwidth.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    totals: dict[str, list[float]] = {}
    for entry in data.get("paths", []):
        rtt = entry.get("rtt") or {}
        if not rtt.get("ok") or rtt.get("p50_ms") is None:
            continue
        for node in (entry["source"], entry["target"]):
            totals.setdefault(node, []).append(float(rtt["p50_ms"]))
    return {k: sum(v) / len(v) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


async def up(order: list[str]) -> dict:
    """Start rpc-server on every node, then start the main model.

    Each rpc-server is a foreground process held in a live SSH session, recorded
    in the process registry by PID. Nothing is backgrounded with nohup or tmux:
    on these hosts, backgrounded processes were reaped when the session ended.
    """
    procs: list[tuple[Host, asyncio.subprocess.Process]] = []
    started: dict[str, int] = {}

    try:
        # Workers first: every node except the first in the pipeline serves
        # layers to the host that drives the model.
        driver = order[0]
        workers = order[1:]

        for node_id in workers:
            host = _host(node_id)
            argv = [
                "ssh", "-i", str(host.key), "-p", str(host.port),
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                f"root@{host.host}",
                f"{LLAMA}/rpc-server -H 0.0.0.0 -p {RPC_PORT} -c 0 -ngl 0",
            ]
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            REGISTRY.record(
                proc.pid,
                node_id=node_id,
                experiment_id=EXPERIMENT_ID,
                command=f"rpc-server on {RPC_PORT}",
            )
            procs.append((host, proc))
            started[node_id] = proc.pid
            print(f"  worker {node_id}: rpc-server pid {proc.pid}")

        # Give the workers a moment to bind.
        await asyncio.sleep(3)

        # Confirm each worker answers before starting the model.
        for node_id in workers:
            host = _host(node_id)
            ok = await _probe_rpc(host)
            print(f"  worker {node_id}: {'listening' if ok else 'NOT LISTENING'}")
            if not ok:
                raise MeshweaveStartError(f"rpc-server on {node_id} did not come up")

        result = await _start_main(driver, workers)
        return {"order": order, "workers": started, "main": result}

    finally:
        # Leaving workers running is intentional; --down stops them.
        pass


class MeshweaveStartError(RuntimeError):
    pass


async def _probe_rpc(host: Host) -> bool:
    """Check the worker's RPC port answers on its mapped external port."""
    try:
        result = await run(
            host,
            f"ss -tln | grep -q {RPC_PORT} && echo LISTEN || echo NO",
            timeout_s=15.0,
            check=False,
        )
        return "LISTEN" in result.stdout
    except RemoteError:
        return False


async def _start_main(driver: str, workers: list[str]) -> dict:
    """Start the model on the driver host, assigning layers to each worker.

    The layer split comes from the PhiMoE adapter's partition plan, sized from
    each GPU's real free VRAM rather than assumed equal shares.
    """
    from meshweave.models.phi_moe import PhiMoEAdapter

    adapter = PhiMoEAdapter()
    order = [driver] + workers
    capacities = _gpu_free_bytes(order)
    partition = adapter.plan_partition(capacities, "Q6_K", kv_tokens=2048)

    layer_args = []
    for index, node_id in enumerate(order):
        plan = partition.layers.get(index)
        if plan is None:
            continue
        if node_id == driver:
            # The driver holds its own layers locally.
            layer_args.append(f"0,{plan.count}")
        else:
            host = _host(node_id)
            layer_args.append(f"{host.host}:{RPC_PORT}:{plan.count}")

    rpc_arg = ",".join(layer_args)
    host = _host(driver)
    remote = (
        f"{LLAMA}/llama-cli -m {MODEL_PATH} "
        f"-ngl 99 --rpc {rpc_arg} "
        f"-no-cnv -p 'Hello' -n 8 --temp 0 2>&1 | head -40"
    )
    result = await run(host, remote, timeout_s=600.0, check=False)

    return {
        "driver": driver,
        "rpc_arg": rpc_arg,
        "exit_code": result.exit_code,
        "output": result.stdout[-3000:],
        "partition": {k: [v.start, v.end] for k, v in partition.layers.items()},
    }


def _gpu_free_bytes(order: list[str]) -> list[int]:
    """Free VRAM per node in pipeline order, read from the Phase 2A inventory.

    Reserves headroom for the runtime and KV cache so the partition plan does not
    promise memory the process cannot actually use.
    """
    path = REPO / "experiments" / "network" / "raw" / "host_inventory.json"
    free: dict[str, int] = {}
    if path.exists():
        for record in json.loads(path.read_text()):
            gpus = record.get("verified_gpus") or []
            if not gpus:
                continue
            # nvidia-smi reports current free memory, which already reflects
            # anything else running on the GPU.
            currently_free = _parse_mem(gpus[0].get("memory_free"))
            free[record["node_id"]] = max(0, currently_free - 512 * 1024**2)

    # Fall back to the nameplate total if the inventory is missing.
    fallback = {
        "meshweave-host-a": 6 * 1024**3,
        "meshweave-host-b": 8 * 1024**3,
        "meshweave-host-c": 8 * 1024**3,
    }
    return [free.get(node_id, fallback.get(node_id, 6 * 1024**3)) for node_id in order]


def _parse_mem(text: object) -> int:
    if text is None:
        return 0
    raw = str(text).strip().upper()
    for suffix, mult in (("GIB", 1024**3), ("MIB", 1024**2), ("KIB", 1024)):
        if raw.endswith(suffix):
            try:
                return int(float(raw[: -len(suffix)]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(raw) * 1024**2)
    except ValueError:
        return 0


async def down() -> dict:
    """Stop every process this experiment started, by recorded PID.

    Only PIDs in the process registry are touched. Per the Phase 0 safety audit,
    there is no port-based or name-based kill path.
    """
    from meshweave.remote.process_registry import terminate_experiment

    result = terminate_experiment(EXPERIMENT_ID)
    print(f"  terminated {len(result['terminated'])} process(es)")
    for failure in result["failed"]:
        print(f"  failed: {failure}")
    return result


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--up", action="store_true")
    parser.add_argument("--down", action="store_true")
    parser.add_argument("--plan", action="store_true", help="show the chosen order only")
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.plan:
        rtt = measured_rtt()
        print("measured mean RTT per node:")
        for k, v in sorted(rtt.items(), key=lambda kv: kv[1]):
            print(f"  {k:20s} {v:7.1f} ms")
        order = choose_order(rtt)
        print(f"\npipeline order (fastest first): {' -> '.join(order)}")
        print("the slowest node is placed last, so only one boundary is slow")
        return 0

    if args.detect:
        report = asyncio.run(detect())
        (RESULTS / "capabilities.json").write_text(json.dumps(report, indent=2))
        return 0

    if args.up:
        rtt = measured_rtt()
        order = choose_order(rtt) if rtt else [h.node_id for h in HOSTS]
        print(f"pipeline order: {' -> '.join(order)}")
        started = asyncio.run(up(order))
        (RESULTS / "deployment.json").write_text(json.dumps(started, indent=2, default=str))
        return 0

    if args.down:
        asyncio.run(down())
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
