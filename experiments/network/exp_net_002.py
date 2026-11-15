"""EXP-NET-002 / P2: Direct TCP connectivity between hosts.

This runs BEFORE any networking software is installed. The question is narrow and
practical: can these hosts exchange data using the port mappings Vast already
gave them?

If yes, MESHWEAVE has a working transport and the overlay becomes optional. If
no, that is the finding that justifies building one.

Two constraints discovered on real Vast hosts shape the implementation:

1. **Vast only maps ports requested at rent time.** There is no API to open a new
   one, and the external port is only visible from inside the container
   (``VAST_TCP_PORT_<internal>``). This experiment therefore uses an internal port
   that is already mapped (10100) rather than picking an arbitrary one.

2. **Background processes do not survive an SSH session ending.** ``nohup`` and
   even ``tmux`` were reaped on two of three hosts. The reliable pattern is a
   *foreground* process inside an SSH session that this script holds open for the
   duration of the measurement.

No SSH tunnels are used for the data path. SSH local forwarding would route
traffic through this machine, which is not inter-host connectivity and must not be
mistaken for it.

All six directions are measured separately because uplinks are frequently
asymmetric.

Usage:
    uv run python -m experiments.network.exp_net_002
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshweave.remote import HOSTS, RAW_DIR, Host, RemoteError  # noqa: E402
from meshweave.remote.process_registry import REGISTRY  # noqa: E402
from meshweave.remote.ssh import run_python  # noqa: E402

# Vast maps this internal port to a random external port on all three instances,
# and it is free on each. Verified before use rather than assumed.
ECHO_INTERNAL_PORT = 10100

# Identifies this run in the process registry, so cleanup can touch only its own
# processes. Per the Phase 0 safety audit, termination is by recorded PID.
EXPERIMENT_ID = "EXP-NET-002"

# The echo server. ``def handle`` must precede the accept loop: placing it after
# an infinite ``while True`` means it is never defined and the first connection
# raises NameError.
ECHO_SCRIPT = """
import socket, sys, threading

def handle(conn):
    try:
        while True:
            data = conn.recv(262144)
            if not data:
                break
            conn.sendall(data)
    except OSError:
        pass
    finally:
        conn.close()

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.listen(16)
print("listening", port, flush=True)
while True:
    conn, _ = s.accept()
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
"""

# Client: connect, send a payload repeatedly, time the round trips.
CLIENT_TEMPLATE = """
import socket, time, json
result = {"ok": False}
try:
    t0 = time.monotonic()
    s = socket.create_connection(("__IP__", __PORT__), timeout=10)
    t1 = time.monotonic()
    result["connect_s"] = round(t1 - t0, 6)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = b"x" * 65536
    total = 0
    t2 = time.monotonic()
    for _ in range(20):
        s.sendall(payload)
        got = 0
        while got < len(payload):
            chunk = s.recv(262144)
            if not chunk:
                raise OSError("peer closed")
            got += len(chunk)
        total += len(payload)
    t3 = time.monotonic()
    result.update(
        ok=True,
        bytes=total,
        throughput_mbps=round(total * 8 / (t3 - t2) / 1e6, 3),
        roundtrip_ms=round((t3 - t2) / 20 * 1000, 3),
    )
    s.close()
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
"""


CAPABILITIES_JSON = """
import json, subprocess
out = subprocess.run(["vast-capabilities"], capture_output=True, text=True).stdout
d = json.loads(out)
inst = d.get("instance", {})
print(json.dumps({
    "open_ports": [
        {
            "container_port": p.get("container_port"),
            "public_port": p.get("public_port"),
            "in_use": p.get("in_use"),
            "service": p.get("service"),
        }
        for p in inst.get("open_ports", [])
    ],
    "public_ip": inst.get("public_ipaddr"),
    "image": (d.get("image") or {}).get("name"),
}))
"""


async def live_ports(host: Host) -> dict:
    """Query vast-capabilities for the authoritative live port manifest.

    This is the source of truth for what is reachable. The guide is explicit that
    external ports are fixed at creation and cannot be added at runtime, and that
    ``in_use`` tells you whether a port is free. Environment variables are a
        secondary source; the manifest is primary.
    """
    from meshweave.remote.ssh import run_python

    try:
        result = await run_python(host, CAPABILITIES_JSON, timeout_s=40.0)
        lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
        if not lines:
            return {"error": f"no manifest (exit {result.exit_code})", "open_ports": []}
        return json.loads(lines[-1])
    except (RemoteError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "open_ports": []}


def pick_free_port(manifest: dict, count: int = 1) -> list[dict]:
    """Choose free TCP ports from the live manifest.

    Prefers a *normal* container port (<= 65535) that is not in use. The guide
    notes that normal ports are NATed to a different external number, which is
    what MESHWEAVE wants for a listener.
    """
    free = [
        p
        for p in manifest.get("open_ports", [])
        if p.get("in_use") is False and isinstance(p.get("container_port"), int)
        and p["container_port"] <= 65535
    ]
    return free[:count]


class EchoServer:
    """A listener running as a foreground process inside a live SSH session.

    Holding the SSH session open is what keeps it alive. Backgrounding with
    ``nohup`` or ``tmux`` was reaped on two of three Vast hosts.
    """

    def __init__(self, host: Host, internal_port: int) -> None:
        self.host = host
        self.internal_port = internal_port
        self.proc: asyncio.subprocess.Process | None = None
        self.reused = False
        self.pid: int | None = None

    async def start(self) -> None:
        """Launch the listener as a foreground process in a held SSH session.

        The script is base64-encoded so the remote shell never sees the newlines
        or quotes in it.

        If the port is already served -- typically by a listener from an earlier
        run -- that is accepted rather than treated as an error. MESHWEAVE must
        never kill processes on a host to obtain a port.
        """
        import base64

        from meshweave.remote.ssh import _ssh_argv

        encoded = base64.b64encode(ECHO_SCRIPT.encode()).decode()
        remote = f"echo {encoded} | base64 -d | python3 - {self.internal_port}"
        argv = _ssh_argv(self.host, remote, 10.0)
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.pid = self.proc.pid
        REGISTRY.record(
            self.pid,
            node_id=self.host.node_id,
            experiment_id=EXPERIMENT_ID,
            command=f"echo server on internal port {self.internal_port}",
        )
        assert self.proc.stdout is not None
        deadline = asyncio.get_running_loop().time() + 25
        while asyncio.get_running_loop().time() < deadline:
            try:
                line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=5)
            except TimeoutError:
                break
            if not line:
                break
            if b"listening" in line:
                return
        # No readiness line. Distinguish "already in use" (a listener exists and
        # is usable) from a genuine failure.
        detail = ""
        if self.proc.returncode is not None:
            _, err = await self.proc.communicate()
            detail = err.decode(errors="replace")
        if "Address already in use" in detail:
            self.reused = True
            return
        raise RemoteError(
            f"{self.host.describe()}: echo server never became ready"
            + (f" ({detail[:200]})" if detail else "")
        )

    async def stop(self) -> None:
        if self.proc is None or self.reused:
            # Never terminate a listener this run did not start.
            return
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except (TimeoutError, ProcessLookupError):
            self.proc.kill()


async def measure(host: Host, ip: str, port: int) -> dict:
    """Run the client on ``host`` against ``ip:port``."""
    script = (
        CLIENT_TEMPLATE.replace("__IP__", ip)
        .replace("__PORT__", str(port))
    )
    try:
        result = await run_python(host, script, timeout_s=120.0)
    except RemoteError as exc:
        return {"ok": False, "error": str(exc)}

    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {
            "ok": False,
            "error": f"no result; exit={result.exit_code} stderr={result.stderr.strip()[:160]}",
        }
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable result: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-NET-002 direct TCP connectivity")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"\n{'=' * 74}")
    print("EXP-NET-002: direct TCP connectivity")
    print(f"{'=' * 74}\n")

    endpoints: dict[str, tuple[str, int]] = {}
    manifests: dict[str, dict] = {}

    print("querying vast-capabilities for the live port manifest")
    for host in HOSTS:
        manifest = asyncio.run(live_ports(host))
        manifests[host.node_id] = manifest
        if manifest.get("error"):
            print(f"  {host.label}: manifest unavailable ({manifest['error']})")
            continue
        chosen = pick_free_port(manifest, 1)
        if not chosen:
            used = [p["container_port"] for p in manifest.get("open_ports", [])]
            print(f"  {host.label}: no free normal port (open: {used})")
            continue
        port = chosen[0]
        endpoints[host.node_id] = (host.host, int(port["public_port"]))
        print(
            f"  {host.label}: container {port['container_port']} -> "
            f"{host.host}:{port['public_port']} (free)"
        )

    # Remember which internal port each host chose, so the listener binds the
    # number that maps to the external port we will dial.
    internal_for: dict[str, int] = {}
    for node_id, (_host, public) in endpoints.items():
        for entry in manifests.get(node_id, {}).get("open_ports", []):
            if entry.get("public_port") == public:
                internal_for[node_id] = int(entry["container_port"])
                break

    if len(endpoints) < 2:
        print(
            "\nBLOCKED: fewer than two hosts expose a usable TCP port. Vast maps "
            "ports at rent time; if no spare port was requested, MESHWEAVE cannot "
            "test direct host-to-host TCP."
        )
        return 1

    print(f"\nmeasuring {len(endpoints)} x {len(endpoints) - 1} directed paths\n")
    results = asyncio.run(_run_all(HOSTS, endpoints, internal_for))

    print(f"  {'path':<40}{'ok':<5}{'connect':>10}{'throughput':>14}{'RTT':>12}")
    for r in results:
        path = f"{r['source']} -> {r['target']}"
        if r.get("ok"):
            print(
                f"  {path:<40}{'yes':<5}{r.get('connect_s', 0)*1000:>9.1f}ms"
                f"{r.get('throughput_mbps', 0):>12.2f} Mb/s"
                f"{r.get('roundtrip_ms', 0):>10.2f}ms"
            )
        else:
            print(f"  {path:<40}{'NO':<5}  {r.get('error', '')[:52]}")

    out = args.out or (RAW_DIR / "direct_tcp_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"endpoints": {k: list(v) for k, v in endpoints.items()}, "results": results},
            indent=2,
        )
    )
    print(f"\nwrote {out}")

    working = [r for r in results if r.get("ok")]
    print(f"\ndirect TCP working paths: {len(working)}/{len(results)}")
    if working:
        fastest = max(working, key=lambda r: r.get("throughput_mbps", 0))
        print(
            f"fastest: {fastest['source']} -> {fastest['target']} "
            f"at {fastest.get('throughput_mbps', 0):.1f} Mbps, "
            f"{fastest.get('roundtrip_ms', 0):.2f} ms RTT"
        )
    return 0


async def _run_all(
    hosts: list[Host],
    endpoints: dict[str, tuple[str, int]],
    internal_for: dict[str, int],
) -> list[dict]:
    by_id = {h.node_id: h for h in hosts}

    # Start a listener on every host that has a reachable endpoint. Each is a
    # foreground process in a held-open SSH session.
    # Note: no cleanup is performed. A listener left by a previous run is reused
    # rather than killed; killing processes on a host to free a port is not an
    # acceptable operation for a measurement tool.

    servers: list[EchoServer] = []
    for node_id in endpoints:
        server = EchoServer(by_id[node_id], internal_for.get(node_id, ECHO_INTERNAL_PORT))
        try:
            await server.start()
            servers.append(server)
        except (RemoteError, TimeoutError, ProcessLookupError) as exc:
            print(f"  listener failed on {node_id}: {exc}")

    print(f"  {len(servers)}/{len(endpoints)} listeners running\n")

    results: list[dict] = []
    try:
        for source in hosts:
            for target_id, (ip, port) in endpoints.items():
                if target_id == source.node_id:
                    continue
                result = await measure(source, ip, port)
                result["source"] = source.node_id
                result["target"] = target_id
                results.append(result)
    finally:
        for server in servers:
            await server.stop()

    return results


if __name__ == "__main__":
    raise SystemExit(main())
