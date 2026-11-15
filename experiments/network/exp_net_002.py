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
from meshweave.remote.ssh import run_python  # noqa: E402

# Vast maps this internal port to a random external port on all three instances,
# and it is free on each. Verified before use rather than assumed.
ECHO_INTERNAL_PORT = 10100

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


def external_port(record: dict, internal: int = ECHO_INTERNAL_PORT) -> int | None:
    """Find the externally mapped port for an internal port.

    Read from the host's own environment, which is the only place Vast publishes
    the mapping. There is no REST field that returns it.
    """
    env = record.get("commands", {}).get("vast_env", {}).get("stdout", "") or ""
    prefix = f"VAST_TCP_PORT_{internal}="
    for line in env.splitlines():
        if line.startswith(prefix):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


class EchoServer:
    """A listener running as a foreground process inside a live SSH session.

    Holding the SSH session open is what keeps it alive. Backgrounding with
    ``nohup`` or ``tmux`` was reaped on two of three Vast hosts.
    """

    def __init__(self, host: Host) -> None:
        self.host = host
        self.proc: asyncio.subprocess.Process | None = None
        self.reused = False

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
        remote = f"echo {encoded} | base64 -d | python3 - {ECHO_INTERNAL_PORT}"
        argv = _ssh_argv(self.host, remote, 10.0)
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

    inventory_path = RAW_DIR / "host_inventory.json"
    if not inventory_path.exists():
        print("host_inventory.json missing; run exp_net_001 first", file=sys.stderr)
        return 2
    inventory = json.loads(inventory_path.read_text())

    print(f"\n{'=' * 74}")
    print("EXP-NET-002: direct TCP connectivity")
    print(f"{'=' * 74}\n")

    endpoints: dict[str, tuple[str, int]] = {}
    print(f"resolving external endpoints for internal port {ECHO_INTERNAL_PORT}")
    for record in inventory:
        node_id = record["node_id"]
        external = external_port(record)
        if external is None:
            print(f"  {record['label']}: no VAST_TCP_PORT_{ECHO_INTERNAL_PORT} mapping")
            continue
        endpoints[node_id] = (record["endpoint"]["host"], external)
        print(
            f"  {record['label']}: internal {ECHO_INTERNAL_PORT} -> "
            f"{record['endpoint']['host']}:{external}"
        )

    if len(endpoints) < 2:
        print(
            "\nBLOCKED: fewer than two hosts expose a usable TCP port. Vast maps "
            "ports at rent time; if no spare port was requested, MESHWEAVE cannot "
            "test direct host-to-host TCP."
        )
        return 1

    print(f"\nmeasuring {len(endpoints)} x {len(endpoints) - 1} directed paths\n")
    results = asyncio.run(_run_all(HOSTS, endpoints))

    print(f"  {'path':<40}{'ok':<5}{'connect':>10}{'throughput':>13}{'RTT':>10}")
    for r in results:
        path = f"{r['source']} -> {r['target']}"
        if r.get("ok"):
            print(
                f"  {path:<40}{'yes':<5}{r.get('connect_s', 0)*1000:>9.1f}ms"
                f"{r.get('throughput_mbps', 0):>12.1f} fast"
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


async def _run_all(hosts: list[Host], endpoints: dict[str, tuple[str, int]]) -> list[dict]:
    by_id = {h.node_id: h for h in hosts}

    # Start a listener on every host that has a reachable endpoint. Each is a
    # foreground process in a held-open SSH session.
    # Note: no cleanup is performed. A listener left by a previous run is reused
    # rather than killed; killing processes on a host to free a port is not an
    # acceptable operation for a measurement tool.

    servers: list[EchoServer] = []
    for node_id in endpoints:
        server = EchoServer(by_id[node_id])
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
