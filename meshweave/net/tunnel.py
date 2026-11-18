"""Expose a local coordinator so remote agents can reach it.

A coordinator running on a laptop is behind NAT: a rented GPU host cannot dial
127.0.0.1:8700 on someone else's machine. This module provides the missing hop.

Two strategies, in preference order:

  1. **Cloudflare quick tunnel** -- Vast images ship ``cloudflared``, and it needs
     no account for a quick tunnel. Gives an https://*.trycloudflare.com URL that
     forwards to the local port. Ephemeral: the URL changes on restart.

  2. **Tailscale/WireGuard overlay** -- if the operator has one, use its private
     IP directly. Stable, private, but needs prior setup.

For anything beyond a demo, run the coordinator on a host with a public IP and
skip this entirely.

Usage:
    uv run python -m meshweave.net.tunnel --port 8700
    uv run python -m meshweave.net.tunnel --port 8700 --connect   # also print the agent command
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time

TUNNEL_HOST = "trycloudflare.com"


def cloudflared_available() -> bool:
    return shutil.which("cloudflared") is not None


def start_quick_tunnel(port: int, timeout_s: float = 30.0) -> str | None:
    """Start a Cloudflare quick tunnel and return its public URL.

    Reads the URL from cloudflared's stderr, which is where it announces the
    tunnel. Returns None if it does not appear within the timeout.
    """
    if not cloudflared_available():
        return None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + timeout_s
    assert proc.stderr is not None
    while time.monotonic() < deadline:
        line = proc.stderr.readline()
        if not line:
            if proc.poll() is not None:
                return None
            continue
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            return match.group(0)

    proc.terminate()
    return None


def local_ip() -> str | None:
    """Best-effort LAN address, for the overlay case."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1, unrouted
        addr: str = s.getsockname()[0]
        return addr
    except OSError:
        return None
    finally:
        s.close()


def agent_command(coordinator_url: str, enroll_token: str, node_id: str) -> str:
    return (
        f"curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/"
        f"deploy/install.sh | sudo bash -s -- "
        f"--coordinator {coordinator_url} --token {enroll_token} "
        f"--node-id {node_id} --enroll"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--connect",
        action="store_true",
        help="print the ready-to-run agent command once the tunnel is up",
    )
    ap.add_argument(
        "--enroll-token",
        default=None,
        help="enrollment token to embed in the printed command",
    )
    args = ap.parse_args()

    print(f"starting a quick tunnel to 127.0.0.1:{args.port} ...", file=sys.stderr)

    url = start_quick_tunnel(args.port, args.timeout)
    if url is None:
        print(
            "cloudflared unavailable or did not produce a URL.\n"
            "Options:\n"
            "  * install cloudflared, or\n"
            "  * run the coordinator on a host with a public IP, or\n"
            f"  * use an overlay: agents could reach http://{local_ip()}:{args.port}\n"
            "    if they are on the same private network.",
            file=sys.stderr,
        )
        return 1

    print(f"coordinator reachable at {url}", flush=True)
    if args.connect and args.enroll_token:
        print()
        print("run this on each new GPU host:")
        print(f"  {agent_command(url, args.enroll_token, '<node-id>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
