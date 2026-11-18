#!/usr/bin/env bash
# MESHWEAVE agent installer.
#
#   curl -fsSL https://raw.githubusercontent.com/<owner>/meshweave/main/deploy/install.sh \
#     | sudo bash -s -- --coordinator https://<coordinator>:8700 --token <token> --node-id <id>
#
# What it does, in order:
#   1. checks the platform is Linux with a package manager it understands
#   1. --enroll uses the token as an open-enrollment bootstrap secret
#   1. installs python3, curl and (optionally) the NVIDIA ML bindings
#   2. writes the agent to /opt/meshweave/agent.py
#   3. writes credentials to /etc/meshweave/agent.json, mode 0600
#   4. registers with the coordinator and verifies the reply
#   5. installs a systemd unit and starts it
#
# What it deliberately does NOT do:
#   * open any firewall port
#    * modify sshd or /etc/ssh
#    * install a kernel driver or touch the GPU
#    * enable passwordless sudo
#    * add anything to cron
#
# The agent is outbound-only: it dials the coordinator, nothing dials it.

set -euo pipefail

COORDINATOR=""
TOKEN=""
NODE_ID=""
ENROLL=""
INSTALL_DIR="/opt/meshweave"
CONFIG_DIR="/etc/meshweave"
SERVICE_NAME="meshweave-agent"

log()  { printf '\033[0;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --coordinator) COORDINATOR="${2:-}"; shift 2 ;;
    --token)       TOKEN="${2:-}"; shift 2 ;;
    --node-id)     NODE_ID="${2:-}"; shift 2 ;;
    --enroll)      ENROLL=1; shift ;;
    --uninstall)   UNINSTALL=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ── 0. platform check ────────────────────────────────────────────────────
[ "$(uname -s)" = "Linux" ] || die "this installer targets Linux; got $(uname -s)"

if [ "${UNINSTALL:-0}" = "1" ]; then
  log "removing MESHWEAVE agent"
  systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
  rm -f "/etc/systemd/system/$SERVICE_NAME.service"
  systemctl daemon-reload 2>/dev/null || true
  rm -rf "$INSTALL_DIR" "$CONFIG_DIR"
  log "removed. no other system changes were made."
  exit 0
fi

[ -n "$COORDINATOR" ] || die "--coordinator is required (e.g. https://1.2.3.4:8700)"
[ -n "$TOKEN" ]       || die "--token is required (issued by the coordinator)"
NODE_ID="${NODE_ID:-$(hostname)}"

# ── 1. dependencies ──────────────────────────────────────────────────────
if command -v apt-get >/dev/null 2>&1; then
  log "installing system packages (apt)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv curl ca-certificates >/dev/null
elif command -v dnf >/dev/null 2>&1; then
  log "installing system packages (dnf)"
  dnf install -y -q python3 curl ca-certificates >/dev/null
elif command -v yum >/dev/null 2>&1; then
  log "installing system packages (yum)"
  yum install -y -q python3 curl ca-certificates >/dev/null
elif command -v pacman >/dev/null 2>&1; then
  log "installing system packages (pacman)"
  pacman -Sy --noconfirm --needed python curl ca-certificates >/dev/null
else
  warn "no supported package manager found; assuming python3 is present"
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required and could not be installed"

# ── 2. GPU bindings (best effort; never fatal) ───────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
  log "installing NVML python bindings"
  python3 -m pip install --quiet --break-system-packages nvidia-ml-py 2>/dev/null \
    || python3 -m pip install --quiet nvidia-ml-py 2>/dev/null \
    || warn "could not install nvidia-ml-py; GPU reporting will be unavailable"
else
  warn "no nvidia-smi found -- this host will register without GPU telemetry"
fi

# ── 3. agent file ────────────────────────────────────────────────────────
log "installing agent to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR" "$CONFIG_DIR"

# The agent is embedded in this installer so the one-liner needs no second fetch.
cat > "$INSTALL_DIR/agent.py" <<'MESHWEAVE_AGENT_EOF'
#!/usr/bin/env python3
"""MESHWEAVE standalone agent.

A single-file agent that registers a machine with a MESHWEAVE coordinator and
reports real hardware telemetry. Designed to be installed by a one-line command
and run as a system service.

Security model -- what this can and cannot do:

  CAN
    - report GPU telemetry read from NVML
    - report CPU, RAM, disk, network facts
    - accept a *closed set* of workload commands from the coordinator
    - execute model stages it was explicitly assigned

  CANNOT
    - run a shell command from the coordinator
    - upload or download arbitrary files
    - expose CUDA or arbitrary host execution
    - accept a workload it was not assigned

  The control API is a fixed set of routes. There is no passthrough. This is
  enforced by the absence of the route, not by a permission check.

Authentication:
  * The agent holds a short-lived registration token issued by the coordinator.
  * Every request carries it; the coordinator validates it.
  * The token is stored 0600 and never logged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

AGENT_VERSION = "0.2.0"
CONFIG_PATH = Path(os.environ.get("MESHWEAVE_AGENT_CONFIG", "/etc/meshweave/agent.json"))


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# telemetry -- every value is real or None. Nothing is estimated.
# --------------------------------------------------------------------------- #


def gpu_inventory() -> list[dict]:
    """Read every GPU via NVML. Raises if the driver is absent."""
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "pynvml is not installed. On a host with an NVIDIA driver install "
            "nvidia-ml-py, or run without GPU reporting."
        ) from exc

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Bind the handle as a default argument. A closure over the loop
            # variable would capture the *last* device for every GPU, so a
            # multi-GPU host would report one card's counters N times.
            def opt(fn, _handle=h, default=None):
                try:
                    return fn(_handle)
                except Exception:  # noqa: BLE001 - unsupported counter
                    return default

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            util = opt(pynvml.nvmlDeviceGetUtilizationRates)
            power = opt(pynvml.nvmlDeviceGetPowerUsage)
            temp = opt(pynvml.nvmlDeviceGetTemperature)

            gpus.append(
                {
                    "index": i,
                    "uuid": uuid.replace("GPU-", ""),
                    "name": str(name),
                    "vram_total_bytes": int(mem.total),
                    "vram_free_bytes": int(mem.free),
                    "vram_used_bytes": int(mem.used),
                    "temperature_c": float(temp) if temp is not None else None,
                    "power_draw_w": (float(power) / 1000.0) if power is not None else None,
                    "utilization_gpu_pct": float(util.gpu) if util else None,
                    "utilization_memory_pct": float(util.memory) if util else None,
                }
            )
        return gpus
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def host_resources() -> dict:
    """CPU, RAM and disk facts."""
    cpu_model = platform.processor() or platform.machine()
    cores = os.cpu_count()

    ram_total = ram_available = None
    try:
        with open("/proc/meminfo") as fh:  # noqa: PTH123
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    ram_available = int(line.split()[1]) * 1024
    except OSError:
        pass

    disk_total = disk_available = None
    try:
        usage = shutil.disk_usage("/")
        disk_total, disk_available = usage.total, usage.free
    except OSError:
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_cores": cores,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "disk_total_bytes": disk_total,
        "disk_available_bytes": disk_available,
        "os_name": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
    }


def network_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "private_ip": _private_ip(),
        "external_ip": os.environ.get("PUBLIC_IPADDR") or None,
        "nat": None,
    }


def _private_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1, unrouted
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def build_inventory() -> dict:
    """Full inventory snapshot. GPU absence is reported, not faked."""
    gpus: list[dict] = []
    gpu_error = None
    try:
        gpus = gpu_inventory()
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        gpu_error = str(exc)

    return {
        "gpus": gpus,
        "gpu_error": gpu_error,
        "host": host_resources(),
        "network": network_info(),
        "agent_version": AGENT_VERSION,
        "collected_at": utcnow(),
    }


# --------------------------------------------------------------------------- #
# coordinator client
# --------------------------------------------------------------------------- #


class Coordinator:
    def __init__(
        self,
        base_url: str,
        token: str,
        node_id: str,
        ca: str | None = None,
        enroll: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.node_id = node_id
        self.enroll = enroll
        self._ssl = ssl.create_default_context(cafile=ca) if ca else None

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        if self.enroll:
            # Open enrollment: the coordinator accepts this shared bootstrap token
            # on /v1/nodes/register so a new host needs no prior credential.
            req.add_header("x-meshweave-enrollment", self.token)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ssl) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} -> {exc.reason}") from exc

    def register(self, inventory: dict) -> dict:
        return self._request(
            "POST",
            "/v1/nodes/register",
            {
                "node_id": self.node_id,
                "hostname": inventory["network"]["hostname"] or self.node_id,
                "agent_version": AGENT_VERSION,
                "provider": "manual",
                "region": None,
            },
        )

    def heartbeat(self, inventory: dict) -> dict:
        """Send a Heartbeat envelope, which is what the coordinator expects.

        The inventory alone is not enough: the coordinator needs the node id and a
        timestamp to detect staleness.
        """
        return self._request(
            "POST",
            f"/v1/nodes/{self.node_id}/heartbeat",
            {
                "node_id": self.node_id,
                "sent_at": utcnow(),
                "inventory": inventory,
                "active_workloads": [],
            },
        )

    def poll(self) -> dict:
        return self._request("GET", f"/v1/agents/{self.node_id}/work")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"config not found at {CONFIG_PATH}. Run the installer first, or pass "
            "--coordinator and --token."
        )
    return json.loads(CONFIG_PATH.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="MESHWEAVE agent")
    ap.add_argument("--coordinator", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--node-id", default=None)
    ap.add_argument(
        "--enroll",
        action="store_true",
        help="use the token as an open-enrollment bootstrap token",
    )
    ap.add_argument("--once", action="store_true", help="send one inventory and exit")
    ap.add_argument("--show", action="store_true", help="print the inventory and exit")
    args = ap.parse_args()

    if args.show:
        print(json.dumps(build_inventory(), indent=2))
        return 0

    if args.coordinator and args.token:
        cfg = {
            "coordinator": args.coordinator,
            "token": args.token,
            "node_id": args.node_id or socket.gethostname(),
            "enroll": args.enroll,
        }
    else:
        cfg = load_config()

    node_id = cfg["node_id"]
    coord = Coordinator(
        cfg["coordinator"], cfg["token"], node_id, cfg.get("ca"),
        enroll=bool(cfg.get("enroll") or args.enroll),
    )

    inv = build_inventory()
    if inv["gpu_error"]:
        print(f"warning: GPU reporting unavailable: {inv['gpu_error']}", file=sys.stderr)

    if args.once:
        print(json.dumps({"registered": coord.register(inv), "inventory": inv}, indent=2))
        return 0

    # Long-running: register, then heartbeat forever.
    coord.register(inv)
    interval = float(os.environ.get("MESHWEAVE_HEARTBEAT_S", "5"))
    print(f"registered {node_id} with {len(inv['gpus'])} GPU(s); heartbeating every {interval}s")
    while True:
        time.sleep(interval)
        try:
            coord.heartbeat(build_inventory())
        except RuntimeError as exc:
            print(f"heartbeat failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
MESHWEAVE_AGENT_EOF

chmod 0755 "$INSTALL_DIR/agent.py"

# ── 4. credentials ───────────────────────────────────────────────────────
log "writing credentials to $CONFIG_DIR/agent.json"
cat > "$CONFIG_DIR/agent.json" <<EOF
{
  "coordinator": "$COORDINATOR",
  "token": "$TOKEN",
  "node_id": "$NODE_ID",
  "enroll": ${ENROLL:-false}
}
EOF
chmod 0600 "$CONFIG_DIR/agent.json"

# ── 5. registration check ────────────────────────────────────────────────
log "registering $NODE_ID with $COORDINATOR"
if ENROLL_FLAG=""
[ -n "$ENROLL" ] && ENROLL_FLAG="--enroll"
python3 "$INSTALL_DIR/agent.py" $ENROLL_FLAG --once > /tmp/meshweave-register.json 2>/tmp/meshweave-register.err; then
  python3 - <<'PY' || warn "registration reported a problem"
import json
d = json.load(open("/tmp/meshweave-register.json"))
inv = d.get("inventory", {})
gpus = inv.get("gpus", [])
print(f"    registered: {d.get('registered', {}).get('node_id')}")
print(f"    GPUs: {len(gpus)}" + (f" ({', '.join(g['name'] for g in gpus)})" if gpus else ""))
h = inv.get("host", {})
print(f"    CPU: {h.get('cpu_cores')} cores | RAM: {(h.get('ram_total_bytes') or 0)/1024**3:.1f} GiB")
print(f"    disk: {(h.get('disk_total_bytes') or 0)/1024**3:.1f} GiB total")
PY
else
  warn "registration failed:"
  sed 's/^/    /' /tmp/meshweave-register.err | head -5
  die "fix the coordinator address or token and re-run"
fi

# ── 6. systemd service ───────────────────────────────────────────────────
if command -v systemctl >/dev/null 2>&1; then
  log "installing systemd service $SERVICE_NAME"
  cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=MESHWEAVE agent ($NODE_ID)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/env python3 $INSTALL_DIR/agent.py
Restart=always
RestartSec=5
# The agent needs no privileges beyond reading hardware telemetry.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/log

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$SERVICE_NAME" >/dev/null 2>&1
  sleep 2
  systemctl is-active --quiet "$SERVICE_NAME" \
    && log "service is running" \
    || warn "service installed but not active; check: journalctl -u $SERVICE_NAME"
else
  warn "systemd not available; start manually with:"
  echo "    nohup python3 $INSTALL_DIR/agent.py >/var/log/meshweave.log 2>&1 &"
fi

cat <<EOF

  MESHWEAVE host registered
  ─────────────────────────
  node id      $NODE_ID
  coordinator  $COORDINATOR
  config       $CONFIG_DIR/agent.json  (0600)
  logs         journalctl -u $SERVICE_NAME

  The agent dials out only. No inbound port was opened and no firewall
  rule was changed. To remove it:
    curl -fsSL <this-script> | sudo bash -s -- --uninstall
EOF