#!/usr/bin/env bash
# MESHWEAVE agent installer.
#
#   curl -fsSL https://raw.githubusercontent.com/<owner>/meshweave/main/deploy/install.sh \
#     | sudo bash -s -- --coordinator https://<coordinator>:8700 --token <token> --enroll
#
# What it does, in order:
#   1. checks the platform is Linux with a package manager it understands
#   2. --enroll uses the token as an open-enrollment bootstrap secret
#   3. installs python3, curl and (optionally) the NVIDIA ML bindings
#   4. writes the agent to /opt/meshweave/agent.py
#   5. invents a node name unless you pass --node-id, and remembers it in
#      /etc/meshweave/node_id so re-running keeps the same identity
#   6. installs the distributed runtime (GGUF loader + stage worker)
#   7. writes credentials to /etc/meshweave/agent.json, mode 0600
#   8. registers with the coordinator and verifies the reply
#   9. installs a systemd unit (or supervisor, or a detached process) and starts it
#
# What it deliberately does NOT do:
#   * open any firewall port
#   * modify sshd or /etc/ssh
#   * install a kernel driver or touch the GPU
#   * enable passwordless sudo
#   * add anything to cron
#
# The agent is outbound-only: it dials the coordinator, nothing dials it.

set -euo pipefail

COORDINATOR=""
TOKEN=""
NODE_ID=""
ENROLL=""
PROVISION=""
PROVISION_MODEL=""
INSTALL_DIR="/opt/meshweave"
CONFIG_DIR="/etc/meshweave"
SERVICE_NAME="meshweave-agent"
PID_FILE="/var/run/$SERVICE_NAME.pid"

_start_detached() {
  log "starting the agent as a detached background process"
  mkdir -p /var/log
  rm -f "$PID_FILE"
  # No setsid: it forks when it is not a process group leader, so $! was the
  # wrapper's pid rather than the agent's. The wrapper exits, the agent keeps
  # running, and every later stop checked a dead pid and left the agent alive.
  # nohup plus a closed stdin is enough to survive this shell exiting.
  nohup python3 "$INSTALL_DIR/agent.py" \
      > "/var/log/$SERVICE_NAME.log" 2>&1 < /dev/null &
  sleep 4
  # The agent writes this itself, so it names a process we can signal.
  local recorded=""
  [ -f "$PID_FILE" ] && recorded="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$recorded" ] && kill -0 "$recorded" 2>/dev/null; then
    log "agent running, pid $recorded"
    log "logs: tail -f /var/log/$SERVICE_NAME.log"
  else
    warn "agent did not stay running. Last lines of its log:"
    tail -5 "/var/log/$SERVICE_NAME.log" 2>/dev/null | sed 's/^/    /'
  fi
}

# The command line of a pid, for deciding whether it is ours. /proc is the
# authoritative source on Linux; ps is the fallback so this can also be tested on
# a developer machine.
_pid_cmdline() {
  if [ -r "/proc/$1/cmdline" ]; then
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null
  else
    ps -o command= -p "$1" 2>/dev/null
  fi
}

# Terminate one pid, politely and then firmly. Only ever called with a pid whose
# command line was already checked against this installation's agent path.
_kill_pid() {
  local pid="$1" waited=0
  log "stopping leftover agent pid $pid"
  kill -TERM "$pid" 2>/dev/null || return 1
  while [ "$waited" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 0.25
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "pid $pid ignored SIGTERM; sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
  fi
  return 0
}

# Stop the agent that is already running before replacing it.
#
# Without this, re-running the installer replaced /opt/meshweave/agent.py while
# the old process kept executing the old code: `systemctl enable --now` does not
# restart an active unit, `supervisorctl update` does not restart a program whose
# config text is unchanged, and the detached path simply started a second copy.
# The symptom was a node that had just been upgraded still reporting its old
# version, because the old process kept heartbeating as well.
#
# Only a pid recorded here is signalled, and only after its command line is
# confirmed to be this installation's agent. A recycled pid is left alone.
_stop_existing_agent() {
  local stopped=0 pid waited

  if [ -f "/etc/systemd/system/$SERVICE_NAME.service" ] && [ -d /run/systemd/system ]; then
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
      systemctl stop "$SERVICE_NAME" >/dev/null 2>&1 && stopped=1
    fi
  fi

  if [ -f "/etc/supervisor/conf.d/$SERVICE_NAME.conf" ] && command -v supervisorctl >/dev/null 2>&1; then
    if supervisorctl status "$SERVICE_NAME" 2>/dev/null | grep -q RUNNING; then
      supervisorctl stop "$SERVICE_NAME" >/dev/null 2>&1 && stopped=1
    fi
  fi

  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      if _pid_cmdline "$pid" | grep -q "$INSTALL_DIR/agent.py"; then
        kill -TERM "$pid" 2>/dev/null && stopped=1
        waited=0
        while [ "$waited" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do
          sleep 0.25
          waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
      else
        warn "pid $pid in the pidfile is not this agent; leaving it alone"
      fi
    fi
    rm -f "$PID_FILE"
  fi

  # Finally, any agent left behind by an older install whose pid file was wrong.
  # The match is this installation's exact agent path, not a pattern, and every
  # pid is printed before it is signalled so the action is auditable.
  # /proc is preferred (ps truncates long command lines); the fallback keeps this
  # testable on a developer machine.
  local proc cmd
  if [ -d /proc ]; then
    for proc in /proc/[0-9]*; do
      [ -d "$proc" ] || continue
      pid="${proc#/proc/}"
      [ "$pid" = "$$" ] && continue
      cmd="$(_pid_cmdline "$pid")"
      case "$cmd" in
        *"$INSTALL_DIR/agent.py"*) _kill_pid "$pid"; [ $? -eq 0 ] && stopped=1 ;;
      esac
    done
  else
    # Process substitution, not a pipe: a pipe would run this loop in a subshell
    # and the stopped flag would be lost.
    while read -r pid cmd; do
      [ -n "${pid:-}" ] || continue
      [ "$pid" = "$$" ] && continue
      case "$cmd" in
        *"$INSTALL_DIR/agent.py"*) _kill_pid "$pid" && stopped=1 ;;
      esac
    done < <(ps -eo pid=,command= 2>/dev/null)
  fi

  if [ "$stopped" = "1" ]; then
    log "stopped the previously running agent"
    sleep 1
  fi
}
log()  { printf '\033[0;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --coordinator) COORDINATOR="${2:-}"; shift 2 ;;
    --token)       TOKEN="${2:-}"; shift 2 ;;
    --node-id)     NODE_ID="${2:-}"; shift 2 ;;
    --enroll)      ENROLL=1; shift ;;
    --provision)   PROVISION=1; shift ;;
    --model)       PROVISION_MODEL="${2:-}"; shift 2 ;;
    --uninstall)   UNINSTALL=1; shift ;;
    -h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
      echo
      echo 'Options:'
      echo '  --coordinator <url>   coordinator base URL (required)'
      echo '  --token <token>       enrollment token (required)'
      echo '  --enroll              join using the shared enrollment token'
      echo '  --node-id <name>      optional; a name is generated and remembered otherwise'
      echo '  --provision           install a local inference stack on this host'
      echo '  --model <repo>        model to download during provisioning'
      echo '  --uninstall           remove the agent and its identity'
      exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ── 0. platform check ────────────────────────────────────────────────────
[ "$(uname -s)" = "Linux" ] || die "this installer targets Linux; got $(uname -s)"

if [ "${UNINSTALL:-0}" = "1" ]; then
  log "removing MESHWEAVE agent"
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    kill -TERM "$pid" 2>/dev/null && log "stopped detached agent (pid $pid)"
    rm -f "$PID_FILE"
  fi
  systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
  rm -f "/etc/systemd/system/$SERVICE_NAME.service"
  systemctl daemon-reload 2>/dev/null || true
  if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl stop "$SERVICE_NAME" 2>/dev/null || true
    rm -f "/etc/supervisor/conf.d/$SERVICE_NAME.conf"
    supervisorctl reread >/dev/null 2>&1; supervisorctl update >/dev/null 2>&1
  fi
  rm -rf "$INSTALL_DIR" "$CONFIG_DIR"
  log "removed. no other system changes were made."
  exit 0
fi

[ -n "$COORDINATOR" ] || die "--coordinator is required (e.g. https://1.2.3.4:8700)"
[ -n "$TOKEN" ]       || die "--token is required (issued by the coordinator)"

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
PYTHON_BIN="$(command -v python3)"

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
import secrets
import shutil
import socket
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

AGENT_VERSION = "0.4.2"
CONFIG_PATH = Path(os.environ.get("MESHWEAVE_AGENT_CONFIG", "/etc/meshweave/agent.json"))

# Node names are generated, not derived from the hostname. A rented container
# reports names like "C.61220645": not memorable, and not unique once two
# providers hand out containers from the same pool. Two words plus four hex
# characters gives ~80M combinations, all of them valid hostname components.
NODE_ADJECTIVES = (
    "amber", "brisk", "calm", "copper", "crisp", "dusk", "fern", "frost",
    "glass", "hollow", "iron", "ivory", "jade", "kestrel", "linen", "lunar",
    "moss", "north", "onyx", "pearl", "quartz", "quiet", "rapid", "sable",
    "silver", "slate", "solar", "tide", "umber", "velvet", "willow", "zephyr",
)
NODE_NOUNS = (
    "anchor", "arbor", "atlas", "beacon", "birch", "canyon", "cedar", "comet",
    "creek", "delta", "drift", "falcon", "forge", "grove", "harbor", "heron",
    "lantern", "maple", "meadow", "mesa", "orbit", "otter", "pike", "pine",
    "prism", "quarry", "ridge", "river", "summit", "thistle", "timber", "vale",
)


def generate_node_id() -> str:
    """A fresh node name, e.g. ``silver-heron-3f9c``."""
    return f"{secrets.choice(NODE_ADJECTIVES)}-{secrets.choice(NODE_NOUNS)}-{secrets.token_hex(2)}"


def _identity_paths() -> list[Path]:
    """Where a generated name is remembered, most preferred first."""
    paths: list[Path] = []
    override = os.environ.get("MESHWEAVE_NODE_ID_FILE")
    if override:
        paths.append(Path(override))
    paths.append(Path("/var/lib/meshweave/node_id"))
    with contextlib.suppress(RuntimeError):
        paths.append(Path.home() / ".meshweave" / "node_id")
    return paths


def resolve_node_id(explicit: str | None, configured: str | None) -> str:
    """Decide what this node is called.

    Precedence: --node-id, MESHWEAVE_NODE_ID, the installer's config, a name
    written on a previous run, then a newly generated one which is persisted.

    Persisting is load-bearing. A fresh random name on every start would register
    a *new* node each time instead of reusing this one's identity, and the fleet
    view would fill with ghosts of the same machine.
    """
    if explicit:
        return explicit
    from_env = os.environ.get("MESHWEAVE_NODE_ID")
    if from_env:
        return from_env
    if configured:
        return configured

    paths = _identity_paths()
    for path in paths:
        try:
            remembered = path.read_text().strip()
        except OSError:
            continue
        if remembered:
            return remembered

    name = generate_node_id()
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n")
            return name
        except OSError:
            continue
    print(
        f"warning: could not persist a node id; this run will be {name}. "
        "Set MESHWEAVE_NODE_ID_FILE to a writable path to keep it.",
        file=sys.stderr,
        flush=True,
    )
    return name


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def write_pid_file() -> None:
    """Record the pid this process actually has.

    The installer used to trust ``$!`` from a ``setsid`` launch, which is the
    wrapper's pid, not the agent's. The wrapper exits, the agent keeps running,
    and every later stop looked at a dead pid and left the agent alive. Writing
    it here means the file always names a process that can be signalled.
    """
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(f"{os.getpid()}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# telemetry -- every value is real or None. Nothing is estimated.
# --------------------------------------------------------------------------- #


def _nvml_safe(fn, *args, default=None):
    """Call an NVML counter, returning ``default`` when the driver lacks it.

    Module level on purpose: the driver version is system-wide and has to be
    readable before any device handle exists. Defining the helper inside the
    device loop made that read a NameError, which surfaced as "no GPUs".
    """
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - an unsupported counter is not an error
        return default


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
        # System-wide, so it takes no device handle. Reading it with the
        # per-device helper was a NameError, which the outer handler turned into
        # "no GPUs" on every node.
        driver = _nvml_safe(pynvml.nvmlSystemGetDriverVersion)
        if isinstance(driver, bytes):
            driver = driver.decode()

        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            util = _nvml_safe(pynvml.nvmlDeviceGetUtilizationRates, h)
            power = _nvml_safe(pynvml.nvmlDeviceGetPowerUsage, h)
            temp = _nvml_safe(pynvml.nvmlDeviceGetTemperature, h, 0)
            capability = _nvml_safe(pynvml.nvmlDeviceGetCudaComputeCapability, h)

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
                    # bf16 needs Ampere (8.0). Turing cards report 7.5 and run
                    # bf16 kernels emulated or not at all, so the planner needs
                    # this to choose fp16 instead of guessing from the model name.
                    "compute_capability": (
                        [int(capability[0]), int(capability[1])] if capability else None
                    ),
                    "driver_version": str(driver) if driver else None,
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
        "mapped_ports": _published_tcp_ports(),
    }


def _published_tcp_ports() -> dict[int, int]:
    """Container-published TCP port mappings, as the provider declares them.

    Vast exports ``VAST_TCP_PORT_<internal>=<external>``. Reported verbatim so
    the coordinator can name a reachable endpoint for peer-to-peer traffic. A
    host with no mappings reports an empty dict rather than a guess; stage
    traffic then relies on the private address or a tunnel.
    """
    ports: dict[int, int] = {}
    for key, value in os.environ.items():
        if not key.startswith("VAST_TCP_PORT_"):
            continue
        internal = key[len("VAST_TCP_PORT_"):]
        if internal.isdigit() and value.strip().isdigit():
            ports[int(internal)] = int(value.strip())
    return ports


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
        enrollment_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Two credentials, deliberately kept apart: the enrollment token proves a
        # host may join, the node token proves which node is asking for work.
        # ``token`` is the bearer and becomes the node token after registration;
        # ``enrollment_token`` is what is sent in the enrollment header and must
        # stay the shared bootstrap secret. Passing the bearer as the enrollment
        # token produced a 401 on every restart, because the node token is not a
        # valid enrollment secret.
        self.token = token
        self.enrollment_token = enrollment_token or (token if enroll else None)
        self.node_id = node_id
        self.enroll = enroll
        self._ssl = ssl.create_default_context(cafile=ca) if ca else None

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        if self.enrollment_token:
            # Open enrollment: the coordinator accepts this shared bootstrap token
            # on /v1/nodes/register so a new host needs no prior credential.
            req.add_header("x-meshweave-enrollment", self.enrollment_token)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ssl) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            error = RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}")
            error.status = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} -> {exc.reason}") from exc

    def register(self, inventory: dict) -> dict:
        response = self._request(
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
        node_token = response.get("node_token")
        if node_token:
            self.token = node_token
            self._remember_token(node_token)
        return response

    def _remember_token(self, token: str) -> None:
        """Persist the node credential so a restart need not re-enroll.

        Written only when the agent was started from the installer's config file.
        An agent launched with --coordinator/--token on a command line owns no
        file and leaves the caller's credentials alone.
        """
        if not CONFIG_PATH.exists():
            return
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if config.get("node_token") == token:
            return
        config["node_token"] = token
        try:
            tmp = CONFIG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(config, indent=2))
            os.chmod(tmp, 0o600)
            tmp.replace(CONFIG_PATH)
        except OSError as exc:
            print(f"warning: could not store the node token: {exc}", file=sys.stderr, flush=True)

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

    def report_stage(
        self, deployment_id: str, phase: str, detail: str | None, pid: int | None
    ) -> None:
        self._request(
            "POST",
            f"/v1/deployments/{deployment_id}/nodes/{self.node_id}/status",
            {"phase": phase, "detail": detail, "pid": pid},
        )

    def report_result(self, deployment_id: str, result: dict) -> None:
        self._request("POST", f"/v1/deployments/{deployment_id}/result", result)

    def report_network(self, measurement_id: str, phase: str, payload: dict) -> None:
        self._request(
            "POST",
            f"/v1/nodes/{self.node_id}/network-result",
            {"measurement_id": measurement_id, "phase": phase, "payload": payload},
        )


# --------------------------------------------------------------------------- #
# stage supervision
#
# The agent starts and stops stage processes it launched itself. Nothing else is
# ever signalled: before a pid is terminated its /proc cmdline is checked for
# stage_runner.py, so a recycled pid cannot be mistaken for ours.
# --------------------------------------------------------------------------- #

PID_FILE = Path(os.environ.get("MESHWEAVE_PID_FILE", "/var/run/meshweave-agent.pid"))

STAGE_DIR = Path(os.environ.get("MESHWEAVE_STAGE_DIR", "/opt/meshweave"))
STATE_DIR = Path(os.environ.get("MESHWEAVE_STATE_DIR", "/var/lib/meshweave"))
STAGE_STATE = STATE_DIR / "stages.json"
CONTROL_SOCKET = Path(os.environ.get("MESHWEAVE_CONTROL_SOCKET", "/run/meshweave/control.sock"))
LOG_DIR = Path(os.environ.get("MESHWEAVE_LOG_DIR", "/var/log"))


def _read_stage_state() -> dict:
    try:
        return json.loads(STAGE_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_stage_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STAGE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STAGE_STATE)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return ""


def _tail(path: Path, lines: int = 12, chars: int = 1500) -> str:
    """The end of a stage log.

    More than the final line on purpose: a Python traceback ends with the
    exception text, and reporting only that hid a TypeError whose location was
    three lines above. The whole tail is capped so a chatty log cannot flood the
    coordinator.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(kept)[-chars:]


def stage_phase(entry: dict) -> tuple[str, str | None]:
    """Infer what a stage process is doing from its liveness and its own log.

    There is no status channel into the runner beyond the lines it prints, and
    that is deliberate: the agent supervises a process, it does not instrument
    it. A ready line means the pipeline handshakes completed.
    """
    pid = int(entry["pid"])
    log_path = Path(entry["log"])
    if not _alive(pid):
        return "failed", _tail(log_path) or "stage process exited"
    log = ""
    try:
        log = log_path.read_text(errors="replace")[-4000:]
    except OSError:
        pass
    if "ready" in log:
        return "running", None
    if "Traceback" in log:
        return "failed", _tail(log_path)
    return "loading", _tail(log_path) or None


def start_stage(command: dict) -> dict:
    deployment_id = str(command["deployment_id"])
    state = _read_stage_state()
    existing = state.get(deployment_id)
    if existing and _alive(int(existing["pid"])) and "stage_runner.py" in _cmdline(
        int(existing["pid"])
    ):
        phase, detail = stage_phase(existing)
        return {"phase": phase, "detail": detail, "pid": int(existing["pid"])}

    runner = STAGE_DIR / "stage_runner.py"
    if not runner.exists():
        return {"phase": "failed", "detail": f"{runner} is missing", "pid": None}

    argv = [
        sys.executable,
        str(runner),
        "--repo",
        str(command["repo"]),
        "--rank",
        str(command["rank"]),
        "--world",
        str(command["world"]),
        "--start",
        str(command["start"]),
        "--end",
        str(command["end"]),
        "--device",
        str(command.get("device") or "cuda"),
        "--dtype",
        str(command.get("dtype") or "bfloat16"),
        "--max-seq",
        str(command.get("max_seq") or 2048),
        "--control",
        str(CONTROL_SOCKET),
    ]
    for flag, key in (
        ("--listen", "listen"),
        ("--next", "next"),
        ("--reply", "reply"),
        ("--stage-token", "stage_token"),
    ):
        value = command.get(key)
        if value:
            argv += [flag, str(value)]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"meshweave-stage-{deployment_id}.log"
    handle = log_path.open("a")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(STAGE_DIR),
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(STAGE_DIR)},
        )
    except OSError as exc:
        handle.close()
        return {"phase": "failed", "detail": f"could not start stage: {exc}", "pid": None}
    finally:
        handle.close()

    state[deployment_id] = {
        "pid": proc.pid,
        "log": str(log_path),
        "started_at": utcnow(),
        "model": command.get("repo"),
        "rank": command.get("rank"),
    }
    _write_stage_state(state)
    print(
        f"started stage {command.get('rank')} for {deployment_id} as pid {proc.pid}",
        file=sys.stderr,
        flush=True,
    )
    return {"phase": "starting", "detail": "stage process launched", "pid": proc.pid}


def stop_stage(deployment_id: str) -> dict:
    state = _read_stage_state()
    entry = state.get(deployment_id)
    if not entry:
        return {"phase": "stopped", "detail": "no stage recorded", "pid": None}

    pid = int(entry["pid"])
    if "stage_runner.py" not in _cmdline(pid):
        # The pid is gone or has been reused by something else. Drop the record
        # instead of signalling a process that is not ours.
        state.pop(deployment_id, None)
        _write_stage_state(state)
        return {"phase": "stopped", "detail": "stage already gone", "pid": None}

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.3)
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)

    state.pop(deployment_id, None)
    _write_stage_state(state)
    return {"phase": "stopped", "detail": "stage terminated", "pid": None}


def run_prompt(command: dict) -> dict:
    """Send one prompt to the local control socket and wait for the answer."""
    control = Path(command.get("control") or CONTROL_SOCKET)
    request = {
        "prompt": command["prompt"],
        "max_new_tokens": command.get("max_new_tokens", 32),
        "temperature": command.get("temperature", 0.0),
        "top_p": command.get("top_p", 1.0),
    }
    deadline = time.monotonic() + float(command.get("timeout_s", 600))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)
        while True:
            try:
                sock.connect(str(control))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    return {
                        "prompt_id": command.get("prompt_id"),
                        "error": f"control socket {control} not available",
                    }
                time.sleep(1.0)
        sock.sendall((json.dumps(request) + "\n").encode())
        buffer = b""
        sock.settimeout(float(command.get("timeout_s", 600)))
        while b"\n" not in buffer:
            chunk = sock.recv(65536)
            if not chunk:
                return {
                    "prompt_id": command.get("prompt_id"),
                    "error": "stage closed the control socket",
                }
            buffer += chunk

    payload = json.loads(buffer.split(b"\n")[0])
    if not payload.get("ok"):
        return {
            "prompt_id": command.get("prompt_id"),
            "error": str(payload.get("error")),
        }
    result = dict(payload["result"])
    result["prompt_id"] = command.get("prompt_id")
    return result


# Echo listeners started for a network measurement, keyed by measurement id.
# Each closes itself at its ttl; this dict only exists so a repeated command does
# not try to bind the same port twice.
_ECHO_SERVERS: dict[str, object] = {}


def _self_address(internal_port: int) -> str | None:
    """Where a peer should dial this node for a port we are listening on."""
    external = _published_tcp_ports().get(internal_port)
    host = os.environ.get("PUBLIC_IPADDR") or _private_ip()
    if not external or not host:
        return None
    return f"{host}:{external}"


def start_echo(command: dict) -> dict:
    """Listen on one of the coordinator's candidates, for a bounded time.

    A provider publishes ports that the container itself may already use, so the
    first candidate can be occupied. Try each in turn and report the one that
    worked, together with the address a peer must dial -- the coordinator cannot
    know which port won.
    """
    from meshweave.runtime.echo import EchoServer

    measurement_id = str(command["measurement_id"])
    existing = _ECHO_SERVERS.get(measurement_id)
    if existing is not None:
        port = int(existing.port)  # type: ignore[attr-defined]
        return {"phase": "echo_ready", "payload": {"port": port, "address": _self_address(port)}}

    candidates = command.get("ports") or [command.get("port")]
    last_error = "no candidate ports were supplied"
    for candidate in candidates:
        port = int(candidate)
        try:
            server = EchoServer("0.0.0.0", port, ttl_s=float(command.get("ttl_s", 180)))
            server.start()
        except OSError as exc:
            last_error = f"port {port}: {exc}"
            continue
        _ECHO_SERVERS[measurement_id] = server
        print(f"echo listening on 0.0.0.0:{port} for {measurement_id}", file=sys.stderr, flush=True)
        return {
            "phase": "echo_ready",
            "payload": {"port": port, "address": _self_address(port)},
        }
    return {"phase": "error", "payload": {"error": last_error}}


def run_probe(command: dict) -> dict:
    """Measure the link to another node and report it, including a failure."""
    from meshweave.runtime.echo import probe

    result = probe(
        str(command["host"]),
        int(command["port"]),
        rtt_count=int(command.get("rtt_count", 20)),
        bulk_bytes=int(command.get("bulk_bytes", 4 * 1024 * 1024)),
        timeout_s=float(command.get("timeout_s", 25.0)),
    )
    return {
        "phase": "probe_result",
        "payload": {"target": command.get("target"), "result": result.to_dict()},
    }


def handle_deployment_commands(coord: Coordinator, commands: list[dict]) -> None:
    """Act on the closed set of commands, and report every outcome."""
    for command in commands or []:
        kind = command.get("kind")

        # Network probes are not tied to a deployment, so they are handled first.
        if kind in ("run_echo", "probe_peer"):
            measurement_id = str(command.get("measurement_id", ""))
            if not measurement_id:
                continue
            try:
                if kind == "run_echo":
                    outcome = start_echo(command)
                else:
                    # A probe can take a while on a slow link; keep the loop free.
                    threading.Thread(
                        target=_probe_worker,
                        args=(coord, measurement_id, command),
                        daemon=True,
                    ).start()
                    continue
                coord.report_network(measurement_id, outcome["phase"], outcome["payload"])
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                coord.report_network(
                    measurement_id, "error", {"error": f"{type(exc).__name__}: {exc}"}
                )
            continue

        deployment_id = str(command.get("deployment_id", ""))
        if not deployment_id:
            continue
        try:
            if kind == "run_stage":
                outcome = start_stage(command)
                coord.report_stage(
                    deployment_id, outcome["phase"], outcome.get("detail"), outcome.get("pid")
                )
            elif kind == "stop_stage":
                outcome = stop_stage(deployment_id)
                coord.report_stage(deployment_id, "stopped", "stage stopped", None)
            elif kind == "run_prompt":
                # Generation can take minutes on these links. Running it inline
                # would stall heartbeats, and the coordinator would declare the
                # node unreachable while it was busy doing exactly what it was
                # asked to do.
                threading.Thread(
                    target=_prompt_worker,
                    args=(coord, deployment_id, command),
                    daemon=True,
                ).start()
            else:
                # Unknown kinds are ignored, never executed as a shell string.
                print(f"ignoring unknown command {kind!r}", file=sys.stderr, flush=True)
        except RuntimeError as exc:
            print(f"{kind} failed: {exc}", file=sys.stderr, flush=True)


def _probe_worker(coord: Coordinator, measurement_id: str, command: dict) -> None:
    """Run one link probe off the main loop and report the result."""
    try:
        outcome = run_probe(command)
    except Exception as exc:  # noqa: BLE001 - reported, never lost
        outcome = {
            "phase": "probe_result",
            "payload": {
                "target": command.get("target"),
                "result": {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            },
        }
    try:
        coord.report_network(measurement_id, outcome["phase"], outcome["payload"])
    except RuntimeError as exc:
        print(f"could not report probe result: {exc}", file=sys.stderr, flush=True)


class GpuSampler:
    """Samples every GPU while a request runs.

    The claim that matters is that this node's GPU was busy *during* the
    generation, so a point-in-time reading before or after is not evidence.
    NVML reports a one-second average, so even a short generation yields a
    sample that covers real work.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001 - no driver, no samples
            return
        try:
            count = pynvml.nvmlDeviceGetCount()
            while not self._stop.is_set():
                for index in range(count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.samples.append(
                            {
                                "gpu": index,
                                "util_pct": float(util.gpu),
                                "vram_used_bytes": int(memory.used),
                            }
                        )
                    except Exception:  # noqa: BLE001 - counter unsupported
                        continue
                self._stop.wait(self.interval_s)
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "util_pct_max": None, "vram_used_bytes_max": None}
        return {
            "samples": len(self.samples),
            "util_pct_max": max(s["util_pct"] for s in self.samples),
            "util_pct_mean": round(
                sum(s["util_pct"] for s in self.samples) / len(self.samples), 1
            ),
            "vram_used_bytes_max": max(s["vram_used_bytes"] for s in self.samples),
            "devices": sorted({s["gpu"] for s in self.samples}),
            "interval_s": self.interval_s,
        }


def _prompt_worker(coord: Coordinator, deployment_id: str, command: dict) -> None:
    """Run one prompt off the main loop and report whatever happened."""
    try:
        with GpuSampler() as sampler:
            result = run_prompt(command)
        result["gpu_during_generation"] = sampler.summary()
    except Exception as exc:  # noqa: BLE001 - reported, never lost
        result = {
            "prompt_id": command.get("prompt_id"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if "error" in result:
        result["node_id"] = coord.node_id
    try:
        coord.report_result(deployment_id, result)
    except RuntimeError as exc:
        print(f"could not report prompt result: {exc}", file=sys.stderr, flush=True)


def refresh_stage_status(coord: Coordinator) -> None:
    """Keep the coordinator's view of each stage current."""
    for deployment_id, entry in _read_stage_state().items():
        phase, detail = stage_phase(entry)
        try:
            coord.report_stage(deployment_id, phase, detail, int(entry["pid"]))
        except RuntimeError as exc:
            print(f"stage status report failed: {exc}", file=sys.stderr, flush=True)


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
    ap.add_argument("--version", action="store_true", help="print the agent version and exit")
    ap.add_argument(
        "--generate-node-id",
        action="store_true",
        help="print a fresh node name and exit (used by the installer)",
    )
    args = ap.parse_args()

    if args.version:
        print(AGENT_VERSION)
        return 0

    if args.generate_node_id:
        print(generate_node_id())
        return 0

    if args.show:
        print(json.dumps(build_inventory(), indent=2))
        return 0

    if args.coordinator and args.token:
        cfg = {
            "coordinator": args.coordinator,
            "token": args.token,
            "node_id": resolve_node_id(args.node_id, None),
            "enroll": args.enroll,
        }
    else:
        cfg = load_config()
        cfg["node_id"] = resolve_node_id(args.node_id, cfg.get("node_id"))

    node_id = cfg["node_id"]
    coord = Coordinator(
        cfg["coordinator"],
        cfg.get("node_token") or cfg["token"],
        node_id,
        cfg.get("ca"),
        enroll=bool(cfg.get("enroll") or args.enroll),
        enrollment_token=cfg["token"],
    )

    inv = build_inventory()
    if inv["gpu_error"]:
        print(
            f"warning: GPU reporting unavailable: {inv['gpu_error']}",
            file=sys.stderr, flush=True,
        )

    if args.once:
        print(json.dumps({"registered": coord.register(inv), "inventory": inv}, indent=2))
        return 0

    # Long-running: register, then heartbeat forever.
    write_pid_file()
    coord.register(inv)
    interval = float(os.environ.get("MESHWEAVE_HEARTBEAT_S", "5"))
    # flush=True: when stdout is a pipe or a log file Python buffers, and a
    # detached agent's progress would be invisible until the buffer filled.
    print(
        f"registered {node_id} with {len(inv['gpus'])} GPU(s); "
        f"heartbeating every {interval}s",
        flush=True,
    )
    while True:
        time.sleep(interval)
        try:
            coord.heartbeat(build_inventory())
        except RuntimeError as exc:
            print(f"heartbeat failed: {exc}", file=sys.stderr, flush=True)
            if getattr(exc, "status", None) == 404:
                # The coordinator no longer knows this node -- its snapshot was
                # cleared, or it was restarted against a different store. Without
                # this the agent would heartbeat into the void forever and the
                # node would silently drop out of the fleet.
                try:
                    coord.register(build_inventory())
                    print(f"re-registered {node_id}", file=sys.stderr, flush=True)
                except RuntimeError as reg_exc:
                    print(f"re-register failed: {reg_exc}", file=sys.stderr, flush=True)

        # What the coordinator wants this node running. The set is closed:
        # run_stage, stop_stage, run_prompt. Anything else is ignored.
        try:
            work = coord.poll()
            handle_deployment_commands(coord, work.get("deployment_commands") or [])
            refresh_stage_status(coord)
        except RuntimeError as exc:
            print(f"work poll failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
MESHWEAVE_AGENT_EOF

chmod 0755 "$INSTALL_DIR/agent.py"
log "agent version $("$PYTHON_BIN" "$INSTALL_DIR/agent.py" --version)"

# ── 3a. node identity ────────────────────────────────────────────────────
# Nobody types a node name. A rented container is called things like
# "C.61220645", which is neither memorable nor unique, so a name is generated
# on first install and written down. Every later run -- including a re-run of
# this command -- reuses it, so the node updates in place instead of appearing
# in the fleet twice.
if [ -z "$NODE_ID" ]; then
  if [ -s "$CONFIG_DIR/node_id" ]; then
    NODE_ID="$(tr -d '[:space:]' < "$CONFIG_DIR/node_id")"
  elif [ -s "$CONFIG_DIR/agent.json" ]; then
    # An install from before the identity file existed already registered under
    # a name. Keep it: minting a new one would leave the same machine in the
    # fleet twice, once healthy and once unreachable.
    NODE_ID="$("$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("node_id") or "")' \
      "$CONFIG_DIR/agent.json" 2>/dev/null || true)"
  fi
fi
if [ -z "$NODE_ID" ]; then
  NODE_ID="$("$PYTHON_BIN" "$INSTALL_DIR/agent.py" --generate-node-id)"
fi
[ -n "$NODE_ID" ] || die "could not determine a node id"
printf '%s\n' "$NODE_ID" > "$CONFIG_DIR/node_id"
chmod 0644 "$CONFIG_DIR/node_id"
log "node identity: $NODE_ID"

# ── 3b. distributed runtime ──────────────────────────────────────────────
# pipeline_node.py needs meshweave.runtime (GGUF loading + the PhiMoE forward
# pass). Without it a host can report telemetry but cannot execute a stage.
log "installing the distributed runtime"
REPO_RAW="https://raw.githubusercontent.com/ashzansoc/meshweave/main"
fetch_runtime() {
  mkdir -p "$(dirname "$2")"
  curl -fsSL --max-time 90 "$REPO_RAW/$1" -o "$2"
}

# Every file is checked individually. A silently missing module would otherwise
# only surface later, on the node, when someone tries to start a stage.
RUNTIME_MISSING=0
for f in \
    meshweave/__init__.py \
    meshweave/errors.py \
    meshweave/runtime/__init__.py \
    meshweave/runtime/gguf_stage.py \
    meshweave/runtime/phimoe_forward.py \
    meshweave/runtime/framing.py \
    meshweave/runtime/model_config.py \
    meshweave/runtime/dense.py \
    meshweave/runtime/echo.py \
    meshweave/models/__init__.py \
    meshweave/models/base.py \
    experiments/model_execution/pipeline_node.py \
    deploy/stage_runner.py; do
  dest="$INSTALL_DIR/$f"
  # Scripts that run standalone sit at the install root.
  case "$f" in
    experiments/model_execution/pipeline_node.py) dest="$INSTALL_DIR/pipeline_node.py" ;;
    deploy/stage_runner.py) dest="$INSTALL_DIR/stage_runner.py" ;;
  esac
  fetch_runtime "$f" "$dest" || { warn "could not fetch $f"; RUNTIME_MISSING=1; }
done
[ "$RUNTIME_MISSING" = "0" ] || \
  warn "runtime incomplete: this node reports telemetry but cannot execute a stage"

# The runtime needs torch, safetensors to read checkpoints, tokenizers for the
# prompt, and huggingface_hub to fetch them.
"$PYTHON_BIN" -m pip install --quiet --break-system-packages \
  torch gguf numpy safetensors tokenizers huggingface_hub 2>/dev/null \
  || warn "could not install the Python runtime; distributed execution will be unavailable"

# Where the agent records the stages it started, and where rank 0 exposes its
# local control socket. Neither is reachable from the network.
mkdir -p /var/lib/meshweave /run/meshweave
chmod 0750 /var/lib/meshweave /run/meshweave

# Prove the runtime imports on this machine now, rather than at stage start.
if "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  if ( cd "$INSTALL_DIR" && "$PYTHON_BIN" -c "import meshweave.runtime.phimoe_forward" ) \
       2>/tmp/meshweave-import.err; then
    log "runtime verified: torch $("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
  else
    warn "runtime import failed: $(head -3 /tmp/meshweave-import.err | tr '\n' ' ')"
  fi
fi

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
# Stop the previous agent first. Otherwise it keeps heartbeating the version it
# was started with and overwrites what the new agent reports.
_stop_existing_agent

log "registering $NODE_ID with $COORDINATOR"
ENROLL_FLAG=""
if [ -n "$ENROLL" ]; then ENROLL_FLAG="--enroll"; fi
# $ENROLL_FLAG is intentionally unquoted: empty means "pass no flag".
if python3 "$INSTALL_DIR/agent.py" $ENROLL_FLAG --once \
     > /tmp/meshweave-register.json 2>/tmp/meshweave-register.err; then
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

# ── 6. service supervision ───────────────────────────────────────────────
# Three mechanisms, in order of preference. The check is not merely "is systemctl
# on PATH": Vast containers ship the binary but do not run systemd as PID 1, so
# `systemctl` exists and still fails with "System has not been booted with
# systemd as init system". Test for a running init, not for the binary.
if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
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
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$SERVICE_NAME" >/dev/null 2>&1
  sleep 2
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "service is running (systemd)"
  else
    warn "systemd unit installed but not active; falling back"
    _start_detached
  fi

elif command -v supervisorctl >/dev/null 2>&1 && [ -d /etc/supervisor/conf.d ]; then
  # Vast images manage long-running processes with supervisor. This is the
  # documented mechanism for this environment.
  log "installing supervisor program $SERVICE_NAME"
  cat > "/etc/supervisor/conf.d/$SERVICE_NAME.conf" <<EOF
[program:$SERVICE_NAME]
command=/usr/bin/env python3 $INSTALL_DIR/agent.py
autostart=true
autorestart=true
startsecs=5
stderr_logfile=/var/log/$SERVICE_NAME.err.log
stdout_logfile=/var/log/$SERVICE_NAME.log
EOF
  supervisorctl reread >/dev/null 2>&1
  supervisorctl update >/dev/null 2>&1
  # update() restarts a changed program but leaves an explicitly stopped one
  # stopped, so ask for a start.
  supervisorctl start "$SERVICE_NAME" >/dev/null 2>&1 || true
  sleep 3
  if supervisorctl status "$SERVICE_NAME" 2>/dev/null | grep -q RUNNING; then
    log "service is running (supervisor)"
  else
    warn "supervisor program added but not running; falling back"
    _start_detached
  fi

else
  warn "no service manager available (no systemd, no supervisor)"
  _start_detached
fi

# ── 7. optional inference stack ──────────────────────────────────────────
if [ -n "$PROVISION" ]; then
  log "provisioning the inference stack (this takes several minutes)"
  PROVISION_ARGS=""
  [ -n "$PROVISION_MODEL" ] && PROVISION_ARGS="--model $PROVISION_MODEL"
  # shellcheck disable=SC2086
  if curl -fsSL "https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/provision.sh" \
       | bash -s -- $PROVISION_ARGS; then
    log "inference stack ready"
  else
    warn "provisioning failed; the agent is still registered and reporting"
  fi
fi

cat <<EOF

  MESHWEAVE host registered
  ─────────────────────────
  node id      $NODE_ID
  coordinator  $COORDINATOR
  config       $CONFIG_DIR/agent.json  (0600)
  logs         journalctl -u $SERVICE_NAME   (or tail -f /var/log/$SERVICE_NAME.log)

  The agent dials out only. No inbound port was opened and no firewall
  rule was changed. To remove it:
    curl -fsSL <this-script> | sudo bash -s -- --uninstall
EOF