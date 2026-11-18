#!/usr/bin/env bash
# Parallax Turing spike.
#
# Answers one question: does Parallax load and run a model on a Turing GPU?
# Everything else about it is documented; this is the part that cannot be read
# from source.
#
# Install note: Parallax is NOT on PyPI under this name. The `parallax` package
# on PyPI is an unrelated SSH fan-out tool (krig/parallax). The supported install
# is a git clone plus the project's own ./install.sh, which creates a venv with
# uv. An earlier version of this script used `pip install parallax` and therefore
# tested the wrong software entirely.
#
# Contained by design:
#   * installs into /opt/parallax-spike, never touching /opt/meshweave
#   * runs a single node, so nothing is published and no peer traffic is needed
#   * writes its result to /var/log/parallax-spike.json
#   * --uninstall removes everything it created
#
# Parallax ships no authentication on its HTTP ports, which is why this stays
# single-node and local.

set -euo pipefail

ROOT="/opt/parallax-spike"
SRC="$ROOT/src"
VENV="$SRC/.venv"
REF="${MESHWEAVE_SPIKE_REF:-v0.1.2}"
MODEL="${MESHWEAVE_SPIKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RESULT="/var/log/parallax-spike.json"
PORT="${MESHWEAVE_SPIKE_PORT:-3000}"
URL="https://github.com/GradientHQ/parallax.git"

log()  { printf '\033[0;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
  if [ -f "$ROOT/parallax.pid" ]; then
    pid="$(cat "$ROOT/parallax.pid")"
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "parallax"; then
      kill -TERM "$pid" 2>/dev/null || true
      log "stopped the Parallax server (pid $pid)"
    fi
    rm -f "$ROOT/parallax.pid"
  fi
  rm -rf "$ROOT" "$RESULT"
  log "removed the spike"
  exit 0
fi

[ "$(uname -s)" = "Linux" ] || die "this spike targets Linux; got $(uname -s)"
command -v git >/dev/null 2>&1 || die "git is required to clone Parallax"
PYTHON_BIN="$(command -v python3)"

mkdir -p "$ROOT"
cd "$ROOT"

# ── 1. Stop a previous server ─────────────────────────────────────────────
if [ -f "$ROOT/parallax.pid" ]; then
  old_pid="$(cat "$ROOT/parallax.pid" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null | grep -q "parallax"; then
    log "stopping the previous server (pid $old_pid)"
    kill -TERM "$old_pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$ROOT/parallax.pid"
fi

# ── 2. Fetch and install ──────────────────────────────────────────────────
if [ ! -d "$SRC/.git" ]; then
  log "cloning Parallax $REF"
  git clone --quiet --depth 1 --branch "$REF" "$URL" "$SRC" 2>"$ROOT/clone.err" \
    || { warn "clone failed:"; head -5 "$ROOT/clone.err" | sed 's/^/    /'; \
         python3 -c "import json; json.dump({'stage':'clone','ok':False,'error':'clone failed'}, open('/var/log/parallax-spike.json','w'))"; exit 1; }
fi

# The tag has no install.sh (that was added on main later). Its documented
# source install is an editable install of the gpu extra inside a venv.
if [ ! -x "$VENV/bin/parallax" ]; then
  PYVER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$PYVER" in
    3.11|3.12|3.13) : ;;
    *) warn "Parallax requires Python >=3.11,<3.14; this host has $PYVER" ;;
  esac

  log "creating a virtual environment at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"

  log "installing the gpu extra (sglang and torch; this is several GB)"
  if ! "$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1; then
    warn "could not upgrade pip; continuing"
  fi
  if ! ( cd "$SRC" && "$VENV/bin/python" -m pip install -e '.[gpu]' ) \
        >"$ROOT/install.log" 2>&1; then
    warn "install failed; last lines:"
    tail -20 "$ROOT/install.log" | sed 's/^/    /'
    python3 -c "import json; json.dump({'stage':'install','ok':False,'error':'pip install -e .[gpu] failed; see /opt/parallax-spike/install.log'}, open('/var/log/parallax-spike.json','w'))"
    exit 1
  fi
fi

VERSION="$("$VENV/bin/python" -c 'from importlib.metadata import version; print(version("parallax"))' 2>/dev/null || echo unknown)"
log "installed Parallax $VERSION"

# ── 3. Start a single node ────────────────────────────────────────────────
cd "$ROOT"
"$VENV/bin/parallax" serve -m "$MODEL" --dtype float16 --host 127.0.0.1 --port "$PORT" \
  > "$ROOT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ROOT/parallax.pid"
log "started the server (pid $SERVER_PID); waiting for it to load"

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done

RESULT_JSON="$ROOT/result.json"
if [ "$READY" = "1" ]; then
  log "server answered on port $PORT"
  curl -fsS -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":16,\"temperature\":0}" \
    > "$RESULT_JSON" 2>"$ROOT/chat.err" || warn "generation request failed"
else
  warn "server did not become ready"
fi

# ── 4. Record the outcome ─────────────────────────────────────────────────
python3 - "$ROOT" "$MODEL" "$VERSION" "$READY" <<'SPIKE_PY'
import json, sys, datetime, subprocess
root, model, version, ready = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"

def nvidia():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"

result = {
    "tested_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "model": model,
    "parallax_version": version,
    "gpu": nvidia(),
    "stage": "serve",
    "server_ready": ready,
    "ok": False,
}
try:
    payload = json.load(open(f"{root}/result.json"))
    result["response"] = payload
    choices = payload.get("choices") or []
    result["text"] = choices[0]["message"]["content"] if choices else None
    result["ok"] = bool(result["text"])
except Exception:
    try:
        result["error"] = open(f"{root}/chat.err").read()[:600]
    except Exception:
        result["error"] = "no response and no error captured"

try:
    result["server_log_tail"] = "\n".join(open(f"{root}/server.log").read().splitlines()[-25:])
except Exception:
    pass

json.dump(result, open("/var/log/parallax-spike.json", "w"), indent=2)
print(json.dumps({k: result[k] for k in ("ok", "server_ready", "text") if k in result}, indent=2))
SPIKE_PY

log "result written to $RESULT"

if grep -q '"ok": true' "$RESULT" 2>/dev/null; then
  log "SPIKE PASSED: Parallax runs this model on this GPU"
else
  warn "SPIKE FAILED or inconclusive"
  python3 - "$RESULT" <<'DIAG_PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("error"):
    print("  error:")
    for line in str(d["error"]).splitlines()[:12]:
        print("   ", line)
if d.get("server_log_tail"):
    print("  server log (last lines):")
    for line in str(d["server_log_tail"]).splitlines()[-12:]:
        print("   ", line)
DIAG_PY
fi

if [ -f "$ROOT/parallax.pid" ]; then
  warn "the server is still running; stop it with: sudo kill $(cat "$ROOT/parallax.pid")"
fi
warn "remove everything with: sudo rm -rf $ROOT $RESULT"