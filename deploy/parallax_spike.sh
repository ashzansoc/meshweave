#!/usr/bin/env bash
# Parallax Turing spike.
#
# Answers one question: does Parallax load and run a model on a Turing GPU?
# Everything else about it is documented; this is the part that cannot be read
# from source.
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
VENV="$ROOT/venv"
MODEL="${MESHWEAVE_SPIKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RESULT="/var/log/parallax-spike.json"
PORT="${MESHWEAVE_SPIKE_PORT:-3000}"

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
command -v python3 >/dev/null 2>&1 || die "python3 is required"

mkdir -p "$ROOT"

# ── 1. Is there a GPU we can test? ────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "no nvidia-smi; this machine cannot answer the Turing question"
fi

# ── 2. Isolated environment ───────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
  log "creating an isolated environment at $VENV"
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true

log "installing Parallax (this pulls vLLM and Lattica; it can take a while)"
if ! "$VENV/bin/python" -m pip install --quiet "parallax" 2>"$ROOT/install.err"; then
  warn "install failed; first lines of the error:"
  head -20 "$ROOT/install.err" | sed 's/^/    /'
  python3 - "$ROOT" "$MODEL" <<'PY'
import json, sys, datetime
root, model = sys.argv[1], sys.argv[2]
json.dump({
    "tested_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "model": model,
    "stage": "install",
    "ok": False,
    "error": "pip install failed; see /opt/parallax-spike/install.err",
}, open("/var/log/parallax-spike.json", "w"), indent=2)
PY
  die "could not install Parallax"
fi

VERSION="$("$VENV/bin/python" -c 'from importlib.metadata import version; print(version("parallax"))' 2>/dev/null || echo "unknown (import failed)")"
log "installed Parallax $VERSION"
if ! "$VENV/bin/parallax" --help >/dev/null 2>&1; then
  warn "the 'parallax' command is not usable in the venv"
fi

# ── 3. Start a single node ────────────────────────────────────────────────
# Single node only: no --join, no published peer ports, no scheduler.
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

cd "$ROOT"
set +e
"$VENV/bin/parallax" serve -m "$MODEL" --dtype float16 --host 127.0.0.1 --port "$PORT" \
  > "$ROOT/server.log" 2>&1 &
SERVER_PID=$!
set -e
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
python3 - "$ROOT" "$MODEL" "$VERSION" "$READY" <<'PY'
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
chat = f"{root}/result.json"
try:
    payload = json.load(open(chat))
    result["response"] = payload
    choices = payload.get("choices") or []
    result["text"] = choices[0]["message"]["content"] if choices else None
    result["ok"] = bool(result["text"])
except Exception:
    err = f"{root}/chat.err"
    try:
        result["error"] = open(err).read()[:600]
    except Exception:
        result["error"] = "no response and no error captured"

try:
    result["server_log_tail"] = "\n".join(open(f"{root}/server.log").read().splitlines()[-25:])
except Exception:
    pass

json.dump(result, open("/var/log/parallax-spike.json", "w"), indent=2)
print(json.dumps({k: result[k] for k in ("ok", "server_ready", "text") if k in result}, indent=2))
PY

log "result written to $RESULT"

if grep -q '"ok": true' "$RESULT" 2>/dev/null; then
  log "SPIKE PASSED: Parallax runs this model on this GPU"
else
  warn "SPIKE FAILED or inconclusive"
  "$VENV/bin/python" - "$RESULT" <<'SPIKE_PY'
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
SPIKE_PY
fi

if [ -f "$ROOT/parallax.pid" ]; then
  warn "the server is still running; stop it with: sudo kill $(cat "$ROOT/parallax.pid")"
fi
warn "remove everything with: sudo rm -rf $ROOT $RESULT"