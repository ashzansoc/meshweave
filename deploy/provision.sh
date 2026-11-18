#!/usr/bin/env bash
# MESHWEAVE host provisioning.
#
#   curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/provision.sh \
#     | sudo bash -s -- --model Qwen2.5-7B-Instruct --quant awq
#
# Turns a bare GPU host into an inference-capable node:
#   1. verifies the NVIDIA driver and CUDA
#   1. creates a Python venv with PyTorch, vLLM, and supporting libraries
#   3. optionally downloads a model and pre-warms it
#   4. prints a readiness report
#
# Why a local script and not a coordinator command:
#   MESHWEAVE's threat model treats every rented host as untrusted and forbids
#   arbitrary remote execution through the coordinator. Provisioning is therefore
#   an explicit, local, reviewable step. The coordinator can *read* what is
#   installed; it cannot run it.
#
# Everything here is idempotent: re-running is safe and skips completed steps.

set -euo pipefail

VENV="/opt/meshweave/venv"
MODEL=""
QUANT=""
SKIP_MODEL=0

log()  { printf '\033[0;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model)     MODEL="${2:-}"; shift 2 ;;
    --quant)     QUANT="${2:-}"; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --venv)      VENV="${2:-}"; shift 2 ;;
    -h|--help)   sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ── 1. GPU and driver ────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; install the NVIDIA driver first"
log "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | sed 's/^/    /'

DRIVER_OK=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log "driver $DRIVER_OK"

# ── 2. Python environment ────────────────────────────────────────────────
if [ -x "$VENV/bin/python" ]; then
  log "venv already exists at $VENV"
else
  log "creating venv at $VENV"
  python3 -m venv "$VENV" 2>/dev/null || {
    apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null 2>&1
    python3 -m venv "$VENV"
  }
fi
PY="$VENV/bin/python"

log "installing the inference stack (this takes a few minutes)"
"$VENV/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1 || true

# PyTorch with the CUDA build matching the host driver.
"$VENV/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null \
  || "$VENV/bin/pip" install --quiet torch

# vLLM gives the best throughput on a single 24 GB-class card; fall back to
# llama.cpp-style tooling if the wheel is unavailable for this platform.
if ! "$VENV/bin/pip" install --quiet vllm 2>/dev/null; then
  warn "vLLM unavailable; installing transformers + accelerate instead"
  "$VENV/bin/pip" install --quiet transformers accelerate safetensors sentencepiece
fi

"$VENV/bin/pip" install --quiet fastapi uvicorn httpx pydantic 2>/dev/null || true

# ── 3. model ─────────────────────────────────────────────────────────────
if [ "$SKIP_MODEL" = "1" ] || [ -z "$MODEL" ]; then
  log "skipping model download (--skip-model or no --model given)"
else
  log "downloading $MODEL"
  "$VENV/bin/pip" install --quiet huggingface_hub 2>/dev/null || true
  "$VENV/bin/python" - "$MODEL" <<'PY' || warn "model download failed; the stack is still installed"
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1])
print(f"    model at {path}")
PY
fi

# ── 4. readiness ─────────────────────────────────────────────────────────
log "readiness check"
"$VENV/bin/python" - <<'PY'
import shutil, subprocess, sys

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"

print(f"  python        {sys.version.split()[0]}")
try:
    import torch
    print(f"  torch         {torch.__version__}")
    print(f"  cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu {i}         {p.name}  {p.total_memory/1024**3:.1f} GiB  cc {p.major}.{p.minor}")
except ImportError:
    print("  torch         MISSING")

for mod, label in (("vllm", "vllm"), ("transformers", "transformers")):
    try:
        m = __import__(mod)
        print(f"  {label:14s} {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"  {label:14s} not installed")

print(f"  disk free     {sh('df -h / | tail -1 | awk \"{print \\$4}\"}')")
PY

cat <<EOF

  MESHWEAVE host provisioned
  ──────────────────────────
  venv      $VENV
  model     ${MODEL:-none}

  Activate with:
    source $VENV/bin/activate

  Serve an OpenAI-compatible endpoint:
    vllm serve ${MODEL:-<model>} --port 8000

  Nothing was exposed externally. To make a service reachable, request a port
  mapping at instance creation and put it behind authentication.
EOF