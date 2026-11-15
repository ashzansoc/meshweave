# MESHWEAVE — Handoff and Resume

**Checkpoint:** git tag `poc-checkpoint-1` (commit `f28c34d`)
**Repo archive:** `~/Downloads/meshweave-backups/meshweave-repo-poc1.tar.gz` (7.9 MB)
**Environment manifest:** `backup/host_manifest.json`

This document exists so the work can be resumed on a different set of hosts, or by
someone who was not present for it.

---

## 1. What has been established

| Claim | Status | Evidence |
|---|---|---|
| One model executes across three independent GPU hosts | **PROVEN** | `docs/PHASE_3B_THREE_HOST_DISTRIBUTED_INFERENCE.md` |
| Numerical parity with llama.cpp | **PROVEN** | cosine 0.9999994, max abs 0.1823 — `docs/PHASE_3A_1_REFERENCE_INSTRUMENTATION.md` |
| Weights load locally, only activations cross the wire | **PROVEN** | 8 KiB per hop; weights never transferred |
| A model too large for any single GPU runs distributed | **PROVEN** (capacity) | Q8_0 OOM on every GPU — `docs/PHASE_4_1_CPU_OFFLOAD_COMPARISON.md` |
| Distributed inference beats simpler alternatives | **DISPROVEN** | CPU offload 58 tok/s vs distributed 2.2 tok/s |
| Three-host decode latency | **453 ms/token** | down from 2,663 ms after two transport fixes |

### The headline finding

**Distributed execution over ≥300 ms links is not competitive for inference on
this fleet.** Every measured configuration lost to a simpler alternative:

| Alternative | Speedup vs 3-host | Basis |
|---|---|---|
| Single GPU, model fits (Q6_K on RTX 2070S) | ~200× | 7.5 vs 453 ms/token |
| Single machine + CPU offload (Q8_0, 2 layers in RAM) | ~26× | 58 vs 2.2 tok/s |
| Quantise to fit one GPU | ~200× | same as above |

It wins only when no single machine can hold the model in **RAM** either. Even the
smallest host has 15 GB RAM against 7.6 GiB of weights.

The one untested configuration that could still win is an **A+C two-node split** —
their link measures 0.5 ms, not 310 ms, so it would pay ~1 ms per token instead of
~450 ms.

---

## 2. Repository layout

```
meshweave/
  coordinator/          FastAPI control plane
  agent/                NVML telemetry, control API, heartbeat
  registry/             live node registry with snapshot persistence
  topology/             link classification from measured RTT
  planner/              Execution Intelligence (cost-based, can refuse)
  benchmarks/           network probes, economics, report validator
  runtime/              gguf_stage.py (local weight loading), phimoe_forward.py
  remote/               SSH exec, ProcessRegistry (safety)
  models/               Qwen2.5 + PhiMoE adapters
  network/              co-location verdicts, link classes, overlay config
  experiments/
    network/            EXP-NET-001..011, host inventory, topology matrices
    model_execution/    pipeline_node.py (the distributed runtime), validation
  tools/                dump_reference.cpp (llama.cpp logits dumper)
  docs/                 one report per phase (see index below)
  benchmarks/           raw JSON per phase
  tests/                111 tests
  backup/               host_manifest.json
```

### Documents, in chronological order

| Phase | Document | What it establishes |
|---|---|---|
| 0 | `PHASE_0_SAFETY_AUDIT.md` | process-ownership safety rules |
| 2A | `PHASE_2A_NETWORK_VALIDATION.md` | all six directional links measured |
| 3 | `THREE_NODE_MODEL_EXECUTION_REPORT.md` | RPC fails at 300 ms; activation passing required |
| 3A | `PHASE_3A_NUMERICAL_CORRECTNESS.md` | three hypotheses eliminated |
| 3A.1 | `PHASE_3A_1_REFERENCE_INSTRUMENTATION.md` | **parity achieved**; the implementation was already correct |
| 3B | `PHASE_3B_THREE_HOST_DISTRIBUTED_INFERENCE.md` | **three hosts execute one model** |
| 3C | `PHASE_3C_PERFORMANCE_REPORT.md` | compute is 2.6 % of latency |
| 3D | `PHASE_3D_NETWORK_LATENCY_REPORT.md` | per-stage timeline |
| 3D.1 | `PHASE_3D_1_DIRECT_RETURN_REPORT.md` | direct return 2.1× |
| 3D.2 | `PHASE_3D_2_TCP_ANALYSIS.md` | **delayed ACK rejected**; 5.9× |
| 4 | `PHASE_4_MODEL_FEASIBILITY.md` | Q8_0 is the only viable candidate |
| 4 | `PHASE_4_DISTRIBUTED_EXECUTION.md` | capacity proven, run blocked |
| 4.1 | `PHASE_4_1_CPU_OFFLOAD_COMPARISON.md` | **CPU offload 26× faster** |
| 4.1 | `PHASE_4_1_HOST_RECOVERY.md` | which hosts need restarting |

---

## 3. Current host state

| Host | GPU | VRAM | RAM | State |
|---|---|---|---|---|
| HOST-A | GTX 1660 SUPER | 6,144 MiB | 62 GB | **blocked** — 276 MiB free |
| HOST-B | Quadro RTX 4000 | 8,192 MiB | 15 GB | available — 7,786 MiB free |
| HOST-C | RTX 2070 SUPER | 8,192 MiB | 31 GB | **blocked** — 2,270 MiB free |

Both blockers are GPU allocations **outside the container PID namespace** —
`/proc/<pid>` does not exist for them, so they cannot be terminated from inside.
Restart HOST-A and HOST-C; the data disk survives.

### If the hosts are replaced entirely

`backup/host_manifest.json` has everything: model URLs and SHA-256, the exact
llama.cpp build command (`GGML_RPC=ON` is essential), Python packages, port
mappings, and an 8-step resume checklist.

The two things most likely to be forgotten:

1. **`-DGGML_RPC=ON`** — without it `ggml-rpc-server` is not built, which cost real
   time to discover.
2. **Port mappings are fixed at rent time.** Only 10100 and 10200 are available, and
   each maps to a *different* external port per instance. Request them when renting.

---

## 4. How to resume the experiment

```bash
# 1. Verify SSH and clean VRAM on all three
for h in meshweave-host-a meshweave-host-b meshweave-host-c; do
  ssh $h 'nvidia-smi --query-gpu=memory.free --format=csv,noheader'
done
# each needs > 3,500 MiB free for a Q8_0 stage

# 2. Start stages in dependency order — C, then B, then A
ssh meshweave-host-c 'cd /root/mw && nohup /venv/main/bin/python -m experiments.model_execution.pipeline_node \
  stage --index 2 --layers auto --listen 0.0.0.0:10200 > /root/stage2.log 2>&1 &'
ssh meshweave-host-b 'cd /root/mw && nohup /venv/main/bin/python -m experiments.model_execution.pipeline_node \
  stage --index 1 --layers auto --listen 0.0.0.0:10200 --next 1.208.108.242:46092 > /root/stage1.log 2>&1 &'
ssh meshweave-host-a 'cd /root/mw && nohup /venv/main/bin/python -m experiments.model_execution.pipeline_node \
  stage --index 0 --layers auto --listen 127.0.0.1:10100 --next 83.63.1.187:33774 > /root/stage0.log 2>&1 &'

# 3. Wait for all three to log "listening" (~60 s each, model load dominates)
# 4. Run the direct-return benchmark (relay vs direct, checks correctness)

# 5. For Q8_0 instead of Q6_K:
#    prefix every stage command with MESHWEAVE_MODEL=/root/models/phi-mini-moe-Q8_0.gguf
```

Stopping: **by exact PID only**, after verifying `/proc/<pid>/cmdline`. Never by
port, never by pattern. That rule exists because an earlier cleanup killed `sshd`
on all three hosts.

---

## 5. The single most valuable next experiment

**A+C two-node pipeline with Q8_0.**

Rationale, from measured data:

- A↔C RTT is **0.5 ms**; A↔B and B↔C are ~310 ms each
- Every token currently pays ~450 ms of WAN round trip
- An A+C split would pay ~1–2 ms
- Q8_0 (7.6 GiB) splits as ~3.8 GiB per host — fits A (5.7 GiB free) and C (7.8 GiB)

If that configuration approaches the 58 tok/s CPU-offload baseline, distributed
capacity has a real use case. If it does not, the definitive finding is that this
fleet cannot benefit from distribution at all — which is worth knowing rather than
inferring.

The code already supports it: stage 0 on A with `--next` pointing at C's mapped
port, two stages instead of three.

---

## 6. Known traps, learned the hard way

| Trap | Symptom | Cause |
|---|---|---|
| `ggml-rpc-server` missing | binary not built | needs `-DGGML_RPC=ON` |
| RPC hangs at load | connection churn | 300 ms RTT breaks llama.cpp RPC weight shipping |
| `pkill -f <pattern>` | kills own SSH session | pattern matches the command line itself |
| PID extraction from `ss` | **sshd killed on three hosts** | broad kill; never do this |
| Port 10100 held by a sink | bind failure | find PID via `ss -tlnp`, verify cmdline, kill exactly |
| Position advancing per stage | cosine 0.99998 but mean abs 3.26 | `pos` is absolute start and must not change along the chain |
| `reply_to` not propagated | direct return silently falls back to relay | upstream stages build a fresh header |
| Blocking on a downstream ack | +620 ms per token | under direct return the client waits on the reply channel |
| Comparing rendered text to logits | two phases of false debugging | llama-cli shows text; the top token was a space |

---

## 7. Test and quality state

```
uv run pytest tests/ -q     # 111 passed
uv run ruff check .         # clean
uv run mypy meshweave       # clean, 53 files
```

Tests cover: GGUF loading and tensor orientation, PhiMoE forward primitives
(RMSNorm, RoPE, KV cache, MoE routing), the planner's rejection cases, link
classification, co-location verdicts, report validation, and the 29-test process
safety audit.
