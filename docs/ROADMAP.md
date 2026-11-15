# Roadmap

Priority order follows the project brief. The ordering principle: **prove
distributed inference before building anything around it.** No Kubernetes, no
containers-as-platform, no training, no cloud management until the first
distributed inference experiment produces real, reproducible results.

---

## Where we are

```
P0  ████████████████████  complete   research, architecture, harness
P1  ████████████████░░░░  code complete, needs hardware
P2  ████████████████░░░░  code complete, needs hardware
P3  ██████████░░░░░░░░░░  harness written, blocked
P4  ██████████████░░░░░░  runtime works over sockets, needs GPUs to prove
P5  ████████████░░░░░░░░  EXP-011 written, blocked  ← current milestone
P6  ░░░░░░░░░░░░░░░░░░░░  not started
P7  █████████████░░░░░░░  planner + EXP-010 done, needs hardware
P8  ██████████░░░░░░░░░░  recovery semantics implemented, not exercised
P9  ░░░░░░░░░░░░░░░░░░░░  deferred
P10 ░░░░░░░░░░░░░░░░░░░░  deferred
```

**The single most important next action:** rent two Vast.ai instances, start agents
on them, and run EXP-011. Everything else is downstream of that.

---

## P0 — Research and architecture ✓

Delivered:

- Competitive analysis of Exo, Petals, Hivemind, vLLM, SGLang, Ray, PyTorch
  Distributed, NCCL, DeepSpeed (`docs/DECISIONS.md`, `docs/BENCHMARKS.md`).
- Architecture document (`docs/ARCHITECTURE.md`).
- Technology selection with reasoning (`docs/DECISIONS.md`, D-001..D-015).
- Network design: three environments, measured not assumed.
- Security design and threat model (`docs/SECURITY.md`).
- Experiment matrix (`docs/EXPERIMENTS.md`).

## P1 — Two-node communication and authentication

Code complete:

- Coordinator with registration, heartbeat, registry, topology, planning,
  workload endpoints.
- Agent with real NVML telemetry, control API, heartbeat loop, serve loop.
- mTLS transport, node certificates, workload-scoped HMAC tokens.
- Verified over real HTTP sockets: 44 tests passing.

Blocked on: a real GPU host to run an agent against.

## P2 — Network benchmarking

Code complete:

- RTT, jitter, packet loss, single/multi-stream TCP throughput, sustained
  throughput, connection stability, UDP support.
- N×N matrix (8 nodes → 28 pairs → 56 directed measurements).
- Topology engine classifying links from measurement, with `require_datacenter()`
  gating tensor parallelism.
- Echo endpoint on each agent.

Blocked on: two hosts to probe.

## P3 — Single-GPU baseline

Harness written (`experiments/exp001_single_gpu.py`) with a concurrency sweep.
Blocked on: a GPU running vLLM.

## P4 — Distributed inference

Working:

- Qwen2.5 adapter with memory arithmetic validated against published configs.
- Partitioner that balances layers and refuses impossible configurations with a
  stated deficit.
- Activation transport with binary framing, per-hop byte and latency recording.
- Pipeline runtime with preflight probing, prefill, decode loop, token relay.
- Verified end to end over real sockets (transport only — no weights).

Blocked on: GPUs with weights loaded. **This is the gap that matters most.**

## P5 — Distributed VRAM capacity validation (EXP-011)

Written and blocked. This is the project's first major technical milestone. See
`docs/EXPERIMENTS.md` for the procedure and success criterion.

## P6 — Four/eight-node scaling

Not started. Depends on P5 succeeding.

## P7 — Replication versus partitioning

Planner and EXP-010 complete. Produces the cost comparison and crossover analysis
from planning assumptions; needs real measurements to be a benchmark.

## P8 — Fault tolerance

Recovery semantics implemented and unit-tested. Heartbeat timeout and reaping work.
Not exercised against live hardware.

Known limitation documented rather than hidden: a pipeline that loses a middle
stage cannot continue the in-flight request, because the KV cache for those layers
is gone. Recovery is a replay from the prompt.

## P9 — Distributed training

Deferred, deliberately. DDP, FSDP, DeepSpeed ZeRO-1/2/3, Ray. Training has
different communication patterns from inference and must not be conflated with it.
Not started until P5 produces a result.

## P10 — Dynamic orchestration

Deferred. Dynamic node discovery, adaptive topology, automatic placement. Ray
integration belongs here, with an honest assessment of where it helps and where it
adds overhead.

---

## Known gaps and cleanup

| Item | Priority | Notes |
|---|---|---|
| Rename on-disk directory `Meshwave` → `meshweave` | low | tool layer blocked the rename; package/CLI already correct |
| Enforce CN-to-node-id binding on every request | medium | P8 security hardening |
| Enforce TLS for non-loopback coordinator binds | medium | documented, not enforced |
| Resource quotas on reported telemetry | medium | a node can currently over-report VRAM |
| Rate limiting on agent endpoints | low | DoS resistance |
| Dashboard (React + TS + Tailwind) | low | reads real telemetry only; no fabricated charts |
| `mypy --strict` clean | medium | in progress |

---

## What success looks like

The brief defines it, and this roadmap adopts it unchanged:

1. **First genuine technical success.** Two independent Vast hosts, two physical
   NVIDIA GPUs, a model partitioned across them, real activation transfers,
   successful inference, correct output, reproducible measurements.
2. **Second.** Adding independently hosted GPUs increases the maximum model size
   that can be executed.
3. **Third.** The system survives a node failure or recovers gracefully.
4. **Fourth.** The distributed configuration has an economically defensible use
   case.

MESHWEAVE is at step zero of step one. Everything is built except the part that
requires renting a GPU.