# Phase 3 baseline

Reproduced before any optimization, with corrected measurement.

**Configuration:** Qwen2.5-0.5B-Instruct, fp16, two replicas (one per node),
16 output tokens, temperature 0. `gpu-node-1` RTX 2070 SUPER, `brisk-thistle-9ae2`
GTX 1660 SUPER, ~226 ms apart.
**Commit:** `ca3c48a` · **Raw data:** `benchmarks/vdc/phase3_baseline.json`

---

## 1. Definitions used everywhere below

These are the only formulas this milestone uses. Every number in every later
document is one of these.

| Metric | Formula |
|---|---|
| **Aggregate TPS** | total output tokens across all requests ÷ wall-clock from first submit to last result |
| **Single-request TPS** | for each request, its output tokens ÷ its own generation time (excludes queueing) |
| **Inter-token latency (ITL)** | the runner's own per-step decode times; p50 and p95 reported |
| **TTFT** | request start to first generated token |
| **Prefill TPS** | total prompt tokens ÷ total prefill time |
| **Decode TPS** | `1 / ITL` — the steady-state rate once generation begins |
| **Queue wait** | wall time − sum of per-request generation times |

Aggregate and single-request differ by construction: aggregate includes the time
a request spent queued, single-request does not. **At concurrency 1 they differ
by exactly the queue wait**, which is what the Phase 2 "inconsistency" was.

## 2. Measured baseline

Three repeats per level; min/mean/max shown because the spread is large.

| conc | aggregate (mean) | runs | single-request | ITL p50 | ITL p95 | prefill TPS | queue wait |
|---|---|---|---|---|---|---|---|
| 1 | 6.50 | 7.65 / 4.20 / 7.64 | 21.4 | 40.2 ms | 40.3 ms | 55.5 | 1.35 s |
| 2 | 8.64 | 9.08 / 7.28 / 9.57 | 21.2 | 39.7 ms | 51.9 ms | 87.1 | 1.93 s |
| 4 | **14.97** | 12.63 / 20.04 / 12.25 | 22.3 | 39.8 ms | 52.1 ms | 85.7 | 2.37 s |

Baseline for the milestone's target: **14.97 aggregate TPS at concurrency 4**.
A 1.5× target is therefore **22.5 TPS**.

## 3. Two corrections to the Phase 2 numbers

**The baseline is higher than Phase 2's 12.74, and that is measurement, not
performance.** The harness polled for results once per second, which added up to a
second to every measured request. It now polls every 150 ms. The old number was
not wrong about the system; it was wrong about the clock.

**The concurrency-1 gap is explained, not mysterious.** Phase 2 reported 1.77
aggregate against 17.7 single-request at concurrency 1. Those measure different
things: 1.77 divides by a wall clock that includes ~5 s of pre-optimization queue
wait, 17.7 divides by the request's own 1.1 s of generation. With the work poll
now at 1 s, queue wait is 1.35 s and the two are 6.50 and 21.4.

## 4. What the numbers say before optimizing

**Decode is ~40 ms per token, which is 25 tok/s on a 0.5B model.** A 2070 SUPER
should complete a 0.5B forward pass in a few milliseconds. The GPU was sampled at
1–2 % utilisation during generation.

That is the dominant fact of this milestone: **the GPUs are idle; the per-token
path is CPU-bound.** Aggregate TPS is limited by how many sequences a machine can
drive through that path, not by GPU compute. This predicts that concurrency —
more replicas, or batching — will help far more than any kernel-level work, and
that prediction is testable.

**Prefill is faster than decode** (85 vs 25 tok/s) because one forward pass covers
many tokens, so the per-token overhead is amortised. That is the same mechanism
batching exploits.

## 5. Spread

Aggregate at concurrency 4 ranged 12.25–20.04 across three repeats. Two replicas
and four requests means the arrival pattern relative to the 1 s poll decides how
much queueing occurs, so a single run is not a measurement. All comparisons in
this milestone use the mean of three and report the spread.

## 6. Reproducing

```bash
python3 benchmarks/vdc/run_dual_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes gpu-node-1 brisk-thistle-9ae2 \
  --mode REPLICA --concurrency 1 2 4 --tokens 16 --repeats 3 \
  --out benchmarks/vdc/phase3_baseline.json
```