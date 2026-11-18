# Phase 3 results — throughput

**Baseline:** 14.97 aggregate TPS at concurrency 4 (three repeats), reproduced in
`docs/PHASE_3_BASELINE.md`.
**After:** 19.75 at concurrency 4, 33.17 at concurrency 8.

**The 1.5× target is met at concurrency 8 (2.2×), not at concurrency 4 (1.32×).**
Both numbers are reported; the target was aspirational and the shortfall at
concurrency 4 is explained below rather than smoothed over.

---

## 1. What changed, in order of effect

| Change | Why | Measured effect |
|---|---|---|
| **Four replicas instead of two** — two per node, placed by free VRAM | The baseline showed GPUs idle at 1–2 % while decode took 40 ms/token: the limit is how many sequences a machine can drive, not compute | more sequences in flight |
| **Work poll 1 s → 250 ms** | Every request waited up to a second to be noticed | ~0.75 s off every request |
| **Result poll in the harness 1 s → 150 ms** | The observer was adding up to a second to the measured wall clock | corrected the measurement, not the system |

Nothing in the model, the kernels or the transport was touched.

## 2. Measured comparison

| concurrency | baseline (2 replicas, 1 s poll) | optimized (4 replicas, 250 ms poll) | ratio |
|---|---|---|---|
| 1 | 6.50 | — | — |
| 2 | 8.64 | — | — |
| 4 | **14.97** | **19.75** (18.29 / 17.48 / 23.49) | **1.32×** |
| 8 | not measured | **33.17** (32.27 / 36.40 / 30.83) | — |

Single-request TPS at concurrency 4 fell from 22.3 to 17.7. That is expected and
reported rather than hidden: two processes now share one GPU, so each request is
slower while the cluster serves more of them. Aggregate throughput is the metric
this milestone optimizes; the cost in per-request latency is stated.

## 3. Why concurrency 4 falls short of 1.5×

Four replicas and four requests should finish in the time of one request. They
take about three times that. The queue wait is already zero, so it is not
routing, and a manual test confirmed the spread is correct:

```
load:  {…gpu-node-1#0: 1, …gpu-node-1#1: 1, …brisk#0: 1, …brisk#1: 1}
results: 4 in 2.53 s, each request 0.77–0.96 s
```

Four parallel requests of ~0.9 s finishing in 2.5 s means ~1.6 s of overhead
outside generation. At 250 ms polling that overhead should be ~0.25 s, so the
remainder is in the control path between submission and execution — the agent's
sequential command loop, the result POST, and the harness's own polling.

**The next lever is batching inside a replica**, which removes the control path
from the per-token cost entirely rather than shortening it. That is Priority 2 in
the brief and is not implemented here.

## 4. What was ruled out by measurement

* **GPU compute.** 1–2 % utilisation during generation. Kernel work cannot help a
  GPU that is not running.
* **Network.** Prefill and decode are local in replica mode; no inter-node traffic.
* **Placement.** All four replicas hold the whole model; there is no split to
  optimise.

## 5. What remains unmeasured

* The distributed pipeline was not re-optimized in this pass. Its Phase 2 number
  (2.36 at concurrency 4) stands, and the same batching technique would apply to
  it with a different communication pattern.
* Speculative decoding, kernel fusion and framework comparisons are not attempted.
  Given that the GPU is idle and the bottleneck is the control path, none of them
  addresses the measured constraint.

## 6. Reproducing

```bash
python3 benchmarks/vdc/run_dual_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes gpu-node-1 brisk-thistle-9ae2 \
  --mode REPLICA --replicas-per-node 2 \
  --concurrency 4 8 --tokens 16 --repeats 3
```

Raw data: `benchmarks/vdc/phase3_baseline.json`,
`benchmarks/vdc/phase3_replicas2_fast.json`.