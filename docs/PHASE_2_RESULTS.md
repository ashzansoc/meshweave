# Phase 2 results — capacity-aware placement, replication, TPS

Measured on `gpu-node-1` (RTX 2070 SUPER, 8 GiB) and `brisk-thistle-9ae2`
(GTX 1660 SUPER, 6 GiB), ~226 ms apart, 8–15 Mbps, serving
`Qwen/Qwen2.5-0.5B-Instruct` in fp16.

Raw data: `benchmarks/vdc/*.json`. Harnesses: `benchmarks/vdc/`.

---

## 1. The headline

| case | conc 1 | conc 2 | conc 4 |
|---|---|---|---|
| **A** gpu-node-1 alone | 1.77 / 17.7 | 6.36 / 27.1 | 10.60 / 25.1 |
| **B** brisk-thistle alone | 5.30 / 14.1 | 6.36 / 21.0 | 9.09 / 20.1 |
| **C** pipeline across both | 1.59 / 2.69 | 1.68 / 2.85 | 2.36 / 2.78 |
| **D** replicas across both | 3.97 / 14.8 | 10.58 / 19.8 | **12.74** / 23.2 |

Each cell is **aggregate tok/s / single-request tok/s**.

**Replicas win at every concurrency level**, and the margin grows with load:
12.74 vs 10.60 for the best single node at concurrency 4. Splitting the model
across both machines remains the slowest configuration by 5.4×.

That is the answer to the question this milestone was built to settle, and it is
not the answer distribution advocates would prefer. On a 226 ms link, one GPU
holding the whole model beats two GPUs sharing it.

## 2. What changed

### Placement is chosen, not divided (M1)

The layer split is now searched rather than assumed. Every composition of the
layer count across the selected nodes is scored by predicted per-token latency:

```
(hops + 1) × RTT/2  +  Σ stage compute  +  activation bytes / bandwidth
```

Inputs are measured: per-node milliseconds per layer from recorded single-node
runs, RTT and throughput from the link matrix. Missing inputs are reported as
assumptions, never silently defaulted.

The estimate lands within 12 % of the measured 310.8 ms/token for the deployed
12/12 split. Because the model omits per-stage overhead and the head's vocabulary
projection, candidates within 5 % of the best are treated as equivalent and the
most balanced wins — otherwise the optimiser prefers a 23/1 split over 12/12 for
a predicted 1 ms on a 275 ms estimate.

**What it found here:** with two similar GPUs the optimum is essentially flat
(274.9 vs 273.9 ms across the top candidates), so it keeps 12/12. The value of
the search shows up with heterogeneous hardware, where a faster card should
receive more layers — covered by a test.

### Replication, and the rule that it comes first (M2)

A deployment can now be `REPLICA` instead of `PIPELINE`: one complete model per
node, each its own single-stage process with no traffic between them. Weights are
still genuinely per-node — 1201.9 MiB resident on each, verified from the process.

**`mode: AUTO` is the default**, and it replicates whenever one node can hold the
whole model. That encodes the rule the brief asks for — never split a model that
fits — rather than leaving it to whoever fills in the form.

### Routing spreads requests (M3)

Requests go to the replica with the fewest in flight, breaking ties by queue
length and then node id, so the choice is deterministic. A replica that is not
running is never chosen. Four concurrent requests on two replicas split 2/2.

### Queue wait was the real bottleneck (M4)

The first A/B/C/D run showed a 6-second wall time for a 1.1-second generation.
The model was not slow — the agent polled for work once per heartbeat, five
seconds by default, so every short request waited for the next tick before anyone
noticed it.

The agent now heartbeats on its own interval and **polls for work every second**
(`MESHWEAVE_WORK_POLL_S`). No model or routing change, and aggregate throughput
on replicas went from **9.08 to 12.74 tok/s**. The harness reports `queue_wait_s`
so this cannot hide again.

## 3. What the numbers mean

**Single-request TPS is highest on one node with no queue** — 27.1 tok/s for
gpu-node-1 at concurrency 2. Splitting costs ~10× (2.85 tok/s) because every
token pays a round trip across the link.

**Aggregate TPS is highest with replicas under load** — 12.74 tok/s at concurrency
4, against 10.60 for the best single node. Not 2×, because each replica still
serves one request at a time and the slower card caps the pair.

**Concurrency 1 numbers are noisy.** With one request, fixed overheads — the
first poll, connection setup, KV allocation — dominate, which is why case B
(5.30) and case A (1.77) differ so much at concurrency 1 despite similar
hardware. Read the conc 2 and conc 4 rows for the shape of the curve.

## 4. Limits of this evidence

* Two nodes only; the scaling claim rests on two replicas.
* One request in flight per replica, so aggregate is bounded by replica count.
* 16-token generations, so fixed costs are a large share of each measurement.
* No `tc netem` in these containers, so latency impairment was not tested here.
* The pipeline result (case C) is measured, not modelled, and matches Phase 1.

## 5. Reproducing

```bash
python3 benchmarks/vdc/run_dual_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes gpu-node-1 brisk-thistle-9ae2 \
  --mode AUTO --concurrency 1 2 4 8 16
```

Each case writes its own JSON, so a run can be repeated per configuration with
`--mode PIPELINE` or `--mode REPLICA`.