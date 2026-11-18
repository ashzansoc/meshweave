# Dual-metric benchmarks

Two questions that pull in opposite directions, kept apart.

**Harness:** `benchmarks/vdc/run_dual_benchmark.py`
**Results:** `benchmarks/vdc/A_single_node1.json`, `B_single_node2.json`,
`C_pipeline.json`, `D_replicas.json`

---

## 1. The two metrics

**Single-request TPS** — for each request, `output_tokens / its own total time`.
It measures how fast one generation completes, and it is what a user feels.

**Aggregate TPS** — `all output tokens / wall clock`, including the time requests
spent queued. It measures what the cluster delivers to many customers at once.

Merging them produces a number that flatters whichever configuration the author
prefers. On this hardware they disagree:

| | best for single-request | best for aggregate |
|---|---|---|
| one node, whole model | **27.1 tok/s** (conc 2) | 10.60 (conc 4) |
| two nodes, replicas | 23.2 | **12.74** |
| two nodes, split | 2.85 | 2.36 |

A single "TPS" column would hide that a split is 10× worse per request and 5×
worse in aggregate, or that replication buys aggregate at a small cost per
request.

## 2. What else is recorded

| Field | Meaning |
|---|---|
| `queue_wait_s` | wall time minus sum of generation times — the cost of being noticed and scheduled |
| `latency_p50_s`, `latency_p95_s` | end-to-end request latency, including queueing |
| `ttft_p50_ms` | time to first token |
| `ms_per_token_mean` | decode speed |
| `activation_bytes_per_token` | what the pipeline put on the wire |
| `succeeded` / `failed`, `errors` | a configuration that fails is reported, not omitted |

`queue_wait_s` earned its place immediately. The first sweep showed a 6-second
wall time for a 1.1-second generation: the agent polled for work once per
heartbeat. After decoupling the work poll (now 1 s) the same configuration reached
12.74 tok/s aggregate, with no change to the model, the placement or the routing.

## 3. The sweep

Concurrency 1, 2, 4, 8, 16 — the harness submits N prompts simultaneously behind a
one-second barrier so they genuinely overlap, then waits for all of them.

```bash
python3 benchmarks/vdc/run_dual_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes gpu-node-1 brisk-thistle-9ae2 \
  --mode AUTO --concurrency 1 2 4 8 16
```

`--mode` fixes the strategy for comparison: `PIPELINE` to force a split,
`REPLICA` to force copies, `AUTO` to let the coordinator choose.

## 4. Reading the curves

**Aggregate should rise with concurrency until the hardware saturates.** A flat
curve means a serialization point, not a hardware limit — which is exactly what
the queue-wait discovery was.

**Single-request should stay flat, then degrade** as queueing appears. When it
falls, the system is overloaded rather than slow.

**Concurrency 1 is noisy.** Fixed overheads dominate a single request, so cases
with near-identical hardware differ by 3× at conc 1 here (B 5.30, A 1.77). Read
conc 2 and conc 4 for the shape.

## 5. Comparability rules

* Same model, same prompt, same token budget, same temperature across cases.
* One case per deployment, stopped afterwards, so a previous run cannot lend
  throughput to the next.
* Each case writes its own JSON so a rerun replaces one case, not the set.
* Numbers here are **measured on the two nodes**, not projected from a model.
  Predictions live in `PHASE_2_RESULTS.md` only where labelled.

## 6. What is not measured

* Prefill and decode are not separated; single-request TPS includes prefill.
* GPU utilisation is sampled separately (1 Hz NVML in the agent), not per request.
* `tc netem` is unavailable in these containers, so no impaired-network sweep.
* Cost per million tokens is not computed here; node prices are unset.