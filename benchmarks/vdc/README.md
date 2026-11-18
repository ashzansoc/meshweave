# VDC benchmarks

Two harnesses, two questions.

| Harness | Measures | Cases |
|---|---|---|
| `run_benchmark.py` | tokens/s, ms/token, TTFT for one prompt | A (one node), B (other node), C (split) |
| `run_dual_benchmark.py` | **single-request TPS** and **aggregate TPS** separately, across a concurrency sweep | any mode, 1/2/4/8/16 |

```bash
# A/B/C: one prompt, three configurations
python3 benchmarks/vdc/run_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes <node-1> <node-2>

# A/B/C/D with a concurrency sweep, metrics kept apart
python3 benchmarks/vdc/run_dual_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes <node-1> <node-2> \
  --mode AUTO --concurrency 1 2 4 8 16
```

`--mode` selects `AUTO` (replicate if the model fits one node), `PIPELINE` (split it),
or `REPLICA` (a full copy per node).

## Why two metrics

They move in opposite directions and merging them hides the trade-off:

- **single-request TPS** — how fast one generation completes. Improves when you
  give a request a whole GPU and never split the model.
- **aggregate TPS** — all tokens produced divided by wall time, including
    queueing. Improves when you add replicas and spread requests.

`queue_wait_s` is reported beside them, because on a polled control plane the wait
to be noticed can exceed the generation itself. It did: a 5-second work poll added
~5 s to a ~1 s request until the agent began polling every second.

## Results

See `results.md` and `dual_results.md`, and the raw JSON in this directory.
