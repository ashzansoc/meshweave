# Benchmarks A, B and C

One model, one prompt, one generation length, three configurations:

| Case | Where it runs |
|---|---|
| A | the whole model on node 1 |
| B | the whole model on node 2 |
| C | the model split across both |

```bash
python3 benchmarks/vdc/run_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes <node-1> <node-2>
```

Writes `results.json` (machine-readable) and `results.md` (the comparison).

Each configuration records its own tok/s, ms/token, TTFT, total time, the
activation bytes put on the wire, the pipeline round trip, and NVML samples
taken **during** the generation. The Markdown report prints the measured ratio
rather than describing distributed execution as an improvement.

What is not measured: `tc netem` latency injection, because these containers have
no privileges for it. Fault experiments that need it say so in their report.
