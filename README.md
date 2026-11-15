<p align="center">
  <img src="docs/assets/logo.svg" alt="MESHWEAVE" width="380">
</p>

<h1 align="center">Many Machines. One Intelligence.</h1>

<p align="center">
  <strong>MESHWEAVE</strong> — Multi-Node Experimental System for Heterogeneous
  Workload Execution and Virtualized AI Execution
</p>

<p align="center">
  An experimental distributed AI compute fabric that combines independently
  rented GPUs into coordinated clusters, and measures honestly whether the
  result is worth anything.
</p>

---

## What this is

MESHWEAVE investigates one question:

> Can geographically distributed, heterogeneous, independently owned GPUs be
> coordinated into a reliable virtual AI computer capable of running models and
> workloads that no individual GPU can handle?

The answer is not assumed. It is measured.

The system starts from an uncomfortable fact that shapes the entire design: for
autoregressive decoding, pipeline parallelism across *N* hosts costs roughly one
network round trip per token per stage boundary, while the payload itself is
tiny — about 7 KiB per token for a 7B model. Bandwidth is not the constraint.
Latency is. Any architecture that ignores this will produce a system that runs
and is useless.

So MESHWEAVE separates the two things "distributed inference" is usually assumed
to mean:

| Path | What it buys | What it costs |
|---|---|---|
| **Replication** | Near-linear throughput scaling. Model replicas on independent GPUs; the network is off the hot path. | Does not make a larger model possible. |
| **Partitioning** | Makes a model larger than one GPU's memory runnable at all. | Per-token latency grows with every stage boundary. |

These are different products. Conflating them is the most common error in this
space, and MESHWEAVE treats them as separate benchmark tracks.

## Status

This is an active research project, not a product. What exists today:

- A working control plane: node registration, authenticated telemetry, a live
  resource registry, and a planner that can refuse a bad configuration.
- A real NVML agent that reports actual GPU telemetry and refuses to invent any.
- A network benchmark subsystem that measures the N×N link matrix before any
  model is loaded.
- A cost model that computes dollars per million tokens, not just tokens per
  second.
- A pipeline runtime with an activation transport that records every hop.
- A report validator that rejects benchmarks which omit failures, omit cost, or
  present simulated data as evidence.

What does **not** exist yet, because no GPU has been rented: a single measured
distributed inference result. Every experiment that needs hardware is present
and blocked, with the exact command that will run it.

## Quick start

```bash
# 1. Install
uv sync --extra dev

# 2. Prove the honesty guarantees hold (no GPU required)
uv run pytest tests/ -q          # 44 tests
uv run ruff check .
uv run mypy meshweave

# 3. Start a coordinator
uv run meshweave coordinator --port 8700

# 4. On each GPU host, start an agent
uv run meshweave certs --node-id node-a --out ./secrets
uv run meshweave agent --node-id node-a --coordinator https://<coordinator>:8700

# 5. Measure the network before anything else
uv run meshweave bench network --nodes host1:9000,host2:9000 --ids node-a,node-b

# 6. Ask the planner what it would do, and what it would cost
uv run meshweave plan --model Qwen2.5-32B-Instruct --compare
```

To run experiments against real GPUs you need a [Vast.ai](https://vast.ai) API
key:

```bash
export VAST_API_KEY=<your key>
uv run python -m experiments.exp011_distributed_vram --dry-run
```

Without a key, the experiment blocks and tells you exactly why and what to run.

## Dashboard

```bash
uv run meshweave coordinator --port 8700   # terminal 1
cd dashboard && npm install && npm run dev # terminal 2
```

Every numeric field in the dashboard is nullable and renders **not measured**
rather than zero. A chart of zeros would be a fabrication, so there isn't one.
Reliability shows as *unknown* until a node has been observed for ten heartbeats,
because a node that has never been seen is not 100% reliable — it is unknown.

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, components, data flow, trade-offs |
| [DECISIONS.md](docs/DECISIONS.md) | Every architectural decision, with its reasoning and what would change it |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | The experiment matrix and how to run each one |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Measurement methodology and results |
| [SECURITY.md](docs/SECURITY.md) | Threat model, authentication, what is deliberately not exposed |
| [ROADMAP.md](docs/ROADMAP.md) | Milestones and current position |

## Engineering rules

These are enforced by tests, not just documented:

1. No mock distributed system. `tests/test_experiment_guard.py` statically
   rejects any experiment importing synthetic telemetry.
2. No fabricated GPU discovery. The NVML path raises rather than inventing a
   device.
3. No fabricated benchmark results. The report validator requires measured
   values, and a measurement without a value must explain why.
4. No claiming aggregated VRAM without an execution that proves it.
5. No cost-free claims. Every configuration reports cost per token, or reports
   that cost is unknown and says what is missing.
6. No silently swallowed errors. Every failure path records a failure.

## License

Apache-2.0. This project is independent of Float16 and shares no code, branding,
dependencies or architectural assumptions with it.