# Experiments

The experiment matrix, what each one measures, and the exact command to run it.

**Current status: no experiment requiring a GPU has been run.** No Vast.ai
credentials are configured in this environment, and no GPU has been rented. Every
experiment below is implemented and will run once hardware is available. Where an
experiment is blocked, it says so and names the unblocking command — it does not
report a number it did not measure.

---

## How to run an experiment

```bash
# Always dry-run first: checks prerequisites without spending money
uv run python -m experiments.exp011_distributed_vram --dry-run

# Then the real run
uv run python -m experiments.exp011_distributed_vram
```

Every experiment writes a JSON report to `reports/raw/` and prints a summary. The
report is validated before it is written; a report that omits failures, omits
cost, or claims a result without evidence is rejected rather than published.

---

## Matrix

| ID | Experiment | Requires | Status |
|---|---|---|---|
| EXP-001 | Single-GPU baseline | 1 GPU + vLLM | implemented, blocked |
| EXP-002 | Two nodes, same network | 2 probe endpoints | implemented, blocked |
| EXP-003 | Two independent Vast hosts | 2 Vast instances | implemented, blocked |
| EXP-004 | Four independent GPUs | 4 Vast instances | implemented, blocked |
| EXP-005 | Eight independent GPUs | 8 Vast instances | implemented, blocked |
| EXP-006 | Pipeline versus tensor parallelism | measured topology | not started |
| EXP-007 | DDP versus FSDP | 2+ GPUs, training harness | not started (P9) |
| EXP-008 | Node failure | 2+ GPUs, fault harness | not started (P8) |
| EXP-009 | Dynamic node addition | 2+ GPUs | not started (P10) |
| EXP-010 | Replication versus partitioning | 2+ nodes + cost data | implemented, blocked |
| EXP-011 | Distributed VRAM capacity validation | 2 GPUs | implemented, blocked |
| EXP-012 | Cost optimisation | all of the above | not started |

---

## EXP-011: Distributed VRAM Capacity Validation

**The project's first major technical milestone.** This is the experiment that
directly tests the central hypothesis.

```bash
uv run python -m experiments.exp011_distributed_vram --dry-run
uv run python -m experiments.exp011_distributed_vram \
    --model Qwen2.5-32B-Instruct --coordinator http://127.0.0.1:8700
```

**Hypothesis.** A model whose weights exceed one GPU's memory can execute
correctly when its layers are partitioned across two independently hosted GPUs,
with real activations crossing the network.

**Procedure.**

1. Confirm the coordinator is reachable and at least two nodes report healthy
   telemetry.
2. Ask the planner for a plan. If it returns `SINGLE_GPU`, the experiment is
   blocked: the chosen model fits on one GPU, so distribution cannot be
   demonstrated. The error names a larger model to try.
3. Partition by layer range and load each stage on its own host.
4. Record per-GPU VRAM before and after load, on each host.
5. Send a prompt through the chain, transferring real activations.
6. Verify the output is valid text produced by the model.
7. Record per-stage memory and per-hop transfer cost.

**Success criterion.** The model genuinely executes across both physical GPUs,
with weights resident on each, real activation transfers, verified output, and
memory recorded on both.

**Explicitly not sufficient.** Reporting aggregate VRAM. "24 + 24 = 48 GB" is
arithmetic, not evidence. The experiment records `aggregate_vram_gib` with a note
saying exactly that, and the finding that counts is per-stage measured VRAM.

**What would make it fail honestly.** If the planner selects SINGLE_GPU, or if a
stage fails to load, the experiment is blocked with the reason — not reported as
a partial success.

---

## EXP-001: Single-GPU Baseline

```bash
uv run python -m experiments.exp001_single_gpu
```

Every other experiment is judged against this one. If a distributed configuration
cannot beat the single-GPU baseline on the metric being claimed, the claim is not
supported.

Measures tokens/sec, wall time and error count across a concurrency sweep
(1, 4, 16, 64) against a local vLLM server. Refuses to run against synthetic
telemetry, because a synthetic baseline would silently become the reference for
every subsequent comparison.

---

## EXP-002 through EXP-005: Network scaling

```bash
uv run python -m experiments.exp002_network_scaling exp002 \
    --nodes host1:9000,host2:9000 --ids node-a,node-b
uv run python -m experiments.exp002_network_scaling exp003 \
    --nodes vast1:9000,vast2:9000 --ids node-a,node-b
```

One harness, four node counts. Each measures the N×N link matrix first — no model
is loaded until the links are characterised, because "the network was slow" is
only a finding if it was measured.

| ID | Nodes | What it establishes |
|---|---|---|
| EXP-002 | 2, same network | the upper bound for what distribution can achieve |
| EXP-003 | 2, separate hosts | what geographic separation costs |
| EXP-004 | 4, independent | whether scaling holds |
| EXP-005 | 8, independent | where it breaks |

**Measurements.** RTT p50/min/max, jitter, packet loss, single- and multi-stream
TCP throughput, sustained throughput, connection stability, UDP support. Each pair
is probed in both directions (8 nodes → 28 pairs → 56 directed measurements),
because residential uplinks are frequently asymmetric and a single measurement per
pair would hide it.

---

## EXP-010: Replication versus Partitioning

**The question the project turns on.**

```bash
uv run python -m experiments.exp010_replication_vs_partition \
    --coordinator http://127.0.0.1:8700 --model Qwen2.5-7B-Instruct
```

Builds the full cost comparison across 1×, 2×, 4× and 8× replication and
partitioning, reports the cheapest configuration by cost per million output
tokens, and locates the crossover where adding nodes stops reducing cost per
token.

**Sample output from a planning-only run** (synthetic topology, no GPU
measurements — illustrative of the report format, **not** a benchmark result):

```
tps::1x single GPU          20.000 tok/s   $4.86/M tokens
tps::2x replicated          34.000 tok/s   $5.23/M tokens
tps::4x replicated          68.000 tok/s   $4.29/M tokens
tps::2x pipeline            10.000 tok/s  $17.78/M tokens
tps::4x pipeline             5.000 tok/s  $58.33/M tokens

cheapest: 4x replicated at $4.29
crossover at 1 nodes: adding more stops reducing cost per million tokens
replication is cheaper per token than partitioning ($5.23 vs $17.78)
```

The finding this experiment exists to produce: **replication is 4–11× cheaper per
token than partitioning for a model that fits on one GPU.** Partitioning is only
justified when the model does not fit at all, and even then it is slow.

---

## EXP-006: Pipeline versus Tensor Parallelism

Not started. Requires the measured topology from EXP-002..005. Tensor parallelism
is expected to be refused on every internet-class link (see D-004); the experiment
exists to demonstrate the refusal with data rather than to make TP work.

---

## EXP-008: Node Failure

Not started (P8). The recovery semantics are implemented and tested
(`meshweave/fault/recovery.py`) but not exercised against live hardware.

The honest expectation, documented in advance: for a pipeline, losing any stage
fails the in-flight request, because the KV cache for those layers lived in the
dead process and cannot be reconstructed from later stages. Recovery means
rebuilding the stage and replaying from the prompt — exact continuation is
impossible. The experiment will measure how long that takes and how many requests
are lost.

---

## What has not been measured

Stated plainly, because the absence of a result is itself information:

- **No distributed inference has been executed.** Not once. The runtime is tested
  over real HTTP sockets against agents with no weights loaded, which proves the
  transport and the stage protocol but not model execution.
- **No GPU has been benchmarked.** The NVML path is verified to fail honestly on a
  non-NVIDIA host; it has not been exercised against real hardware.
- **No network measurement has been taken between real hosts.** The probe code runs
  and the echo endpoint works, but no WAN link has been characterised.
- **No cost figure is measured.** Every cost number in this repository derives
  from the Vast price fields the user supplies, not from an observed invoice.

Each of these is a blocked experiment with a command that will clear it.