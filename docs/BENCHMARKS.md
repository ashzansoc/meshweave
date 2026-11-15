# Benchmarks

Measurement methodology, and what has actually been measured.

**The short version: MESHWEAVE has produced no GPU benchmark results.** No GPU has
been rented, no model has been executed across hosts, and no WAN link has been
characterised. What follows separates *verified behaviour of the software* (measured
in this repository, reproducible with `uv run pytest`) from *properties of the
hardware and network* (not yet measured).

Conflating the two is the exact failure this document exists to prevent.

---

## 1. What has been measured

### 1.1 The planner's arithmetic against real published model configs

The Qwen2.5 adapter's memory model is validated against the published
configurations, not against invented ones. Verified by `tests/test_model_adapter.py`:

| Model | bf16 weights | int4 weights | KV/token/layer (GQA) | Activation/token |
|---|---|---|---|---|
| Qwen2.5-1.5B | 2.88 GiB | 0.72 GiB | 512 B | 3,072 B |
| Qwen2.5-7B | 14.19 GiB | 3.55 GiB | 1,024 B | 7,168 B |
| Qwen2.5-14B | 27.51 GiB | 6.88 GiB | 2,048 B | 10,240 B |
| Qwen2.5-32B | 61.03 GiB | 15.26 GiB | 2,048 B | 10,240 B |

The 7B figure of 14.19 GiB corresponds to ~7.6 B parameters at bf16, which matches
the published model. That agreement is the check that the arithmetic is right.

**The number that matters most:** activation size is independent of layer count and
parameter count. 7 KiB per token for a 7B model, 10 KiB for a 32B. This is the
structural fact the whole latency argument rests on.

### 1.2 Partitioning behaviour

Verified by `tests/test_model_adapter.py`:

| Model | Capacities | Result |
|---|---|---|
| 7B bf16 | 2 × 22 GiB free | 14/14 layers, balanced |
| 7B bf16 | 1 × 22 GiB free | single stage (correct — it fits) |
| 32B bf16 | 1 × 22 GiB free | **rejected**: "short by 9.5 GiB" |
| 32B bf16 | 3 × 22 GiB free | 21/22/21 layers |
| 32B bf16 | 4 × 22 GiB free | 15/17/17/15 layers |
| 14B bf16 | 2 × 14 GiB free | 24/24 layers |

The rejection case is the important one: the planner refuses an impossible
configuration and states the deficit in GiB rather than producing a plan that would
fail at load time.

### 1.3 Planner decisions on a 4-node fleet

Synthetic topology (RTX 4090/4090/3090/4080 at Vast prices $0.35/0.29/0.22/0.19),
no network measurement, 20 tok/s/GPU assumed:

| Model | Strategy | Stages | Est. tok/s | Est. $/M output tokens |
|---|---|---|---|---|
| Qwen2.5-7B bf16 | SINGLE_GPU | — | 20.0 | 2.64 |
| Qwen2.5-14B bf16 | PIPELINE | 2 | 10.0 | 10.56 |
| Qwen2.5-32B bf16 | PIPELINE | 4 | 5.0 | 52.78 |

These are planner estimates from assumed per-GPU throughput, **not measurements**.
They demonstrate that the planner makes defensible choices and refuses to
distribute a model that fits; they do not demonstrate that a 7B model actually runs
at 20 tok/s on an RTX 4090.

### 1.4 Cost comparison (EXP-010, planning only)

Same synthetic fleet, planner estimates:

| Configuration | Est. tok/s | $/M output tokens |
|---|---|---|
| 1× single GPU | 20.0 | 4.86 |
| 2× replicated | 34.0 | 5.23 |
| 4× replicated | 68.0 | 4.29 |
| 2× pipeline | 10.0 | 17.78 |
| 4× pipeline | 5.0 | 58.33 |

**The finding:** replication is 4–11× cheaper per token than partitioning for a
model that fits on one GPU. This is the economically useful result, and it holds
under the planner's assumptions. Whether it holds on real hardware is what EXP-010
will determine.

### 1.5 Transport behaviour over real sockets

Verified by `tests/test_agent_e2e.py` — two agent servers on real ports, real HTTP:

- Stage registration succeeds and a duplicate returns 409.
- An activation traverses a two-stage chain and arrives byte-identical.
- A request with an unknown workload token is rejected with 401.
- `/agent/sample` without loaded weights returns **501**, not a fabricated token.

This proves the transport and stage protocol. It does **not** prove model
execution — the agents have no weights loaded, so the activation path echoes rather
than computes.

### 1.6 Honesty guarantees

Verified by `tests/test_experiment_guard.py` and `tests/test_report.py`:

- The synthetic telemetry module stamps `synthetic` into version, UUID and driver.
- An experiment importing synthetic telemetry fails a static AST test.
- The real NVML path raises on a host with no NVIDIA driver rather than inventing
  devices (verified on this Apple Silicon host:
  `NvmlUnavailableError: NVML Shared Library Not Found`).
- A report claiming completion with no measurements, no failure investigation, or no
  cost is rejected.
- A synthetic report is never counted as evidence, and cannot claim a real result.

Test suite: **44 tests, all passing.** `uv run pytest tests/ -q`.

---

## 2. What has not been measured

Stated explicitly, because a benchmark document that hides its gaps is worse than
no document.

| Claim | Status |
|---|---|
| Any model executes across two physical GPUs | **not measured** |
| Any model runs at any throughput on our hardware | **not measured** |
| Per-hop latency on any real link | **not measured** |
| Bandwidth between any two real hosts | **not measured** |
| NCCL over any real network | **not measured** |
| Any cost per token from an actual invoice | **not measured** |
| Node failure recovery time | **not measured** |
| Any training throughput (DDP/FSDP) | **not measured** |

Each is a blocked experiment with the unblocking command recorded in
`EXPERIMENTS.md`.

---

## 3. Methodology for when measurements exist

These are the rules the experiments will follow. They are written down now, before
any data exists, so that the methodology cannot be adjusted after seeing results.

### 3.1 Baseline first

Every distributed configuration is compared against a single-GPU baseline measured
in the same session, on the same model, at the same precision, with the same prompt
set. A distributed result without a same-session baseline is not reportable.

### 3.2 Identical workloads

All configurations use identical prompts, context lengths, precision, and
concurrency. Any deviation is recorded in the report's `config` block.

### 3.3 Concurrency sweep, not a single number

Throughput is reported at concurrency 1, 4, 16, 64, 128. A single tok/s figure
hides whether the result is latency-bound or throughput-bound, which is the whole
question.

### 3.4 Latency distributions, not means

Time to first token and inter-token latency are reported as p50/p95, not means. A
mean inter-token latency of 50 ms is compatible with a system that occasionally
stalls for 5 s, and that distinction determines whether the system is usable.

### 3.5 Per-hop bytes on the wire

Every pipeline hop records `bytes_sent`. The report states measured bytes per token
so the claim "the payload is tiny" can be checked against the wire rather than
against a formula.

### 3.6 Measured latency versus estimated latency

The runtime records both. Where they diverge, the report says so and treats the
divergence as a finding — this is the mechanism by which the `N × RTT` model gets
corrected by data (see D-002).

### 3.7 Failures are reported, not filtered

A failed request is recorded with its error. A report with zero failures must state
that failures were investigated. Reports that hide failures are rejected by the
validator.

### 3.8 Cost per token, or explicit unknown

Every configuration reports cost per million output tokens, or reports that cost is
unknown and lists what is missing. Never an estimate presented as a measurement.

---

## 4. Comparison targets

MESHWEAVE's results will be compared against:

| System | What it provides | Why it is a relevant comparison |
|---|---|---|
| Single GPU + vLLM | the throughput baseline | the thing distribution must beat |
| Single GPU + quantization | the capacity baseline | 4-bit 33B fits in 24 GiB — cheaper than any WAN cluster |
| Exo (Apache-2.0) | pipeline inference over heterogeneous devices | the closest prior art; pipeline-by-layer-range over TCP |
| Petals (Apache-2.0) | per-block inference over the internet | the extreme case: 1 RTT per block |

The expected shape of the result, stated in advance so it can be falsified:
quantization will beat distribution for every model up to ~33B on a single 24 GiB
card, and the WAN cluster will only win where the model does not fit at any price
the operator is willing to pay for a single GPU.

If the data shows otherwise, that is the finding.