# Current architecture audit

Written against the code and the recorded measurements, not against the roadmap.
Every claim below is tagged **[verified]** (executed and measured in this
repository), **[implemented]** (present in source, not independently measured),
or **[gap]**.

Date: 2026-11-18 · 229 tests passing · 28 documents · benchmarks in
`benchmarks/vdc/results.json`.

---

## 1. What already works

### 1.1 Distributed inference across two real machines **[verified]**

Qwen2.5-0.5B-Instruct, 24 layers, split `[0,12)` on an RTX 2070 SUPER and
`[12,24)` on a GTX 1660 SUPER. Each stage held 0.587 GiB; a single-node run held
1.174 GiB and produced **byte-identical text**. Both GPUs showed activity in the
same one-second NVML sample window.

| Configuration | tok/s | ms/token | TTFT |
|---|---|---|---|
| single node 1 | 19.02 | 38.8 | 368 ms |
| single node 2 | 16.67 | 41.0 | 497 ms |
| split across both | 2.93 | 310.8 | 1053 ms |

The per-stage resident memory is reported from the process, not estimated from
parameter counts, and the layer ranges are checked so every layer runs exactly
once.

### 1.2 Numerical correctness **[verified]**

`meshweave/runtime/dense.py` matches `transformers` to 1.5e-07 in fp32 for a
single stage, a three-way split and a decode step (`tools/check_dense_parity.py`),
and to 8.4e-05 against the real 0.5B checkpoint. Two processes over TCP produce
identical tokens.

Two config traps were found and fixed this way: `rope_theta` moving under
`rope_parameters` in transformers v5, and a missing `attention_bias` key that
caused attention to be computed without the checkpoint's q/k/v biases.

### 1.4 Node identity and credentials **[verified]**

Node ids are generated once (`silver-heron-3f9c`) and persisted, so a restart
does not create a second node. Registration returns a signed node token; the work
poll requires it; the stage handshake requires a per-deployment token. The agent
is outbound-only and has no shell endpoint — its command set is
`run_stage`, `stop_stage`, `run_prompt`, `run_echo`, `probe_peer`.

### 1.5 Link measurement **[verified]**

Both directions between the two hosts: 226–228 ms RTT, 8–15 Mbps, 0/10 failed
round trips. Orchestrated through the work queue because nothing dials in.

### 1.5 Cluster formation **[verified]**

`F16-VDC-001` groups members, validates them, and reports capacity as *aggregate
distributed* with an explicit note that it is not one addressable pool.

### 1.6 Placement refusals **[verified]**

A deployment is refused, with the node and the shortfall named, when a node has
no GPU, too little free VRAM for its layer range, too little disk for the
checkpoint, no published TCP port, or an agent too old to poll.

### 1.5 Benchmark harness **[verified]**

A/B/C with identical prompts and lengths, writing JSON and a Markdown comparison
that prints the measured ratio rather than describing distribution as faster.

---

## 2. What is incomplete

### 2.1 Placement is an even split **[gap] — the largest gap**

`partition_layers()` divides layers equally and front-loads the remainder. It
ignores GPU speed, free VRAM, memory bandwidth, temperature, and measured link
quality. A 2070 SUPER and a 1660 SUPER get the same 12 layers despite different
throughput. This is the component the brief most directly contradicts.

### 2.2 No replication path **[gap]**

Selecting two nodes always splits the model. A 0.5B model that fits comfortably
on either GPU is split anyway, paying 6.5× latency for nothing. There is no
"run N independent replicas and route" mode.

### 2.1 One request in flight **[gap]**

Rank 0 serialises generation on a lock; a second request gets 429. No continuous
batching, no prefix caching, no KV reuse across requests.

### 2.1 No streaming **[gap]**

The gateway returns one response. There is no SSE path.

### 2.1 No mid-generation recovery **[gap]**

A stage that dies takes the request with it. Re-partitioning a running
deployment is not implemented.

### 2.1 Single architecture family **[gap, by decision]**

Qwen2 only. Other architectures are refused by name rather than run with
approximate maths.

### 2.1 Operational state **[gap]**

Node and deployment state persist as JSON snapshots; workload state is in memory.
Fine for two nodes, not for a fleet.

---

## 3. Existing bottlenecks, with numbers

**Latency, not bandwidth.** Distributed decode is 310.8 ms/token; the pipeline
round trip is 309.8 ms against a 226–228 ms RTT. The activation payload is 1.8
KiB per decode step — at 8–15 Mbps that transfer is a small fraction of the
round trip. Adding bandwidth would not help; reducing round trips would.

**Splitting hurts the common case.** A 0.5B model fits either GPU with room to
spare, so distribution is pure overhead for it: 19.02 tok/s becomes 2.93.

**Compute is a few percent of a distributed token.** Consistent with the earlier
three-host Phi-MoE phases (compute ≈2.6 %), where direct token return took decode
from 2,663 to 453 ms/token.

**Serialisation and concurrency.** One request per deployment means a second
customer waits, even though each GPU is mostly idle between tokens.

---

## 4. Abstractions worth reusing

| Abstraction | Where | Why keep it |
|---|---|---|
| Agent work queue | `meshweave/coordinator/app.py`, `deploy/agent.py` | Closed command set, pull-based, no inbound port |
| Framing | `meshweave/runtime/framing.py` | Length-prefixed, CRC-checked, 65 lines |
| Geometry from config | `meshweave/runtime/model_config.py` | Reads the model's own config; sizes memory without loading weights |
| Refusal model | `plan_deployment` | Names the node and the shortfall instead of half-starting |
| Process ownership | `stop_stage`, `_pid_cmdline` | Never signals a pid it did not record |
| Benchmark harness | `benchmarks/vdc/run_benchmark.py` | Same prompt and length across configurations |
| Evidence discipline | `Measured` / "not measured" in the dashboard | Never renders a zero as a measurement |

---

## 5. Components to replace or rework

| Component | Why | Replacement direction |
|---|---|---|
| `partition_layers` | Equal split ignores hardware and links | Capacity- and latency-aware placement, evaluated per candidate split |
| Deployment shape | Always splits; no replicas | Add a replication strategy with routing |
| Single-request constraint | Serialises customers | Continuous batching, or N replicas behind a router |
| Workload state in memory | Lost on restart, invisible across processes | Persist, or accept and document |
| JSON snapshot registry | Whole-file rewrite per heartbeat | Database or append-only log |

## 6. Components that must not change

* `meshweave/runtime/dense.py` — parity-verified against two reference paths.
* `deploy/stage_runner.py` transport — direct token return is what removes a
  round trip per token.
* The process-ownership rules in the agent and installer. They exist because an
  earlier cleanup routine killed `sshd` on three live hosts.
* The refusal discipline in planning and the "not measured" rendering in the
  dashboard.

---

## 7. Where this leaves the Float16 brief

The brief asks for intelligent, topology-aware placement across heterogeneous
GPUs, with the ability to choose between distributing a model and replicating it.
The infrastructure to do that exists and is measured. The **decision layer does
not**: placement is a fixed even split, and there is no replication path.

That is the next milestone's work, and it is well-bounded: the geometry, the
link measurements, the per-stage memory accounting and the benchmark harness are
all already real.

The three external frameworks are assessed separately in
`docs/FRAMEWORK_COMPATIBILITY_MATRIX.md`. In short: DeepSpeed and Dynamo are
datacenter technologies and do not fit this environment; Parallax is
architecturally aligned but unproven on Turing, and gets a contained spike.