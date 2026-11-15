# MESHWEAVE Architecture

Status: P0 complete. Describes the system as implemented, not as aspiration.
Where a component is a stub, this document says so.

---

## 1. The constraint that shapes everything

MESHWEAVE's design follows from one measured fact about transformer inference.

For autoregressive decoding, a pipeline of *S* stages adds approximately *S − 1*
network round trips to **every generated token**, because the token cannot be
computed by stage *n+1* until stage *n* has produced its hidden state. The
payload per hop is one hidden-state tensor:

| Model | Hidden size | Bytes per token per hop (fp16) |
|---|---|---|
| Qwen2.5-1.5B | 1536 | 3,072 |
| Qwen2.5-7B | 3584 | 7,168 |
| Qwen2.5-14B | 5120 | 10,240 |
| Qwen2.5-32B | 5120 | 10,240 |

7 KiB at 100 Mbps is 0.57 ms of transmission. The round trip is 40–100 ms. **The
payload is noise; the latency is the cost.** This is why MESHWEAVE:

- measures per-hop latency as a first-class metric rather than assuming it,
- records bytes-per-token on the wire so the claim can be checked,
- treats pipeline parallelism as a **capacity** mechanism, not a throughput one,
- excludes tensor parallelism from the WAN path entirely (2 collectives per layer
  per token = 64 blocking collectives per token for a 32-layer model),
- refuses prefill/decode disaggregation over the internet (the KV cache for a 4k
  prompt on a 7B model is ~2 GB, which is ~160 s at 100 Mbps).

Every one of these is an *estimate until measured*. The runtime instruments the
real numbers and the experiment reports them.

---

## 2. Component overview

```
                        ┌─────────────────────────────┐
                        │       COORDINATOR           │
                        │  FastAPI control plane      │
                        │  registry · planner ·       │
                        │  workload mgmt · topology   │
                        └──────────┬──────────────────┘
                                   │ mTLS, workload-scoped tokens
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
       │   AGENT     │      │   AGENT     │      │   AGENT     │
       │  NVML       │      │  NVML       │      │  NVML       │
       │  control API│      │  control API│      │  control API│
       │  echo :9000 │      │             │      │             │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              │                    │                    │
         RTX 4090             RTX 3090             RTX 4080
```

Two distinct network paths, deliberately kept separate:

**Control plane** — HTTP over TLS between agents and the coordinator. Carries
registration, heartbeats, telemetry, assignments and metrics. Latency-tolerant.

**Data plane** — activation frames between adjacent pipeline stages. Latency-bound.
Uses a compact binary framing (8-byte header, JSON metadata, raw tensor bytes)
rather than JSON-encoding the tensor, because a 7 KiB activation JSON-encoded is
~30 KiB and that is 4× the bytes on a link whose cost is already dominated by RTT.

---

## 3. Coordinator

A modular monolith. Deliberately not microservices: the failure modes of a
distributed control plane would multiply before the execution path is proven.

**Endpoints** (`meshweave/coordinator/app.py`):

| Route | Purpose |
|---|---|
| `POST /v1/nodes/register` | agent first contact |
| `POST /v1/nodes/{id}/heartbeat` | telemetry + liveness |
| `GET /v1/nodes` | registry snapshot with aggregates |
| `POST /v1/topology` | submit a measured N×N matrix |
| `POST /v1/plan` | produce one plan |
| `POST /v1/plan/compare` | cost comparison across configurations |
| `POST /v1/workloads` | plan + dispatch |
| `GET /v1/metrics` | cluster metrics for the dashboard |

**Design decision:** the coordinator does not hold model weights and does not
talk to GPUs. It reasons about *reported* capacity. If a node lies about its
VRAM, the plan is wrong — which is why the registry records per-node reliability
and why the planner treats unverified nodes as untrusted.

**No single point of failure yet.** The registry is in-process and the state is
in SQLite. A coordinator restart loses in-flight workload state. This is
acknowledged in `ROADMAP.md` as P10 work; making it distributed before proving
inference works would be premature.

---

## 4. Agent

Runs on every GPU host. Deliberately small: it knows about *stages*, not about
*weights*.

**Telemetry** (`meshweave/agent/telemetry.py`) — real NVML. Rules enforced in
code:

- No NVML → `NvmlUnavailableError`, never a synthetic device.
- A counter NVML does not support → `None`, never zero.
- Simulation lives in a separate module that `telemetry.py` never imports.

Verified on a non-NVIDIA host: raises `NvmlUnavailableError: NVML Shared Library
Not Found` rather than returning plausible numbers.

**Control API** (`meshweave/agent/api.py`) — a closed set:

| Route | Behaviour without weights loaded |
|---|---|
| `/agent/ping` | liveness + inventory summary |
| `/agent/assign` | register a stage; 409 if already loaded |
| `/agent/activation` | echo the activation (transport test) |
| `/agent/sample` | **501**, not a fabricated token |
| `/agent/memory` | per-stage VRAM |
| `/agent/cancel` | drop a workload |
| `/agent/drain` | stop accepting work |

`/agent/sample` returning 501 without weights is the important one. An agent that
invented a token would silently convert a transport test into a fake inference
result.

**Echo endpoint** (`meshweave/bench/echo.py`) — the only non-HTTP listener. It
echoes bytes and does nothing else. It cannot execute anything, so it needs no
authentication: reaching it only lets you make the host echo bytes you sent.

---

## 5. Resource registry

Holds live node records. Reliability is **earned**, not configured: `None` until
10 heartbeats have been observed, then `1 − missed/(observations + missed)`.

A node that has never been seen is not "100% reliable" — it is unknown. This
matters because a planner that trusts an unknown node will place a model on a
host that is about to disappear.

---

## 6. Topology engine

Assigns an environment to each link from **measured** latency:

| Environment | Measured RTT (p50) | Permitted strategies |
|---|---|---|
| A_DATACENTER | ≤ 2 ms | tensor, pipeline, replicated |
| B_REGIONAL | ≤ 20 ms | pipeline, replicated |
| C_INTERNET | > 20 ms | pipeline, replicated |

A node's environment is the *worst* of its links, so one distant peer prevents
treating the node as datacenter-grade.

Tensor parallelism requires `require_datacenter()` to pass: measured A_DATACENTER
links **and** sustained bandwidth above 1 Gbps. Over ordinary internet this
returns a rejection reason, which the planner surfaces rather than hiding.

---

## 7. Execution Intelligence (planner)

Deterministic and auditable. Answers nine questions:

1. One GPU or several? 2. Replicate or partition? 3. Which nodes form the group?
4. Which strategy? 5. How are layers distributed? 6. Expected latency?
7. Expected cost? 8. What happens if a node disconnects? 9. Why?

**It can say no.** A plan that would force distribution on a model that fits on
one GPU is a defect, not a conservative default. The planner enumerates every
candidate strategy, costs each, and selects the cheapest — including when the
cheapest answer is "use one GPU".

**Measured on a 4-node fleet (RTX 4090/4090/3090/4080 at real Vast prices):**

| Model | Strategy | Stages | Est. tok/s | $/M output tokens |
|---|---|---|---|---|
| Qwen2.5-7B bf16 | single GPU | — | 20.0 | 2.64 |
| Qwen2.5-14B bf16 | pipeline | 2 | 10.0 | 10.56 |
| Qwen2.5-32B bf16 | pipeline | 4 | 5.0 | 52.78 |

The 7B model stays on one GPU because that is correct. The 32B model distributes
because nothing else fits, and the planner states plainly that it will be 10×
more expensive per token than the single-GPU baseline.

**The crossover detector** reports where adding nodes stops reducing cost per
token. That is the economically useful number: it tells you when to stop renting.

**A correction worth recording.** An early version scaled replicated throughput
perfectly linearly, which meant replication always looked cheapest and the
comparison was meaningless. Real replication has an efficiency factor (0.85) for
router queueing and imperfect batching, and the planner now models it. With
uniform prices replication still improves cost per token; with dispersed prices
it does not. Both behaviours are covered by tests.

---

## 8. Memory fabric

**Logical aggregation, not a virtual address space.** The objective stated in the
brief is explicit about this, and MESHWEAVE honours it.

The fabric tracks model weights, shards, activations, KV cache and reservations,
and distinguishes:

- **aggregate physical VRAM** — arithmetic over GPUs, which proves nothing
- **usable model capacity** — what actually fits after KV headroom and the
  embedding and LM head are placed

The partitioner refuses to plan a model that does not fit and explains the
deficit in GiB. A 32B model at bf16 on one 24 GiB card raises:

> model needs 58.1 GiB of weights + 0.2 GiB KV headroom across stages, but only
> 48.6 GiB is available (short by 9.5 GiB). Use a smaller model, a lower
> precision, or fewer KV tokens.

---

## 9. Runtime

`PipelineRuntime` drives an ordered chain of stages:

1. **Preflight** — probe every hop, record the real per-hop latency, and compare
   it with the planner's estimate. A disagreement is a finding.
2. **Prefill** — push the prompt down the chain once.
3. **Decode** — for each token: final stage samples, then the token flows back to
   stage 0 and the chain runs again.
4. **Record** — every hop's bytes, latency and stage go into `StageTransfer`.

There is no tensor parallelism, no expert routing, no microbatch scheduler. Those
are experiments, not architecture. The runtime is deliberately small enough to
read in one sitting.

**Known limitation, stated plainly:** if a middle stage dies, its KV cache dies
with it. Those layers' hidden states cannot be reconstructed from later stages.
Exact continuation is impossible and the request must restart from the prompt.
`meshweave/fault/recovery.py` documents this rather than pretending otherwise.

---

## 10. Cost model

Single source of truth (`meshweave/bench/economics.py`) so the planner, the
dashboard and every report agree.

Refuses to guess a price: unknown hourly cost propagates as `None` and is
reported as an unknown component rather than a plausible number. Cost per token
is computed only from measured throughput and known cost.

---

## 11. Security posture

Full detail in [SECURITY.md](SECURITY.md). Summary:

- Every agent has a certificate; the coordinator binds a TLS connection to a
  registered node identity.
- Every workload gets a short-lived token scoped to one workload on one node.
- No shell execution, no file upload, no arbitrary code path through the
  coordinator. The agent's control API is a closed enum.
- The echo endpoint can only echo.
- Every independently hosted GPU node is treated as potentially untrusted, as the
  brief requires.

**Computational correctness and data confidentiality are different properties.**
MESHWEAVE can verify a model produced correct output while having no way to
guarantee the node did not inspect the weights it was given. That distinction is
documented, not papered over.

---

## 12. What is deliberately not here

- **Kubernetes.** No container orchestration. A research cluster of 2–8 nodes
  does not need it, and adding it would multiply failure modes before the
  execution path is proven.
- **Ray.** Deferred to P10. Ray provides orchestration, not GPU communication;
  adopting it before the planner works would obscure which layer is responsible
  for what.
- **DDP/FSDP/DeepSpeed.** Deferred to P9. Training has different communication
  patterns from inference and must not be conflated with it.
- **General model support.** One architecture (Qwen2.5) behind an adapter
  interface. Supporting every Hugging Face model would trade a working experiment
  for an untestable abstraction.