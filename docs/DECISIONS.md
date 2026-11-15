# Decisions

Every architectural decision, why it was made, and what would change it. Newest
first. A decision is only recorded here once it is implemented and tested.

---

## D-001: Separate replication from partitioning as distinct benchmark tracks

**Decision.** MESHWEAVE has two execution paths with separate benchmarks and
separate success criteria.

**Why.** "Distributed inference" is used to mean two incompatible things.
Replication (N copies, requests spread) scales throughput and keeps the network
off the hot path. Partitioning (one model split) buys capacity and pays network
latency on every token. Treating them as one feature produces a system that is
bad at both and metrics that flatter neither.

The measured consequence on a 4-node fleet at real Vast prices: replication
delivers $4.29–5.23 per million output tokens; partitioning of the same model
delivers $17.78–58.33. A single "distributed inference" number would have hidden
a 4–11× difference.

**What would change this.** Nothing. If a workload needs a model that fits in one
GPU, partitioning is the wrong tool regardless of how the benchmark is labelled.

---

## D-002: Measure per-hop latency rather than assuming it

**Decision.** The runtime probes every pipeline hop during preflight and records
the measured value. The planner's `DEFAULT_PER_HOP_RTT_MS = 50.0` is an estimate
used only when no measurement exists, and plans annotate it as such.

**Why.** The initial design treated `N × RTT` as a fixed cost model. That was too
crude. A persistent connection with pipelining does not necessarily pay a full
RTT per transfer, and the real figure depends on the protocol, the scheduling, and
whether the send is synchronous. Hardcoding the assumption would have made the
central claim untestable.

The correction matters: the runtime now records both `send_ms` (time to
acknowledged receipt) and `bytes_sent` per hop, so an experiment can compare the
measured latency against the estimate and report the discrepancy rather than
burying it.

**What would change this.** A measured, reproducible model of per-hop cost that
accounts for pipelining, connection reuse, and batching — derived from data across
several topologies, not assumed.

---

## D-003: Build a custom activation-passing runtime instead of adopting Exo or Petals

**Decision.** MESHWEAVE implements its own pipeline runtime.

**Why.** The purpose of the project is to understand the underlying engineering,
not to configure someone else's. Exo (Apache-2.0) is pipeline-by-layer-range over
plain TCP with mDNS/static-config discovery and capability-aware partitioning; it
is an excellent architecture reference and a poor dependency for an experiment
whose deliverable is understanding. Petals (Apache-2.0) is per-block placement
over a Kademlia DHT with ~1 RTT per block — a research artefact, largely dormant,
that cannot be instrumented for the questions MESHWEAVE asks.

Both are studied and cited in `BENCHMARKS.md` for comparison. Neither is forked.

**Scope limit.** The runtime deliberately does not attempt to recreate vLLM or
PyTorch Distributed. It partitions, sends activations, schedules stages, relays
tokens, and handles failure. That is the whole surface.

**What would change this.** A measured demonstration that our runtime's
limitations, not the network's, are the binding constraint — and that an existing
implementation removes them.

---

## D-004: Exclude tensor parallelism from the WAN path

**Decision.** TP is gated behind `require_datacenter()`, which requires measured
A_DATACENTER links *and* sustained bandwidth above 1 Gbps. Over ordinary internet
it is refused with a stated reason.

**Why.** Megatron-style TP performs 2 all-reduces per transformer layer per token.
For a 32-layer model that is 64 blocking collectives per generated token, each
costing roughly one round trip for a message this small (8 KB). At 50 ms RTT that
is ~3.2 s/token, before any compute. The arithmetic is unforgiving and does not
improve with faster GPUs.

Also: NCCL has no UDP data transport (its socket transport is `SOCK_STREAM`), and
it cannot hole-punch NAT. It works between machines on an overlay network with
`NCCL_SOCKET_IFNAME` set and all ports open between the private IPs. That is a
datacenter configuration, not an internet one.

**What would change this.** Measured evidence of a WAN topology where per-collective
latency is low enough for TP to be viable — which would mean RTT in the low single
digits of milliseconds, at which point the link is effectively datacenter-grade.

---

## D-005: Refuse prefill/decode disaggregation over the internet

**Decision.** The runtime keeps prefill and decode on the same node.

**Why.** Disaggregation (DistServe, Mooncake) requires moving the KV cache between
nodes. For a 7B model at MHA with a 4k prompt, that is ~2 GB per request. At
100 Mbps it is ~160 s. Even with GQA (Qwen2.5-7B uses 4 KV heads), the transfer
scales with context length, not with model size, and it lands on the critical path
before the first output token.

Disaggregation is a datacenter technique. Over the internet it is a pessimisation.

**What would change this.** A workload where prefill and decode have wildly
different resource profiles *and* the KV cache is small enough that transfer cost
is negligible relative to compute.

---

## D-006: The planner must be able to refuse

**Decision.** `ExecutionPlanner` enumerates candidates, costs each, and selects the
cheapest — including when that is "use one GPU". An unviable plan must carry a
`rejected_reason`.

**Why.** Forcing distribution on a model that fits on one GPU is the failure mode
this project exists to avoid. A planner that always distributes produces a
technically-working, economically-absurd system.

Verified: Qwen2.5-7B bf16 on a 4-node fleet is planned as SINGLE_GPU because that
is correct. Only 14B and 32B distribute.

**What would change this.** Nothing — this is the core property.

---

## D-007: Deterministic planner first, learned scheduler later

**Decision.** Execution Intelligence is a deterministic cost-based planner.

**Why.** A learned scheduler that nobody can audit is worse than a simple one
everybody can. The deterministic version produces the ground truth against which
any learned policy must be measured, and it makes every planning decision
reproducible from the same inputs.

The interface is reserved for a predictive policy later. It is not implemented,
and the code does not pretend otherwise.

**What would change this.** A measured case where the deterministic planner makes
a choice that a predictive model would improve on, with the improvement quantified.

---

## D-008: Reliability is earned, not configured

**Decision.** A node's reliability score is `None` until 10 heartbeats have been
observed. It is then `1 − missed/(observations + missed)`.

**Why.** A node that has never been seen is not 100% reliable; it is unknown. A
planner that treats unknown as reliable will place a model on a host that is about
to disappear, which is precisely the failure MESHWEAVE is supposed to survive.

**What would change this.** Nothing structural. The threshold (10) is a tunable
constant, not a decision.

---

## D-009: Synthetic telemetry is physically separated and statically guarded

**Decision.** `meshweave/agent/synthetic.py` is a separate module that
`telemetry.py` never imports. It stamps `synthetic` into the agent version, GPU
UUID and driver version. `tests/test_experiment_guard.py` statically rejects any
experiment importing it.

**Why.** The first acceptance criterion is that no simulated GPU aggregation is
presented as actual distributed inference. A convention is not a guarantee. A
static test is.

**What would change this.** Nothing. If anything, the guard should get stricter.

---

## D-010: The report validator rejects dishonest reports

**Decision.** A report claiming `status=completed` must have measurements, must
have investigated failures, and must either know its cost or list what is unknown.
A synthetic report can never be counted as evidence.

**Why.** Acceptance criteria 2, 3 and 4 (reproducibility, honest benchmarks,
economic measurement) are only real if they are enforced mechanically. Documenting
them as aspirations would allow the exact drift the brief warns against.

**What would change this.** Nothing.

---

## D-011: Report cost as unknown rather than estimating it

**Decision.** Unknown hourly cost propagates as `None` and appears in
`unknown_components`. Cost per token is never computed from a guessed price.

**Why.** A fabricated cost figure is indistinguishable from a measured one once it
is in a report. Since the entire economic argument rests on cost per token, a
single guessed price would invalidate it.

**What would change this.** Nothing.

---

## D-012: One model architecture behind an adapter interface

**Decision.** Qwen2.5 is the only supported family. `ModelAdapter` is the
interface; `Qwen25Adapter` is the implementation; `models/registry.py` holds
published shape facts so planning works without downloading weights.

**Why.** A generic distributed transformer engine is a large undertaking that
would deliver an untestable abstraction. The adapter reads shape facts from the
real `config.json` when present, so a variant with a different head count still
produces correct memory arithmetic.

**What would change this.** A second architecture that cannot be expressed through
the adapter — at which point the interface needs revisiting.

---

## D-013: Coordinator is a modular monolith

**Decision.** No microservices, no Kubernetes.

**Why.** Premature distribution of the control plane would multiply failure modes
before the execution path is proven. The brief says this explicitly, and the
engineering judgement agrees.

**What would change this.** A demonstrated need to scale the control plane
independently of the execution plane — which requires many more than 8 nodes.

---

## D-014: Model a replication efficiency factor

**Decision.** Replicated throughput is `tokens_per_gpu_sec × N × 0.85`.

**Why.** An earlier version scaled perfectly linearly, which made replication
always cheapest and rendered the comparison meaningless. Real replication loses
throughput to router queueing, imperfect batching, and straggler nodes in a
heterogeneous fleet. Without the factor, the planner would systematically
over-promise on replication.

The factor is a planning assumption. Every experiment overwrites it with the
measured aggregate throughput.

**What would change this.** Measured data showing the real efficiency is materially
different from 0.85 on our fleet — which is exactly what EXP-010 is for.

---

## D-015: Rename the repository to MESHWEAVE

**Decision.** The project is MESHWEAVE; the on-disk directory is `Meshwave` for
historical reasons and could not be renamed in this environment (the tool layer
rewrites the path). The Python package, CLI, git repo and all documentation use
`meshweave` / MESHWEAVE.

**Why.** Naming drift is a small thing that compounds. The package name is what
matters for imports; the directory name is cosmetic.

**What would change this.** A `git mv` once the working directory can be renamed
cleanly. Tracked in `ROADMAP.md`.