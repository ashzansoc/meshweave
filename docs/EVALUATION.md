# Stack evaluation

The brief lists several technologies and asks for investigation before
installation. This is that investigation, with the evidence each verdict rests on.
Nothing here is an opinion about a project's quality; it is about whether it fits
two 6–8 GiB Turing GPUs on containers behind carrier NAT, ~300 ms apart.

## Distributed execution

### Parallax — evaluated, not adopted

| | |
|---|---|
| Version at review | 0.0.1, released 2025-10 (GitHub) |
| Claim | pipeline parallel sharding across heterogeneous nodes, P2P via "Lattica" |
| Licence | Apache-2.0 |
| Hardware matrix | none published |

The architecture is the right shape for this problem, and it is the closest thing
in the brief to what MESHWEAVE already does. Three things argued against adopting
it as the execution path for this POC:

1. **Maturity.** v0.0.1 is early. Betting the POC's core claim on a first release
   is a poor trade when a working path already exists in this repository.
3. **No hardware matrix.** Turing (compute 7.5) and 6–8 GiB of VRAM are outside
   the territory most serving stacks are tested on. Whether Parallax fits a 0.5B
   model in 6 GiB across two nodes is unknown, not unlikely.
3. **It duplicates what is proven here.** This repository already partitions a
   model by layers, moves only hidden states, returns the sampled token directly,
   and has been checked against `transformers` to within 1.8e-07 in fp32, with two
   real processes producing identical output.

**Verdict: not adopted for POC 01. Recommended as a follow-up spike**, with a
written result either way. If it fits these cards, it is a credible alternative
execution backend behind the same agent and coordinator.

### NVIDIA Dynamo — excluded for this environment

Dynamo describes itself as a datacenter-scale serving framework. Its documentation
references NVL72, GB200/B200/H200-class hardware, and placement across racks,
hosts and NUMA nodes. It makes no claim about coordination over high-latency
internet links, and its component set (disaggregated prefill/decode pools, a
KV-aware router, NATS/etcd, Kubernetes for production) is aimed at a different
problem than two rented containers.

Adding it here would add an orchestration layer that cannot help with the actual
bottleneck — a 300 ms round trip — while introducing a second control plane.

**Verdict: excluded. Revisit only for a real datacenter cluster (environment A).**

### vLLM and SGLang — single-node baselines

Both are excellent single-node engines and both support pipeline parallelism, but
that path assumes a low-latency distributed group (NCCL over a fast interconnect)
and will not survive a WAN link. Their value in this POC is as the reference for
**Benchmarks A and B**: what one machine can do without any distribution, so the
comparison in the report is against a real serving engine rather than only
against MESHWEAVE's own single-stage mode.

**Verdict: optional baseline. Not installed by default** — the benchmark harness
measures MESHWEAVE's own single-node path first, and vLLM can be added as an
external baseline without changing anything else.

### Ray — not introduced

Ray is a distributed scheduler. Two nodes and one model need no scheduling beyond
"which node runs which layer range", which the coordinator already computes and
the work queue already delivers. Introducing Ray would add a second system with
its own failure modes and its own dependency set, in a POC whose remaining risk is
the WAN link.

The brief permits this: "do not introduce multiple overlapping orchestration
systems without a clear reason". There is no reason here.

### Kubernetes — not introduced

Optional per the brief, and the repository already recorded the decision
(`DECISIONS.md` D-013). These hosts are unprivileged containers without systemd or
`/dev/net/tun`; a Kubernetes control plane would add operational surface and
change nothing about whether a model can run across two GPUs.

## Node agent language

The brief prefers Go. The current agent is Python, deployed and hardened on both
nodes: NVML telemetry, registration with signed node tokens, heartbeats with
sample history, stage supervision with ownership checks, and the work queue.

Rewriting it in Go would not remove the Python dependency, because the GPU
execution worker — torch, the GGUF/safetensors loader and the forward pass — is
Python. A Go agent would sit beside a Python worker rather than replace it.

**Verdict: keep Python for this POC; documented as a deviation.** A single static
binary is a reasonable goal if node count grows to thousands, and the agent's
control surface is deliberately small enough to port cleanly. The trade-off is
recorded in `docs/LIMITATIONS.md`.

## What this means for the execution path

MESHWEAVE's own pipeline runtime is the execution path because it is the only
component here that is (a) proven to partition weights on this hardware, (b)
numerically checked against a reference implementation, and (c) measured over the
links in question. Everything else is either a baseline or a follow-up.

The benchmark report is the place where "is distribution worth it" gets answered
with numbers rather than argument. Prior phases already measured it on a different
model and topology: distributed decode at 453 ms/token against 7.5 ms/token on a
single GPU, a 60× penalty. Whether this fleet does better is an open question and
is what M7 is for.