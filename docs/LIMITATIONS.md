# Limitations

What this POC does not do, and where the boundaries are structural rather than
unfinished.

## Model support

One architecture family: `Qwen2ForCausalLM`, which covers Qwen2 and Qwen2.5 dense
models. Anything else is refused at deploy time with the architecture named. There
are no MoE models, no quantised checkpoints, and no vision or audio models.

`rope_scaling` and `partial_rotary_factor != 1` are refused rather than
approximated, because a wrong rotary angle produces plausible-looking but wrong
attention.

## Precision

The checkpoint's dtype is used, downgraded to fp16 on cards below compute 8.0
(the 2070 SUPER and 1660 SUPER in this fleet). There is no quantisation path yet,
so VRAM use is the model's native size divided across stages.

## Concurrency

One request per deployment. Rank 0 holds a single KV cache and serialises
generation on a lock; a second request gets HTTP 429 rather than queueing behind
work that cannot progress. Real concurrency would mean either multiple deployments
or a per-request cache, neither of which is implemented.

## Streaming

The gateway is non-streaming. Tokens arrive as one response. Streaming would need
an incremental control protocol from rank 0 through the agent to the coordinator,
and is deliberately left out rather than faked with a delayed single chunk.

## Chat templates

`messages` is flattened to a prompt with role markers. No chat template is
applied, so a multi-turn conversation is concatenated text, not a role-aware
prompt. This is stated in the API rather than hidden.

## Recovery and re-partitioning

There is no mid-generation recovery. A stage that dies takes the request with it,
because its KV cache cannot be reconstructed. There is no re-partitioning of a
running deployment: changing the split means stopping and starting again.

`meshweave/fault/recovery.py` contains the reasoning for why a lost pipeline
cannot continue, but it is not wired into the deployment path.

## Network

* Packet loss is **not** measured. TCP retransmits, and ICMP is unavailable in
  these containers, so a probe reports RTT, jitter, throughput and a failure
  count instead.
* `tc netem` latency injection is not available, because the containers are
  unprivileged. Fault experiments that need it say so.
* Clock synchronisation is not established, so no one-way latency is reported —
  only RTT, and `rtt/2` where an estimate is labelled as one.
* The coordinator's public URL in this POC is an ephemeral quick tunnel. A
  multi-day run needs a stable endpoint, and the agents must be re-run against it
  if it changes.

## Performance expectations

Distribution over these links is a **capacity** mechanism, not a latency one.
Prior phases measured 453 ms/token for a three-host pipeline against 7.5 ms/token
on a single GPU — a 60× penalty. The POC exists to prove that both machines
genuinely compute and that the model runs at all; it does not claim to be faster.

## Node agent language

The agent is Python, not Go as the brief prefers. The GPU worker it launches is
Python regardless, so a Go agent would sit beside it rather than replace it. A
single static binary is worth doing when node count grows; the control surface is
small enough to port cleanly.

## Security

* The stage ports are published for peer traffic and are protected by a per
  deployment token in the TCP handshake. They are still internet-reachable; a
  production deployment should put them behind an authenticated overlay.
* The work endpoint accepts the shared enrollment token during upgrades, so a
  leaked enrollment token is a way to ask for work until agents are updated. The
  node token is the intended long-term credential.
* The node token has no expiry. It is signed and scoped to one node, but rotation
  is not implemented.
* The gateway's API key is optional; without it the inference endpoint is open to
  anyone who can reach it, and it warns on startup.
* The coordinator's own control routes (topology, plan, deployments) are
  unauthenticated. The admin secret protects destructive actions only. This is
  acceptable for a local POC and not for an exposed deployment.

## Operational

* The coordinator holds node state in a JSON snapshot and deployment state in
  another; there is no database, and concurrent coordinators would race on the
  same files.
* Telemetry history is in memory only and is lost on restart. It is labelled as
  session-only in the API.
* The dashboard is a local dev server; it is not built for remote access without a
  tunnel.

## What was not done in this POC

Kubernetes, Ray, Parallax integration, vLLM/SGLang backends, tensor parallelism,
quantisation, multi-tenant clusters, and training. Each is either a follow-up
spike with a written verdict (`docs/EVALUATION.md`) or explicitly out of scope for
a two-node inference POC.