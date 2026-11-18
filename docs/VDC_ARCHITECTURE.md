# Float16 VDC — architecture

The short version: a coordinator that never touches a GPU, agents that only dial
out, and stages that hold a slice of a model and pass hidden states to each other.

```
                ┌───────────────────────────────────────────┐
                │  Coordinator (control plane)              │
                │  nodes · clusters · deployments · links    │
                │  work queue · evidence                     │
                └───────────────┬───────────────────────────┘
                                │ outbound HTTPS (agent dials)
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼─────────┐                             ┌───────▼─────────┐
│ Node 01         │                             │ Node 02         │
│ agent (0.4.0)   │                             │ agent (0.4.0)   │
│  ├ NVML telemetry                             │  ├ NVML telemetry
│  ├ stage worker │◄──── hidden states ────────►│  ├ stage worker │
│  └ unix control │      (direct TCP)           │  └ unix control │
└─────────────────┘                             └─────────────────┘
```

## Components

| Component | Lives in | Does |
|---|---|---|
| Coordinator | `meshweave/coordinator/app.py` | REST control plane; registers nodes, plans deployments, runs the work queue, stores evidence |
| Node registry | `meshweave/registry/store.py` | node records, heartbeats, snapshot persistence, reaping, reliability |
| Cluster manager | `meshweave/coordinator/clusters.py` | names groups of nodes, validates, computes aggregate capacity and health |
| Deployment planner | `meshweave/coordinator/deployments.py` | splits layers, checks VRAM/disk/ports per node, refuses with a reason |
| Link measurement | `meshweave/coordinator/network.py` + `meshweave/runtime/echo.py` | orchestrates node-to-node probes; produces the topology matrix |
| Network intelligence | `meshweave/network/intelligence.py` | per-strategy suitability from measured communication vs measured compute |
| Agent | `deploy/agent.py` | telemetry, registration, work queue, stage supervision, prompt relay |
| Stage runner | `deploy/stage_runner.py` | owns a layer range; chained over TCP; samples on the last stage |
| Model runtime | `meshweave/runtime/dense.py` | Qwen2 maths: RMSNorm, RoPE, GQA, SwiGLU, KV cache |
| Gateway | `meshweave/gateway/app.py` | OpenAI-compatible API in front of a running deployment |
| Dashboard | `dashboard/src` | nodes, readiness, clusters, deployments, topology, network, planner |

## The request path

A prompt does not travel the path you would expect.

1. The gateway (or the dashboard) POSTs to `/v1/deployments/{id}/prompt`.
2. The coordinator queues the prompt and returns its id.
3. The agent on rank 0 polls `/v1/agents/{node}/work`, receives `run_prompt`, and
   writes it to a **unix socket** on the same host.
4. Rank 0 embeds the tokens, runs its layers, and sends the hidden state down the
   chain.
5. Each stage runs its own layers and forwards the hidden state.
6. The last stage samples the next token and sends the **token id directly back to
   rank 0**, not back along the chain.
7. Rank 0 posts the result to the coordinator; the gateway polls for it by id.

Nothing dials into a node. The stage ports are published for peer traffic, but
they only accept frames carrying the deployment's stage token, and the control
socket is a local file with mode 0600.

## What is distributed and what is not

* **Weights are split.** Each stage loads only its own layers from the checkpoint
  on its own disk. A stage that owned 12 of 24 layers holds about half the
  parameters, not a copy.
* **Compute is split.** Both GPUs run a forward pass during the same generation,
  and NVML is sampled on each node *during* the request to record it.
* **VRAM is not pooled.** Aggregated VRAM is reported as capacity across separate
  devices, with a note that it is not one addressable pool. A model that no
  single member can hold will not run by pretending otherwise.
* **Disk is not split.** Every node downloads the full checkpoint, because the
  safetensors shards are not split finely enough to fetch one layer range.

## Node states

The registry tracks lifecycle (`REGISTERING`, `HEALTHY`, `DEGRADED`,
`UNREACHABLE`, `DRAINING`, `RETIRED`). The API also reports an **availability**
derived from that plus whether the node is currently hosting a stage:

| Availability | Means |
|---|---|
| `AVAILABLE` | healthy, nothing running |
| `BUSY` | healthy, hosting a stage |
| `OFFLINE` | unreachable or still registering |
| `DEGRADED` | reported degraded by its own checks |
| `MAINTENANCE` | drained for maintenance |

The mapping is deliberate: "alive" and "free to use" are different questions, and
conflating them is how a scheduler ends up planning onto a busy GPU.

## Credentials

* **Enrollment token** — proves a host may join; sent as `x-meshweave-enrollment`
  to `/v1/nodes/register`. Configured via `MESHWEAVE_COORD_ENROLLMENT_TOKEN`.
* **Node token** — issued by the coordinator at registration, signed with the
  coordinator secret, stored by the agent with mode 0600, and required to ask for
  work. It names exactly one node, so one node cannot request another's queue.
* **Stage token** — per deployment, required in the TCP handshake between stages.
* **Admin secret** — `reports/.coordinator_secret`, required for destructive
  control-plane actions.
* **Gateway key** — `MESHWEAVE_GATEWAY_KEY`, checked as a bearer token on the
  inference API.

The enrollment token is still accepted by the work endpoint so an agent that
predates node tokens keeps working during an upgrade. That is a migration aid, and
it is documented as such.

## Failure behaviour

Reaping runs on a timer (`heartbeat_grace_s / 3`), not on request. A node that
stops heartbeating is marked `UNREACHABLE` and the state is persisted, so a
restart does not resurrect it as healthy. A deployment whose node was reaped is
marked `FAILED` with the reason; it does **not** claim to keep generating.

There is no mid-generation recovery. The KV cache for a lost stage lives in that
stage's memory and is unrecoverable, so the request fails. Recovery means
re-partitioning and starting again, which is a later capability and is labelled
that way rather than implied.

## Where to look next

* `docs/VDC_API.md` — every endpoint
* `docs/VDC_RUNBOOK.md` — adding a node, deploying a model, benchmarking
* `docs/LIMITATIONS.md` — what this POC does not do
* `docs/EVALUATION.md` — why the stack is what it is