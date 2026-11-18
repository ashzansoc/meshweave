# API

Two services. The **coordinator** on `:8700` is the control plane. The **gateway**
on `:8701` is the OpenAI-compatible inference surface.

## Coordinator

### Nodes

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/nodes/register` | register or re-register an agent; returns a signed `node_token` |
| POST | `/v1/nodes/{node_id}/heartbeat` | update inventory; marks the node HEALTHY |
| GET | `/v1/nodes` | all nodes plus aggregates |
| GET | `/v1/nodes/{node_id}` | one node, status, degrade reason |
| GET | `/v1/nodes/{node_id}/series` | recent telemetry samples (session-only) |
| GET | `/v1/readiness` | per node: GPU, CPU, RAM, disk, ports, agent version, availability |
| DELETE | `/v1/nodes/{node_id}` | remove a node (admin secret) |

Enrollment requires `x-meshweave-enrollment: <token>` when the coordinator is
configured with one. Requesting work requires `Authorization: Bearer <node_token>`
or the enrollment token while an older agent is upgrading.

`availability` on readiness is one of `AVAILABLE`, `BUSY`, `OFFLINE`,
`DEGRADED`, `MAINTENANCE`.

### Clusters

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/clusters` | `{name, node_ids}` → creates `F16-VDC-001` |
| GET | `/v1/clusters` | all clusters with capacity and health |
| GET | `/v1/clusters/{id}` | one cluster |
| POST | `/v1/clusters/{id}/validate` | re-run the static checks now |
| DELETE | `/v1/clusters/{id}` | remove |

Creation refuses with 422 and a list of reasons when a member is offline, has no
GPU, runs an agent older than 0.4.0, or publishes no TCP port other than SSH.
Unmeasured links are a **warning**, not a refusal, because a cluster can still be
formed and measured afterwards.

### Deployments

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/deployments` | `{model, node_ids}` or `{model, cluster_id}` → plan and dispatch |
| GET | `/v1/deployments` | all deployments |
| GET | `/v1/deployments/{id}` | plan, per-stage phase, recent results |
| POST | `/v1/deployments/{id}/prompt` | queue a prompt (409 unless RUNNING) |
| POST | `/v1/deployments/{id}/nodes/{node}/status` | agent reports a stage phase |
| POST | `/v1/deployments/{id}/result` | agent reports a prompt result |
| POST | `/v1/deployments/{id}/stop` | ask every stage to stop |
| DELETE | `/v1/deployments/{id}` | stop and forget |

A refusal names the node and the shortfall: insufficient VRAM with the numbers,
insufficient disk with the numbers, no published port, or an agent that cannot
poll. A successful plan returns, per stage, which layers it holds, the VRAM it
will occupy and the address peers will dial.

### Network

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/network/measure` | `{node_ids, rtt_count, bulk_bytes}` → start a measurement |
| GET | `/v1/network/measurements` | recent measurements |
| GET | `/v1/network/measurements/{id}` | one measurement, including every direction |
| POST | `/v1/nodes/{node}/network-result` | agent reports an echo ready or a probe result |
| GET | `/v1/network/intelligence` | per-strategy suitability from the latest links |

Measurement is orchestrated, not dialled: each node is told to listen on a
published port, then each node is told to probe the others. A finished
measurement becomes the topology matrix the planner reads.

## Gateway

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/v1/models` | models currently served, from live deployments |
| POST | `/v1/chat/completions` | OpenAI chat shape |
| POST | `/v1/completions` | OpenAI text shape |

```bash
curl -s http://127.0.0.1:8701/v1/chat/completions \
  -H "Authorization: Bearer $MESHWEAVE_GATEWAY_KEY" \
  -H 'content-type: application/json' \
  -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 24,
        "temperature": 0
      }'
```

The response carries the usual `choices` and `usage`, plus a `meshweave` block
naming each stage's node and layers, the pipeline round trip and the activation
bytes per token — the evidence for what actually ran.

Errors are OpenAI-shaped: `400` bad body, `401` bad key, `429` the deployment is
already generating, `503` no deployment serves that model, `504` generation timed
out, `502` the model itself errored.

## Notes that matter

* One request per deployment at a time. Rank 0 serialises generation on its KV
  cache, so a second request gets 429 rather than queueing behind something that
  cannot make progress.
* `messages` is flattened to a prompt; no chat template is applied.
* The dashboard's `/api` prefix is a dev-server proxy to the coordinator, so
  dashboard calls are the same coordinator routes.