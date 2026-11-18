# Runbook

## Add a GPU node

On the new machine, one command. It is idempotent and keeps the node's identity
across re-runs.

```bash
curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/install.sh \
  | sudo bash -s -- \
      --coordinator https://<your-coordinator-url> \
      --token <enrollment-token> \
      --enroll
```

What that does: installs python, the NVIDIA bindings, the agent and the
distributed runtime, invents a node name (`silver-heron-3f9c`) and remembers it in
`/etc/meshweave/node_id`, registers, prints the machine's real GPU/CPU/RAM/disk,
and starts the agent under systemd, supervisor, or as a detached process.

**A multi-node deployment needs a published TCP port on the instance** — any port
other than 22. The provider maps it, the agent reports it, and the coordinator
uses it as the address peers dial. A node that publishes only SSH can still
report telemetry but cannot join a cluster of more than one node, and the API says
so.

Verify:

```bash
curl -s https://<coordinator>/v1/readiness | python3 -m json.tool
```

Each node should show `agent 0.4.0`, a GPU, and `mapped_ports` with at least one
entry.

## Form a cluster

```bash
curl -s -X POST http://127.0.0.1:8700/v1/clusters \
  -H 'content-type: application/json' \
  -d '{"name":"VDC-001","node_ids":["<node-1>","<node-2>"]}'
```

Returns `F16-VDC-001`, its members, their availability, and aggregate capacity
with the distributed-capacity note. A 422 lists exactly what is wrong.

The same thing is in the dashboard under **Clusters**.

## Measure the link before trusting it

```bash
curl -s -X POST http://127.0.0.1:8700/v1/network/measure \
  -H 'content-type: application/json' \
  -d '{"node_ids":["<node-1>","<node-2>"]}'
```

Both directions are measured: RTT, jitter, upload and download throughput, and
failures. Poll `GET /v1/network/measurements/<id>` until `phase` is `done`, then
`GET /v1/network/intelligence` for what that means per strategy.

A finished measurement becomes the topology, so `GET /v1/topology` stops saying
"no topology measured yet".

## Deploy a model

Dashboard **Deploy**, or:

```bash
curl -s -X POST http://127.0.0.1:8700/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","cluster_id":"F16-VDC-001"}'
```

The response is the plan: which node runs which layers, the VRAM each stage will
hold, and the address each stage will be dialled at.

Watch it come up:

```bash
watch -n 2 'curl -s http://127.0.0.1:8700/v1/deployments | python3 -m json.tool | head -40'
```

Each stage goes `pending → starting → loading → running`. `loading` is the
download; the model is cached afterwards, so a re-run is fast.

Once every stage is `running`, send a prompt (dashboard, or):

```bash
curl -s -X POST http://127.0.0.1:8700/v1/deployments/<id>/prompt \
  -H 'content-type: application/json' \
  -d '{"prompt":"The capital of France is","max_new_tokens":24,"temperature":0}'
```

The result arrives with token counts, TTFT, ms/token, the pipeline round trip,
activation bytes, and per-node GPU samples taken during the generation.

## Use the OpenAI-compatible endpoint

```bash
MESHWEAVE_GATEWAY_KEY=<choose one> \
MESHWEAVE_GATEWAY_COORDINATOR=http://127.0.0.1:8700 \
.venv/bin/meshweave gateway --port 8701
```

Then point any OpenAI client at `http://127.0.0.1:8701/v1` with that key.

## Benchmark A, B and C

```bash
python3 benchmarks/vdc/run_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes <node-1> <node-2>
```

Writes `benchmarks/vdc/results.json` and `results.md`.

## Add a third node

Nothing changes in the design: run the same installer command with a new node id,
then either form a new cluster with it or add it to an existing one by removing
and recreating the cluster. Deployments then split 24 layers across three nodes
instead of two, and the stage chain lengthens by one hop.

The per-node checks are unchanged: a published port, enough disk for the whole
checkpoint, enough VRAM for that node's layer range.

## Restart and recovery

* **Coordinator restart** — node state is restored from
  `reports/registry_snapshot.json`, clusters from `reports/clusters.json`,
  deployments from `reports/deployments.json`. The first heartbeat after a restart
  refreshes inventory. In-flight requests are not resumed; they fail.
* **Agent restart** — the agent re-registers if the coordinator no longer knows it
  (404 on heartbeat), and re-runs any stage it was assigned if the recorded pid
  is not alive.
* **Node death** — after `heartbeat_grace_s` (30 s by default) the node is
  `UNREACHABLE`, its stages are failed, and any deployment it hosted is `FAILED`.
* **Stage death** — the request fails. There is no mid-generation recovery; the
  KV cache that made it possible lived in that process.