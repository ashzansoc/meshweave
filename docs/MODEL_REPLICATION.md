# Model replication

Running independent full copies of a model on separate machines, and routing
requests between them.

**Why it exists:** splitting a model across machines costs a round trip per token.
Replicating it costs nothing between nodes. On a 226 ms link with two 8 GiB cards,
replication is both faster per request and the only configuration whose
throughput grows when you add machines.

**Measured:** replicas beat the best single node at every concurrency tested and
beat the split by 5.4× at concurrency 4 (`docs/PHASE_2_RESULTS.md`).

---

## 1. What a replica is

A replica is a complete model in one process: `world=1`, all 24 layers, its own
KV cache, its own control socket. Nothing is transferred between replicas, and
they do not know about each other.

That is deliberate. The alternative — sharing work between nodes that hold the
same weights — would add latency to buy nothing.

Each replica's VRAM is measured from the process, not estimated:
1201.9 MiB resident on both nodes here.

## 2. The rule: replicate first

`mode: AUTO` (the default) asks whether any one selected node can hold the whole
model — weights plus a KV cache for the requested context — and replicates if so.
Only when no single node fits does it fall back to a layer split.

The reason is measured, not aesthetic:

| | tok/s (concurrency 4) |
|---|---|
| one node, whole model | 10.60 |
| two nodes, split 12/12 | 2.36 |
| two nodes, replicas | 12.74 |

Splitting a model that fits makes it **4.5× slower** than leaving it alone.

## 3. Capacity is per node

A cluster's aggregate VRAM is reported as capacity across separate devices, never
as one addressable pool. A replica must fit on **one** node, and the check is
against that node's free memory with 15 % headroom.

## 4. Disk is not replicated

Every node downloads the full checkpoint (about 1 GiB for a 0.5B model in bf16)
because safetensors shards are not split finely enough to fetch one layer range.
A node with insufficient disk is refused by name.

## 5. Operating

```bash
# replicate explicitly
curl -s -X POST http://127.0.0.1:8700/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","node_ids":["gpu-node-1","brisk-thistle-9ae2"],"mode":"REPLICA"}'
```

Each stage reports `running` independently. `GET /v1/deployments/{id}` shows
`replica_load` per node, so the spread is observable rather than asserted.

## 6. Scaling

Adding a node means running the installer on it and either forming a new cluster
or re-deploying across the enlarged set. There is no autoscaler: replica count is
a deployment decision, which is honest for a POC and is where a production system
would add health-based scaling.

## 7. Limits

* One request in flight per replica. The stage runner serialises on its KV cache,
  so a second request queues rather than running concurrently.
* No cross-replica cache sharing, so a repeated prompt recomputes its prefill on
  whichever replica receives it.
* A replica that dies takes its in-flight requests with it; the router stops
  choosing it once its phase is not `running`, but there is no request replay.