# Capacity-aware placement

How MESHWEAVE decides which node runs which decoder layers.

**Replaces:** an even split, which gave every node the same number of layers
regardless of how fast it was or how much memory it had free.

**Verified:** the estimator lands within 12 % of the measured 310.8 ms/token for
the deployed 12/12 split (`docs/PHASE_2_RESULTS.md`).

---

## 1. The cost model

For one generated token, a pipeline of `N` stages costs:

```
(N_forward_hops + 1) × RTT/2     activation travel, out and back
+ Σ stage_compute                 each stage runs its own layers
+ Σ activation_bytes / bandwidth  serialisation and transfer
```

The `RTT/2` per hop is not a fudge. The token returns **directly** from the last
stage to rank 0 rather than travelling back along the chain, so a two-stage
pipeline costs one round trip per token, not two. That property is what took the
earlier three-host decode from 2,663 to 453 ms/token.

`stage_compute` is `layers × ms_per_layer` for that node. `ms_per_layer` comes
from a recorded single-node run when one exists.

## 2. Where the inputs come from

| Input | Source | If missing |
|---|---|---|
| ms per layer per node | recorded single-node ms/token ÷ layer count | `DEFAULT_MS_PER_LAYER` (2.0), **reported as an assumption** |
| RTT per direction | `/v1/topology`, from a real measurement | hop cost omitted, **reported** |
| Bandwidth per direction | same | transfer cost omitted, **reported** |
| Layer count, hidden size | the model's own `config.json` | refused |

Nothing is silently defaulted. `Candidate.assumptions` lists every estimate, so a
report can never present a guess as a measurement.

## 3. The search

Every composition of the layer count across the selected nodes is scored. For 24
layers over 2 nodes that is 23 candidates; over 3 it is 253. The search caps at
400 candidates and reports that it truncated — the even split is always included
so the comparison against the old behaviour survives.

**A balance tolerance of 5 %** decides ties. The model omits per-stage overhead
and the head's vocabulary projection, so its error is larger than the difference
between most splits. Without the tolerance the optimiser "prefers" a 23/1 split
over 12/12 for a predicted 1 ms on a 275 ms estimate, which is overfitting and
makes the pipeline fragile.

On the current fleet the outcome is 12/12: the two cards are close enough in
speed that the predicted spread across candidates is ~1 ms. The search earns its
keep with heterogeneous hardware — a test asserts that a node twice as fast
receives more layers.

## 4. What is checked before a split is attempted

Planning refuses, with the node and the shortfall named:

* no GPU reported
* free VRAM below the stage's weights plus its own KV reservation
* disk below the full checkpoint (every node downloads all of it)
* no published TCP port, when there is more than one stage
* an agent too old to poll for work

The KV reservation is per stage, counting only the layers that stage owns. It
previously counted the whole model's KV on every node, over-reserving by the node
count and refusing deployments that would have fitted.

## 5. Using it

The dashboard's **Deploy** view, or:

```bash
curl -s -X POST http://127.0.0.1:8700/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","cluster_id":"F16-VDC-001","mode":"PIPELINE"}'
```

The response carries each stage's layer range, resident VRAM and dial address.
`mode: AUTO` asks the coordinator to choose between this and replication.

## 6. Limits

* The model is per-token steady state; it does not model prefill separately.
* Embedding and head costs are folded into the per-layer average rather than
  counted separately, and the result says so.
* Per-node speed is only as good as the last recorded run on that node. A node
  that has never served reports the default and is labelled.
* Bandwidth is a single number per direction; the link is not modelled as
  window-limited, which earlier phases showed it is above 8 KiB.