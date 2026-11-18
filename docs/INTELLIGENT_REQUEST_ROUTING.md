# Intelligent request routing

Which replica serves a given request.

**Implemented in:** `choose_replica` (`meshweave/coordinator/deployments.py`),
called from `POST /v1/deployments/{id}/prompt`.

**Verified:** four concurrent requests on two replicas split 2/2, and every result
released the load it was counted against.

---

## 1. The policy

```
candidates = replicas whose phase is "running" AND whose node is HEALTHY
choose     = min by (in_flight, queued_for_this_node, node_id)
```

Three signals, in that order:

1. **In flight** — requests dispatched to this replica and not yet answered.
   This is the load that matters, because the runner serves one at a time.
2. **Queued** — assigned but not yet picked up, a tie-breaker for the gap
   between dispatch and the agent's poll.
3. **Node id** — a deterministic final tie-break, so two identical requests
   cannot oscillate between replicas and make measurements irreproducible.

A replica is only a candidate when it is both **running** (it has loaded and is
listening) and its node is **HEALTHY** (it is heartbeating). A deployment with no
healthy replica returns **503** rather than sending work somewhere that cannot
take it.

## 2. Why not more signals

The brief lists KV-cache utilisation and estimated completion time. Neither is
implemented, and the reason is that neither is observable here yet:

* **KV-cache utilisation** — each replica resets its cache per request and holds
  one request at a time, so utilisation is either zero or full. Routing on it
  would be routing on a constant.
* **Estimated completion time** — this needs a per-replica latency model. The
  per-node ms/layer used by placement is a reasonable prior and is where this
  would come from, but it has not been wired into routing.

Reporting these as implemented would be false. They are the next increment, and
both need per-replica telemetry the agent does not yet send.

## 3. Load accounting

`replica_load[node]` is incremented when a prompt is assigned and decremented when
a result arrives.

Two bugs were fixed here, both of which made the router believe replicas were
permanently busy:

* the agent attributed results to a node only on **error**, so a successful
  result never released its slot. It now always attributes.
* when a result carried no node, the coordinator had nothing to decrement. It now
  falls back to the prompt's `assigned_node`.

Both are covered by tests, including a result that omits its origin entirely.

## 4. Failure behaviour

| Situation | Behaviour |
|---|---|
| replica not running | excluded from candidates |
| node not HEALTHY | excluded from candidates |
| no candidates | 503, with a reason |
| replica dies mid-request | result reports an error; load released; the replica stops being a candidate while it is down |

No request replay. A lost generation is lost; the platform reports it rather than
pretending it continued.

## 5. What routing cannot fix

Routing spreads load; it does not make a single generation faster. The
single-request numbers in `PHASE_2_RESULTS.md` show replicas at 23.2 tok/s against
25.1 for one node alone at concurrency 4 — routing adds coordination, and the gain
appears in **aggregate** throughput (12.74 vs 10.60), not per request.

## 6. Inspecting it

```bash
curl -s http://127.0.0.1:8700/v1/deployments/<id> | python3 -m json.tool | grep -A4 replica_load
```

```json
"replica_load": { "gpu-node-1": 1, "brisk-thistle-9ae2": 1 }
```

## 7. Limits

* Least-in-flight is greedy; it does not account for a replica running on a
  slower GPU. Weighting by measured ms/layer would be the next refinement.
* No admission control: at high concurrency requests queue inside a replica
  rather than being rejected or shed.
* The choice is made at submission time and never re-evaluated. There is no
  migration of an in-flight request.