# Unified inference architecture

The target design for Float16, and the reasoning behind it. This describes where
the system is going; `docs/CURRENT_ARCHITECTURE_AUDIT.md` describes where it is.

The organising idea: **Float16 owns intelligence and placement; backends own
execution.** A customer submits one request and never learns which engine ran it,
because the choice is made from measured hardware, measured links and the model's
own geometry.

---

## 1. Two objectives that pull in opposite directions

These are separate targets and must be measured separately. Conflating them is
how a platform ends up reporting a single throughput number that flatters one and
hides the other.

### 1.1 Single-request TPS — how fast one generation completes

Levers, in order of effect on this hardware:

1. **Do not split what fits.** Splitting a 0.5B model across a 226 ms link costs
   6.5× (19.02 → 2.93 tok/s measured). Placement must prefer replication.
2. **Remove round trips.** Direct token return already removed one; there is one
   per token left and it is the dominant cost.
3. **Prefix caching.** Reuse the KV cache for a repeated system prompt or
   conversation prefix, so a request skips prefill for the shared part.
4. **Speculative decoding.** Investigate, not assume: a draft model proposes k
   tokens, the target verifies them in one pass. On a link where each round trip
   costs 226 ms, verifying several tokens per round trip is the only way to
   amortise it — but acceptance depends on the draft model and it must preserve
   autoregressive sampling exactly.

Distribution is a last resort for this metric, used only when the model cannot
fit one GPU.

### 1.2 Aggregate TPS — tokens across many concurrent customers

Levers:

1. **Independent replicas.** A 0.5B model fits a 6 GiB card several times over;
   running one per node and routing between them scales throughput linearly with
   nodes, with no inter-node traffic at all.
3. **Batching.** Continuous batching raises GPU utilisation, but on a WAN link
   the per-request round trip dominates and batching trades latency for
   throughput. It belongs on the low-latency tier.
4. **Routing.** Least-loaded or KV-aware routing across replicas. This is where a
   Dynamo-style frontend eventually fits, once datacenter nodes exist.

### 1.3 The rule

For a model that fits one GPU, **replicate**. For a model that does not, and only
then, **partition** — and partition where the measured links make it least
painful.

---

## 2. Placement by environment class

Classes are assigned from measured link characteristics, never from geography.

| Class | Interconnect | Default strategy | Why |
|---|---|---|---|
| A — datacenter | NVLink / InfiniBand / same rack | tensor parallel (DeepSpeed or vLLM) | per-layer collectives are microseconds; TP is simpler and faster than PP |
| B — regional | fast Ethernet, <20 ms | tensor or pipeline, measured | collectives are affordable |
| C — internet | 100–300 ms, variable | **replicate**, partition only if it does not fit | one round trip per token; splitting multiplies it |

Within class C, placement must be **capacity- and latency-aware**, not an even
split:

* Size each stage from the model's geometry and the node's *free* VRAM, not its
  total.
* Prefer giving more layers to the faster GPU.
* Place the embedding and head where the link cost is lowest, since the head
  stage returns the token.
* Evaluate several candidate splits and estimate each one's per-token cost as
  `hops × RTT + bytes / bandwidth + compute`, using measured RTT and bandwidth
  rather than a constant.
* Choose the cheapest *viable* candidate, and refuse if none is viable — never
  divide evenly and hope.

That estimation is exactly what `meshweave/network/intelligence.py` already
computes per strategy. It needs to feed placement rather than only the dashboard.

---

## 3. Where each technology sits

```
                    Float16 API  (OpenAI-compatible, auth, streaming)
                              |
                    Global control plane
                              |
        +-------------+-------+--------+-------------+
        |             |                |             |
   Resource     Network          Execution      Serving
   registry    intelligence        strategy      orchestration
        |             |                |             |
        +-------------+----------------+-------------+
                              |
              +---------------+---------------+
              |               |               |
        NATIVE (default)   PARALLAX        DEEPSPEED / DYNAMO
        pipeline, split    WAN candidate    datacenter class A
        by layer           if Turing        and production serving
              |               |               |
              +---------------+---------------+
                              |
                          GPU NODES
```

This is a placement of responsibilities, not a pipeline that every request
traverses. A request goes through exactly one execution backend.

| Technology | Where it fits | Current status |
|---|---|---|
| Native runtime | Classes B and C, and the fallback everywhere | default |
| Parallax | Class C, if it runs on Turing | spike pending |
| DeepSpeed | Class A | not installed; not viable on WAN |
| Dynamo | Serving tier above class A | not installed; Turing outside its matrix |
| vLLM / SGLang | Class A single-node baselines and future workers | optional |

---

## 4. The execution interface

Every backend implements the same contract, so the strategy engine can choose
without special cases:

```
capabilities()      -> architectures, dtypes, min_vram, max_nodes, requires_nvlink
validate(model)     -> ok, reason
validate(cluster)   -> ok, reasons[]        # links, VRAM, ports
estimate(model, placement) -> ttft_ms, ms_per_token, bytes_per_token
deploy(model, placement)   -> deployment_id
generate(request) -> tokens + metrics
metrics()           -> per-stage timing, VRAM, transfers
terminate(id)
```

The important method is `capabilities()`. A backend that cannot run on this
hardware returns `supported: false` with a reason, and the strategy engine moves
on. Nothing pretends.

---

## 5. Data flow for one request

```
client ──► gateway ──► strategy engine
                          │
                 capabilities + measured links + model geometry
                          │
              ┌───────────┴────────────┐
          fits one GPU?            does not fit?
              │                        │
        pick a replica            plan a layer split
              │                        │
        route by load            dispatch stages
              │                        │
              └───────────┬────────────┘
                          │
                     tokens + metrics
```

Metrics recorded per request, and never merged: time to first token, ms per
token, activation bytes, pipeline round trip, per-stage VRAM, GPU utilisation
during the request, and which backend and placement produced it.

---

## 6. What changes from today

| Today | Target |
|---|---|
| Even layer split | Capacity- and latency-aware placement |
| Always splits | Replicates when the model fits |
| One request per deployment | Queue with batching on the fast tier |
| No streaming | SSE through the gateway |
| Frameworks refused | Frameworks behind `capabilities()` |
| One tok/s number | Single-request and aggregate reported separately |

## 7. What stays

The agent work queue, the framing layer, the geometry module, the refusal model,
the benchmark harness, the dashboard's evidence discipline, and the
process-ownership rules. Those are the parts that are already right, and the audit
says so explicitly.

## 8. Sequencing

1. Capacity-aware placement (this is the missing intelligence).
2. Replication with routing, and the dual-metric benchmark harness.
3. Streaming through the gateway.
4. Parallax behind the interface, if the spike passes.
5. Batching and prefix caching on the fast tier.
6. Speculative decoding, if measurement justifies it.

Each step has a baseline, a hypothesis and an A/B result. None of them requires
changing the hardware or the other frameworks.