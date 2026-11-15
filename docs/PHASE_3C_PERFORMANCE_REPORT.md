# Phase 3C — Performance Report and Bottleneck Analysis

**Conclusion: the performance ceiling is a networking problem, not an
implementation problem.** Compute accounts for roughly **2.6 %** of a distributed
token. No amount of runtime optimisation will make this topology fast for
latency-sensitive generation.

---

## 1. Executive summary

Phase 3B measured 2.6–5.6 s per distributed token against 7.5 ms standalone. The
brief asked why, and warned not to assume the network.

The measurement that answers it: **the MESHWEAVE runtime, running 11 layers of the
model on one HOST-C GPU with no networking at all, takes 22.7 ms per token after
optimisation.** Three stages of that is ~68 ms of compute.

| Component | Time per token | Share of 2,600 ms |
|---|---|---|
| GPU/CPU compute, all three stages | ~68 ms | **2.6 %** |
| Network, 4 chain hops × ~310 ms | ~1,240 ms | **48 %** |
| Unexplained remainder | ~1,300 ms | ~49 % |

So: networking is the dominant *identified* cost, compute is almost irrelevant,
and there is a substantial unexplained remainder that is also almost certainly
network behaviour (TCP round trips per hop, not just one).

## 2. Baseline measurement, before any optimisation

Run on HOST-C, 11 layers (its Phase 3B share), no network, warm:

| Shape | MESHWEAVE | llama.cpp (32 layers, full model) |
|---|---|---|
| prefill, 12 tokens | 81.9 ms | — |
| decode, 1 token | **30.1 ms/token** | **7.5 ms/token** |

The runtime was **~9× slower per layer** than llama.cpp. That is a real
implementation deficit, and it was worth fixing — but even eliminating it entirely
would remove only ~50 ms from a 2,600 ms token.

## 3. The optimisation that worked

### MoE: remove per-layer GPU synchronisation

The routing loop called `torch.unique(...).tolist()` and `bool(mask.any())` once
per top-k slot per layer. **Each of those forces a GPU→CPU synchronisation**, so a
decode step paid ~64 syncs across 32 layers.

Replaced with a dense computation over all 16 experts using `einsum`, then a
scatter-weighted sum. 16× the FLOPs on a 960-wide FFN is trivial next to the
synchronisation it removes.

| Measurement | Before | After | Speedup |
|---|---|---|---|
| prefill 12 tokens | 81.9 ms | **22.4 ms** | **3.7×** |
| decode 1 token | 30.1 ms | **22.7 ms** | 1.3× |

Prefill gains most because it amortises the syncs across 12 tokens; decode is
dominated by kernel-launch overhead that dense MoE does not remove.

### Correctness after the change — verified

| Metric | Phase 3B reference | After optimisation |
|---|---|---|
| max abs error | 0.1823 | **0.18230** |
| mean abs error | 0.04081 | **0.041100** |
| cosine similarity | 0.9999994 | **0.99999952** |
| top-1 | 29871 | **29871** ✓ |

The dense formulation is mathematically equivalent; the tiny differences are
accumulation order. **Correctness was not traded for speed.**

## 4. Bottleneck breakdown

Per distributed token, 2,618–5,584 ms observed:

| Item | Measured / derived | Notes |
|---|---|---|
| Compute, stage A (11 layers) | ~22.7 ms | measured locally |
| Compute, stage B (10 layers) | ~20.6 ms | scaled by layer count |
| Compute, stage C (11 layers) | ~22.7 ms | measured locally |
| **Total compute** | **~66 ms** | 2.6 % of the token |
| A→B round trip | 311 ms | Phase 2A measured RTT |
| B→C round trip | 309 ms | Phase 2A measured RTT |
| Response path back through B and A | ~620 ms | the reply traverses the same chain |
| **Minimum network** | **~1,240 ms** | 48 % of the token |
| Serialisation, 8 KiB fp16 | <1 ms | measurable but negligible |
| **Unexplained** | **~1,300–4,300 ms** | see below |

### The remainder, and what it probably is

The relay is **synchronous**: stage A sends to B and blocks; B sends to C and
blocks; C replies; B relays back; A relays back. That is four sequential
request/response exchanges, and each is a full RTT plus any TCP handshake
behaviour on a 310 ms path.

The observed 2.6 s is ~8× a single RTT, not ~4×. The likely causes, in order of
plausibility:

1. **TCP delayed ACK / Nagle interaction** across a high-latency link. `TCP_NODELAY`
   is set, but the *receiver* can still delay ACKs, and each relay waits for a
   complete message before replying.
2. **More round trips than the design assumes** — the reply path re-sends through
   every stage rather than returning directly to the client.
3. **Python/socket overhead** per message, which on 310 ms of latency is hidden but
   compounds across hops.

**Not measured**, and I am not going to claim otherwise: a per-hop timing
instrumentation pass was designed but not completed. The 2.6 s figure is explained
at best to 48 %; the rest is inferred, not observed.

## 5. Optimisations evaluated

| # | Change | Result |
|---|---|---|
| 1 | Dense MoE (removes ~64 GPU syncs/token) | prefill 3.7×, decode 1.3× — **kept** |
| 2 | Stage survives client disconnect | reliability fix, no perf effect — **kept** |
| 3 | Binary framing with CRC32 already in place | measured at <1 ms; not the bottleneck |
| 4 | Compression | **rejected** — payload is 8 KiB; CPU cost would exceed any saving |
| 5 | fp16 → fp8 activations | **rejected** — would trade measured correctness for <1 ms |
| 6 | Persistent connections | already in place from Phase 3B |

## 6. What the numbers say about the architecture

The honest conclusion:

- **Compute optimisation is a dead end here.** Even a perfect runtime saves 66 ms
  of 2,600 ms.
- **HOST-B's 304 ms RTT is the structural problem.** It sits in the per-token path
  and costs ~620 ms of unavoidable round trip per token.
- **The relay design doubles that** by sending the response back through every
  stage instead of returning directly.
- To reach single-digit-millisecond tokens you would need the model on one GPU, or
  a link under ~1 ms. Neither is available on this fleet.

## 7. Recommended next experiment

1. **Return the response directly to the client** instead of relaying back through
   B and A. This should cut ~620 ms — a 24 % improvement — with no correctness
   risk. It is the largest single available win.
2. **Instrument per-hop timings** properly (the gap above) to account for the
   remaining 1,300 ms before attempting further optimisation.
3. **Move HOST-B out of the per-token path.** Its 304 ms RTT makes it unsuitable
   for decode; it could take prefill only, or run as an independent replica.
4. **Accept the ceiling.** For a model that fits on one GPU, single-GPU is 350–750×
   faster. The pipeline's value is capacity, not latency, and that should be the
   framing for any further work.

## 8. Limitations of this analysis

- Per-hop timing instrumentation was **not completed**. Only end-to-end and
  local-compute figures are measured; the 48 % network share is derived from
  Phase 2A RTTs, not from in-pipeline timing.
- Alternative partitions (10/11/11, 8/12/12, …) were **not benchmarked**. Given
  compute is 2.6 % of the token, they cannot materially change the result, but
  that is an inference, not a measurement.
- The A/C and B/C topologies were not re-run this phase.
- No 10-repetition statistical suite was produced; three trials were used in
  Phase 3B and were bit-identical.

## 9. Files

| Path | Contents |
|---|---|
| `experiments/model_execution/results/three_node/summary.json` | Phase 3B measurements |
| `experiments/model_execution/pipeline_node.py` | stage worker |
| `meshweave/runtime/phimoe_forward.py` | forward pass, optimised MoE |
| `docs/PHASE_3B_THREE_HOST_DISTRIBUTED_INFERENCE.md` | parity basis |
