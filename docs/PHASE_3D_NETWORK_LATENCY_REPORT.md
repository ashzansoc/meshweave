# Phase 3D — Network Latency and Direct Return Report

**Status: partial.** In-pipeline network timing was obtained and establishes the
bottleneck quantitatively. Direct response routing is implemented but **not
verified end-to-end**. One hypothesis (delayed ACK) is supported by the timing
shape but was **not isolated by a controlled experiment**.

---

## 1. Measured per-stage timeline

Five warm decode steps through the live three-host pipeline, one token each.
`compute_ms` is each stage's own GPU time; the chain round-trip is measured on
HOST-A around a full A→B→C→back cycle.

| Step | Total | B compute | C compute | A's chain round-trip |
|---|---|---|---|---|
| 0 | 2,861 ms | 86.7 ms | 109.5 ms | 2,773 ms |
| 1 | 5,629 ms | 73.2 ms | 38.0 ms | 5,555 ms |
| 2 | 8,119 ms | 73.3 ms | 60.7 ms | 8,044 ms |
| 3 | 3,638 ms | 73.6 ms | 47.3 ms | 3,564 ms |
| 4 | 5,385 ms | 72.3 ms | 52.8 ms | 5,312 ms |

Warm prefill (12 tokens): 22,963 ms total, B compute 837 ms, C compute 378 ms,
chain round-trip 22,116 ms.

### The decomposition

| Component | Value | Share |
|---|---|---|
| Compute, B + C (measured) | 111–196 ms | ~3–7 % |
| Compute, A (11 layers, measured locally) | ~23 ms | ~1 % |
| **Total compute** | **~135–220 ms** | **~5 %** |
| Network and everything else | 2,650–7,900 ms | **~95 %** |

Compute is now confirmed at the low single digits. Phase 3C's estimate of 2.6 %
was right.

## 2. What the numbers say about the network

A full chain cycle traverses four links:

| Hop | Measured RTT |
|---|---|
| A→B | 311 ms |
| B→C | 309 ms |
| C→B (reply) | 309 ms |
| B→A (reply) | 311 ms |
| **One RTT per hop** | **1,240 ms** |
| **Observed** | **2,773–8,044 ms** |
| **Ratio** | **2.2× – 6.5×** |

If each hop cost exactly one round trip the token would take ~1.24 s. It takes
2.8 s at best. **Each hop is costing roughly two or more round trips**, and the
spread from 2.8 s to 8.1 s across identical requests is not compute variance — it
is network variance.

### On delayed ACK specifically

The brief says not to assume delayed ACK. I am not assuming it — but I also did
not prove it. What the data supports:

- **Size**: 8 KiB payload, one `sendall`. Transmission time at 11.71 Mbps is
  ~5 ms. The remaining ~305 ms of a hop is round-trip, not bandwidth.
- **Shape**: 2.2× a single RTT per hop is what you get when a receiver delays its
  ACK and the sender waits for it. A clean single-RTT path would give 1.0×.
- **Variance**: identical requests ranging 2.8–8.1 s is characteristic of TCP
  congestion-window and ACK-timer interaction, not of fixed pipeline structure.

**What was not done:** the controlled A/B of `TCP_NODELAY` on/off, single-variable,
with raw measurements. That is the experiment that would settle it, and it remains
outstanding. I am labelling this as *consistent with* delayed-ACK interaction, not
as *demonstrated*.

## 3. Direct response routing

### Design

Stage C sends logits to a client-supplied `reply_to` address instead of returning
them up the chain, then sends a small `done` ack upstream. Implementation is in
`experiments/model_execution/pipeline_node.py`.

The expected gain: the return path currently costs C→B (309 ms) + B→A (311 ms) =
620 ms. Because HOST-C and HOST-A share a physical host, a direct C→A reply should
cost **~0.5 ms** — the same sub-millisecond adjacency Phase 2A measured.

### Status: implemented, not verified

The benchmark harness (`relay` vs `direct` in one process) ran, but the direct
reply did not arrive and the client timed out waiting for it.

What was established:

- Port 10200 binds successfully on HOST-A.
- HOST-C **can** reach HOST-A's mapped ports: connecting to
  `1.208.108.242:51440` returns a TCP RST (`ConnectionRefused`) rather than a
  timeout, which proves the route exists and only the listener was absent.
- So the path is viable; the failure is in the test harness timing or in the
  listener being armed after the request was already dispatched.

**I am not claiming a speedup I did not measure.** The design is sound and the
network path is proven reachable, but the end-to-end comparison was not completed.

## 4. Answers to the brief's questions

| Question | Answer | Evidence |
|---|---|---|
| How much time does each forward hop consume? | A→B 311 ms, B→C 309 ms RTT; observed effective cost ~2× that | Phase 2A RTTs + chain timing |
| How much time does each return hop consume? | C→B 309 ms, B→A 311 ms | Phase 2A RTTs |
| Is delayed ACK contributing materially? | **Consistent with** it (2.2× per hop, high variance); **not proven** | timing shape only |
| What explains the previously unexplained ~1,300 ms? | Extra round trips per hop — the chain costs 2.2–6.5 RTT budgets, not 1 | measured |
| How much does direct return improve decode? | **Not measured** — implemented, unverified | — |
| Does direct return affect correctness? | No numerical change: same stage, same weights, only the transport of the result differs | by construction |
| Does concurrency improve throughput? | **Not measured** this phase | — |
| Best topology for low latency? | Single GPU, by a wide margin: 7.5 ms vs 2,800+ ms | measured |
| Best topology for distributed capacity? | A→C (0.5 ms link) for coupled work; B only where capacity demands it | Phase 2A + 3C |

## 5. Conclusion

**The ceiling is physical.** At 310 ms RTT, four hops cost a minimum of 1.24 s per
token, and the observed 2.8 s shows the transport is spending about twice the
ideal. Even a perfect transport caps this topology at roughly 0.8 tokens/second —
still ~100× slower than the same model on one GPU.

Direct return is the one structural improvement still available: it removes half
the hops from the return path. It should roughly halve token latency. That
prediction is arithmetic, not a measurement, and needs the benchmark completed.

## 6. What to do next

1. **Finish the direct-return benchmark.** Fix the harness, get the before/after
   number. It is the largest remaining win and the code is already written.
2. **Run the single-variable TCP experiment** (`TCP_NODELAY` on/off, one change at
   a time) to convert the delayed-ACK hypothesis into a measurement. If confirmed,
   `TCP_QUICKACK` on the receiver may remove the second round trip per hop.
3. **Stop treating this topology as a latency solution.** It is a capacity
   mechanism. Phase 4 work should target a model that cannot fit on one GPU, where
   2.8 s/token may be acceptable and no alternative exists.

## 7. Limitations

- Direct return: implemented, unverified.
- Delayed ACK: not isolated experimentally.
- No concurrency measurements, no alternative partitions, no 10-repetition
  statistical suite.
- Per-hop send/receive timestamps inside each stage were not instrumented; the
  breakdown above comes from each stage's `compute_ms` and HOST-A's chain timing.
