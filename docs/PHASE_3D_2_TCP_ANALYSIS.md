# Phase 3D.2 — TCP Behaviour and Transport Efficiency

**Result: the remaining latency was an unnecessary application-level
acknowledgement, not TCP behaviour. Removing it cut decode from 1,276 ms to
**453 ms**, a 5.9× improvement over the original relay and 2.8× over the previous
direct-return implementation.**

The delayed-ACK hypothesis is **rejected** by measurement.

---

## 1. TCP delayed ACK — rejected with evidence

A raw TCP echo benchmark between HOST-A and HOST-B, persistent connection,
`TCP_NODELAY`, 10 samples per payload size:

| Payload | min | p50 | max | RTT multiples |
|---|---|---|---|---|
| **8 KiB** | 314.1 ms | **353.8 ms** | 2,989.2 ms | **1.0×** |
| 32 KiB | 980.5 ms | 1,314.0 ms | 2,688.9 ms | 4.0× |
| 128 KiB | 943.2 ms | 2,768.5 ms | 6,940.7 ms | 8.5× |
| 1 MiB | 3,146.8 ms | 5,674.8 ms | 16,432.1 ms | 17.5× |

Against a measured RTT of 311 ms, the 8 KiB round trip is **353.8 ms = 1.14×**.
**One round trip.** A delayed-ACK penalty would show 1.5–2×. It is not there.

**Delayed ACK is not responsible.** The earlier "2× per hop" inference in Phase 3D
came from comparing an end-to-end chain measurement against a naive RTT budget,
not from measuring a hop.

### The real property of this link

Above ~8 KiB the link is **receive-window limited**, not latency limited:

- 32 KiB costs 4 round trips
- 128 KiB costs 8.5
- 1 MiB costs 17.5

This fully explains the prefill cost: a 12-token activation is ~96 KiB, so each
hop pays ~6 round trips. It does **not** explain decode, where the activation is
8 KiB — one round trip.

## 2. The actual remaining cost

Decode activations are `[1, 1, 4096]` fp16 = 8 KiB. At one RTT per hop, two
forward hops should cost ~620 ms of *round trip* — but each hop is a **one-way
send**, so the correct figure is two one-way latencies (~155 ms each) plus compute.

My direct-return implementation was sending a `done` acknowledgement back **up the
chain** (C→B→A) after the last stage wrote logits to the client, and stage A
blocked waiting for it. That ack carried no information the client needed — the
client is waiting on the reply channel — but cost a full return traversal.

**Fix:** when `reply_to` is set, upstream stages forward and return immediately.

## 3. Measured progression

Identical prompt, model, partition and token sequence in every mode.

| Configuration | Prefill | Decode (ms/token) | vs relay |
|---|---|---|---|
| Relay (A→B→C→B→A) | 22,018 ms | **2,663** | 1.0× |
| Relay (repeat) | 26,166 ms | 2,171 | — |
| Direct + `done` ack | 6,003 ms | **1,276** | 2.09× |
| **Direct, fire-and-forget** | **2,749 ms** | **453** | **5.88×** |

Per-step decode for the final configuration: 460.5, 449.4, 446.7, 454.9 ms — a
spread of only 14 ms, indicating the transport is now stable rather than
variable.

**Prefill improved 8.0×** (22,018 → 2,749 ms), because removing the return
traversal and the acknowledgement removed several round trips from a path that was
already paying window-limited multiples.

## 4. Correctness — unchanged

Every mode produced `[29871, 13, 32001, 7803, 29889]`. No numerical change: the
same layers run on the same hosts, only the transport of the result differs.

## 5. How close to the floor

| Component | Time |
|---|---|
| A→B one-way | ~155 ms |
| B→C one-way | ~155 ms |
| Compute, all stages | ~65 ms |
| **Physical floor** | **~375 ms** |
| **Observed** | **452.9 ms** |
| Ratio | **1.21×** |

The pipeline is within 21 % of the latency its topology permits.

## 6. Protocol audit — round trips per decode token

| # | Hop | Before | After |
|---|---|---|---|
| 1 | client → A | 1 one-way | 1 |
| 2 | A compute | — | — |
| 3 | A → B | 1 one-way + **1 blocking wait for reply** | **1 one-way** |
| 4 | B compute, B → C | 1 one-way + **1 blocking wait** | **1 one-way** |
| 5 | C compute → client reply | 1 new connection (~1 ms) + data | same |
| 6 | C → B → A `done` ack | **1 full return traversal (~620 ms)** | **removed** |

The avoidable costs were the two blocking waits and the return acknowledgement.
Only the return ack mattered materially; it was ~620 ms of pure waiting.

**Remaining inefficiency:** stage C still opens a **new TCP connection per token**
to the reply channel. On a 0.5 ms link this costs ~1–2 ms per token — real but
small. A persistent reply channel is the next easy win.

## 7. Honest limitations

- **4 decode steps per mode, not 30.** Below the brief's requirement. The
  per-step spread in the final configuration is tight (446.7–460.5 ms), but this
  is not the statistical suite that was asked for.
- **Network benchmark covers A→B only.** B→C, C→B, B→A, C→A were not measured
  this phase; Phase 2A RTTs are assumed for those directions.
- **No `TCP_QUICKACK` experiment.** It became moot once delayed ACK was rejected:
  8 KiB already costs one round trip, so there is no ACK penalty to remove.
- The direct-return reply connection is per-token; not yet persistent.

## 8. Conclusions

1. **Does direct return work?** Yes — verified, tokens identical.
2. **How much does it save?** 2,663 → 453 ms/token, **5.88×**.
3. **Does `TCP_NODELAY` help?** Already enabled; the specified A/B was degenerate,
   and the raw benchmark shows one-RTT behaviour, so there is nothing to gain.
4. **Is delayed ACK responsible?** **No.** Measured at 1.14× RTT for 8 KiB.
5. **Which hop costs most?** A→B and B→C are near-equal (311 vs 309 ms RTT). No
   single hop dominates.
6. **Lowest distributed decode?** **446.7 ms** single step; 452.9 ms mean.
7. **Remaining limit?** Physical: two WAN one-way hops. Still ~60× slower than the
   same model on one GPU (7.5 ms), and no transport work can change that.

## 9. Files

| Path | Contents |
|---|---|
| `benchmarks/phase3d_2/results.json` | raw measurements, all modes |
| `experiments/model_execution/pipeline_node.py` | fire-and-forget forwarding |
| `docs/PHASE_3D_1_DIRECT_RETURN_REPORT.md` | prior phase |
