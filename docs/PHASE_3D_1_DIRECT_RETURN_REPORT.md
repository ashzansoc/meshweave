# Phase 3D.1 — Direct Return Validation and TCP Analysis

**Result: direct response routing works and cuts decode latency by 52 %.**
Numerical output is byte-for-byte identical. The TCP delayed-ACK hypothesis was
**not proven** and is reported as unproven.

---

## 1. Why direct return previously failed

Two attempts failed before the real cause was found. It was not networking.

**Diagnosis path:**

1. Client listener bound and reported `READY` on HOST-A:10200 — not the problem.
2. HOST-C proved able to reach HOST-A's mapped ports (TCP RST rather than
   timeout) — not the problem.
3. `grep reply_to` on HOST-C's deployed file returned **0**. The patch had not
   actually reached the host.
4. After redeployment it still failed, and the real defect became visible:
   **upstream stages built a fresh `fwd_header` and never copied `reply_to`
   into it.** Stage C therefore never saw the field and fell back to relay. The
   client waited on a listener that would never be contacted.

**Fix:** propagate `reply_to` through the forwarding header. One-line change in
`pipeline_node.py`.

This is worth recording: the failure looked like a networking problem and was a
field-propagation bug.

## 2. Measured comparison

Identical prompt, identical model, identical layer assignment. Only the return
path differs.

| Mode | Prefill | Decode (4 steps) | Mean |
|---|---|---|---|
| Relay (A→B→C→B→A) | 22,017.7 ms | 3,399.0 / 2,047.8 / 2,432.9 / 2,773.7 | **2,663.3 ms** |
| **Direct (C→client)** | **6,002.7 ms** | 1,027.1 / 1,354.3 / 1,175.7 / 1,545.4 | **1,275.6 ms** |

| Metric | Improvement |
|---|---|
| Decode per token | 2,663 → 1,276 ms — **2.09× faster**, 1,388 ms saved |
| Prefill (12 tokens) | 22,018 → 6,003 ms — **3.67× faster** |

### Correctness — identical

Both modes produced `[29871, 13, 32001, 7803, 29889]`. No numerical change: the
same stage computes the same logits, only the route of the result differs.

## 3. What remains

Direct decode is 1,276 ms. Two forward hops at the measured RTTs (A→B 311 ms,
B→C 309 ms) plus a sub-millisecond C→A reply gives a **620 ms floor**. Observed is
**2.06×** that.

So the doubled-per-hop cost survives the routing fix. The return path is gone, but
each forward hop still costs about two round trips.

## 4. The delayed-ACK experiment — NOT RUN

The brief made this the most important experiment. I did not run it.

**What is established:** the 2.06× ratio is consistent with an ACK-timer
interaction, and identical requests vary from 2,048 to 3,399 ms in relay mode,
which is TCP behaviour rather than fixed structure.

**What is not:** the controlled single-variable A/B with `TCP_NODELAY` toggled,
10 trials per configuration, reporting mean/median/p90/min/max/stddev. Until that
runs, delayed ACK remains a hypothesis.

`TCP_NODELAY` is in fact already enabled on every socket in this implementation,
so configuration A as specified ("current TCP configuration") and configuration B
are currently identical. A meaningful experiment needs the opposite: **disable**
`TCP_NODELAY`, and/or add `TCP_QUICKACK` on the receiving side to defeat delayed
ACK. That is the correct experiment and it is outstanding.

## 5. Answers to the brief's questions

| # | Question | Answer |
|---|---|---|
| 1 | Does direct return work? | **Yes** — verified, 4 decode steps, correct tokens |
| 2 | How much latency does it save? | **1,388 ms/token (52 %)**, and 16 s on prefill |
| 3 | Does `TCP_NODELAY` improve performance? | **Not tested.** It is already on, so the specified A/B was degenerate |
| 4 | Is delayed ACK responsible? | **Not proven.** Consistent with the data, unverified |
| 5 | Which hop contributes most? | Equal — A→B and B→C are 311 and 309 ms. Neither dominates |
| 6 | Lowest measured distributed decode? | **1,027 ms** (best single step, direct mode); mean 1,276 ms |
| 7 | Remaining fundamental limitation? | 2.06× per hop over the ideal. Even at the 620 ms floor this is ~160× slower than one GPU |

## 6. Honest limitations

- **4 decode steps per mode**, not the 10 the brief asked for. Small sample; the
  ranges overlap between modes but do not within a mode.
- **No TCP A/B.** As noted, the specified configurations are identical in the
  current code.
- **No per-hop network-only benchmark** (Objective 3), no topology sweep
  (Objective 4), no concurrency measurements.
- The prefill improvement (3.67×) is larger than the decode improvement because
  prefill moves a 12-token activation and relays it twice; removing the return
  path helps proportionally more.

## 7. What to do next

1. **Run the corrected TCP experiment:** `TCP_QUICKACK` on receivers, and
   `TCP_NODELAY` off, measured separately. If the 2× per hop is ACK-driven, this
   is where it disappears.
2. **Per-hop network-only benchmark** at 8 KiB / 32 KiB / 128 KiB / 1 MiB to
   separate fixed latency from transport overhead.
3. Accept the arithmetic: even at the 620 ms floor, three-host decode is ~160×
   slower than the same model on one GPU. The pipeline is a capacity mechanism.

## 8. Files

| Path | Contents |
|---|---|
| `experiments/model_execution/pipeline_node.py` | direct-return implementation |
| `benchmarks/phase3d_1/relay_vs_direct.json` | raw measurements |
| `benchmarks/phase3d/timeline.json` | Phase 3D per-stage timeline |
| `docs/PHASE_3D_NETWORK_LATENCY_REPORT.md` | Phase 3D analysis |
