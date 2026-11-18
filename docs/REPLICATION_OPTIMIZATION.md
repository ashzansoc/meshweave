# Replication optimization

What was made faster, and what it cost.

**Baseline:** 14.97 aggregate TPS at concurrency 4.
**Result:** 19.75 at concurrency 4, 33.17 at concurrency 8.
Details and raw data in `docs/PHASE_3_BENCHMARK_RESULTS.md`.

---

## 1. Several replicas per node

A node may now host more than one full copy of the model. The count is chosen
from that node's **free VRAM**: `floor(free × 0.85 / (weights + KV reserve))`,
capped at what the deployment asks for.

For this fleet a 0.5B replica holds 1.2 GiB, so a 6–8 GiB card hosts two
comfortably. Both nodes report the true per-replica figure from the process.

`replicas_per_node` is opt-in and defaults to 1, so no existing plan changed by
accident.

### Why it helps here

The baseline measured decode at ~40 ms per token with the GPU at 1–2 %
utilisation. The GPU is not the constraint; the number of sequences a machine can
drive through its per-token path is. A second process on the same card therefore
buys throughput at the cost of VRAM, not compute.

### What it costs

Single-request throughput at concurrency 4 fell from 22.3 to 17.7 tok/s, because
two processes now share one card. Aggregate rose. That trade is stated rather
than hidden — the brief forbids hiding a latency increase behind a throughput
increase.

## 2. Polling, again

The work poll was 1 s. A request submitted just after a poll waited ~1 s before
any agent saw it, which dominated short requests: four parallel requests of
~0.9 s took 2.5 s.

At 250 ms the waiting cost falls to a quarter. The poll is one cheap GET per agent
per interval; with two nodes that is eight requests per second against the
coordinator.

This is the second time the same defect has appeared (Phase 2 improved it from
5 s to 1 s). The lesson is recorded here: **on a polled control plane, the poll
interval is a latency floor for every request**, and it is invisible in
single-request timings because those start after dispatch.

## 3. Identity: deployment#node#replica

Adding a second replica per node forced stage identity to be explicit. It is now
`deployment#node#replica`, and it must include all three: without the node, replica
0 on two machines shares one status entry and one load counter, and routing sees
no healthy replica at all.

Each replica gets its own control socket,
`/run/meshweave/control-<deployment>-<replica>.sock`, because a fixed path makes
the second replica on a node collide with the first.

## 4. What did not help, and was not adopted

* **More VRAM headroom** — 0.85 is already conservative; raising it risks OOM
  under longer contexts for no measured gain.
* **Weighting routing by measured GPU speed** — the two cards are close (38.8 vs
  41.0 ms per token); least-in-flight already balances them.
* **Prefetching weights** — weights are already resident for the deployment's
  lifetime; there is no reload between requests.

## 5. Known limits

* Replicas are independent processes, so there is no cross-replica KV reuse.
* Each process loads the checkpoint separately into VRAM (1.2 GiB each).
* The control path still sits between submission and execution; batching inside a
  replica is the way to remove it rather than shorten it.
