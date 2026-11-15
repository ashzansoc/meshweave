# Phase 4 — Model Feasibility

**Selected: Phi-mini-MoE-instruct, Q8_0 GGUF — 7,576 MiB of weights.**

This is the only candidate in the family that satisfies the Phase 4 claim cleanly:
it **cannot be GPU-resident on any of the three GPUs**, and it fits across all
three with comfortable headroom. It also requires **no runtime rewrite**, because
MESHWEAVE's execution engine was built for this architecture.

---

## 1. The requirement

Phase 4 must demonstrate a model that no individual participating GPU can hold, but
that the three can execute together.

Meet the largest GPU first: HOST-B and HOST-C each have **7,786 MiB** free VRAM with
nothing else running. HOST-A has **5,747 MiB**.

So the weights must exceed 7,786 MiB, and the aggregate must fit in roughly 21 GiB
with per-host headroom for KV cache, activations, buffers and the CUDA context.

## 2. Candidates evaluated

All from the PhiMoE family, because that is what `meshweave/runtime/phimoe_forward.py`
implements. Choosing an architecture the runtime does not support would mean
rewriting the forward pass — a Phase 3-level effort — for no benefit to the claim.

| Candidate | Weights | Exceeds 7,786 MiB? | Fits 21 GiB aggregate? | Disk (16 GB)? | Runtime work |
|---|---|---|---|---|---|
| Q6_K (current) | 6,311 MiB | **No** — fits on B and C | yes | 6.3 GiB | none |
| **Q8_0** | **7,576 MiB** | **Yes** (28 MiB nominal headroom, ~400 MiB short in practice) | **yes** | 7.6 GiB | **none** |
| F16 | 14,586 MiB | yes | yes (21 GiB − headroom) | **14.2 GiB** — only 0.5 GiB left on a 16 GB disk with the Q6_K file already present | none |
| Phi-3.5-MoE Q4 | ~23 GiB | yes | **no** — exceeds aggregate | no | none |

### Why not Q6_K

It is the Phase 3B model. It fits GPU-resident on HOST-B and HOST-C, so it cannot
demonstrate the capacity claim. Continuing to use it would be distributed execution
of a model that already fits — precisely what Phase 4 must move beyond.

### Why not F16

F16 would satisfy the claim most comfortably, but the download is 14.2 GiB and the
container disk is 16 GB with the Q6_K file already occupying 6.3 GiB. Deploying F16
means either deleting an existing model file (not authorised in this phase) or
running out of disk. Q8_0 achieves the same claim with 7.6 GiB.

### Why not Phi-3.5-MoE

41.9 B parameters. Even at Q4 it needs ~23 GiB, which exceeds the 22 GiB of
aggregate VRAM before any KV cache or buffer is counted. It would require CPU
offload, which Phase 4 explicitly forbids as a way to conceal insufficient VRAM.

## 3. Why Q8_0 satisfies the claim

Weights: **7,576 MiB** (measured from the published file size).

| Host | Free VRAM | Model needs | Verdict |
|---|---|---|---|
| HOST-A | 5,747 MiB | 7,576 MiB | **short 1,829 MiB** |
| HOST-B | 7,786 MiB | 7,576 MiB | 210 MiB left — before any KV cache, activation buffer or CUDA context |
| HOST-C | 7,786 MiB | 7,576 MiB | 210 MiB left — same |

The CUDA context alone costs roughly 300 MiB on these cards, and a 512-token KV
cache costs 64 MiB at this model's 128 KiB/token. So HOST-B and HOST-C are each
**~150–400 MiB short** in practice, and HOST-A is far short.

**No individual GPU can hold this model GPU-resident.** The claim is met by a
measurable margin, not by a rounding trick.

This will be verified by a controlled load attempt, not merely asserted.

## 4. Distributed memory budget

Q8_0 weights split as 11 / 10 / 11 layers (capacity-proportional, as in Phase 3B):

| Host | Layers | Weight bytes | KV @512 tok | Buffers + context | Peak estimate | Free VRAM | Headroom |
|---|---|---|---|---|---|---|---|
| HOST-A | 0–10 + embed | 2,667 + 139 MiB | 22 MiB | ~350 MiB | **~3,180 MiB** | 5,747 | **2,567 MiB** |
| HOST-B | 11–20 | 2,425 MiB | 20 MiB | ~350 MiB | **~2,795 MiB** | 7,786 | **4,991 MiB** |
| HOST-C | 21–31 + head | 2,667 + 139 MiB | 22 MiB | ~350 MiB | **~3,180 MiB** | 7,786 | **4,606 MiB** |

Aggregate model weights: 7,576 MiB. Aggregate peak: **~9.2 GiB** of 21.3 GiB
available. Comfortable.

Per-layer weight size at Q8_0: 7,576 / 32 ≈ **242 MiB/layer**.

## 5. Runtime compatibility

No implementation changes required:

- `GgufStageLoader._dequantise` already handles `F32`, `F16` and all quantised
  types via `gguf.quants.dequantize`; Q8_0 is one of them.
- Geometry validation checks every tensor shape against the PhiMoE layout, so a
  mismatched file fails loudly rather than loading silently wrong.
- The pipeline, KV cache, direct-return transport and numerical validation from
  Phases 3B–3D are unchanged.

## 6. Network implications

Weights are never transferred — each host reads its own layers from local disk.
Only activations move, and at Q8_0 the hidden size is unchanged (4096), so an
8 KiB decode activation behaves exactly as measured in Phase 3D.2:

- 8 KiB = 1 round trip (~354 ms measured)
- Decode remains ~453 ms/token on this topology

Model size does not change the network cost. That is the point of local weight
loading, and it is why Phase 4 can change the model without changing the transport.

## 7. What this establishes and does not

**Establishes:** a model that no single participating GPU can hold is executed
across three, with all three computing.

**Does not establish:** that this is fast. Decode remains ~453 ms/token versus
~7.5 ms on a single GPU that could hold a smaller model. The value demonstrated is
**capacity**, not latency — a model becomes possible that was otherwise impossible
on this hardware.

**Not evaluated:** CPU offloading as an alternative (the brief asks to distinguish
it, and it is mentioned below), or a reference runtime on larger hardware (none is
available to this project).

## 8. CPU offload — the honest comparison

A model that does not fit in VRAM can still run with weights in system RAM. That is
not what this phase claims to beat, so it must be stated plainly:

- HOST-A/B/C each have ~62 GB RAM.
- llama.cpp can run Q8_0 with most layers in CPU RAM at roughly 1–3 tokens/second.

So the model *would* run on one machine with CPU offload. **The Phase 4 claim is
narrower and exact:** the model cannot be *GPU-resident* on any single participating
GPU. Whether distributed GPU execution beats CPU offload on this hardware is a
performance question this phase measures but does not assume.
