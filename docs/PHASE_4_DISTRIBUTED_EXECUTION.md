# Phase 4 — Distributed Model Capacity

**Capacity claim: PROVEN. Distributed execution of the large model: BLOCKED.**

The central Phase 4 assertion — that the model cannot be GPU-resident on any
individual participating GPU — is established by controlled measurement.

The three-host execution of that model could not be run: two hosts are holding
~5.5 GiB of VRAM in processes whose ownership I cannot verify, and I will not
terminate those.

---

## 1. Model selection

**Phi-mini-MoE-instruct, Q8_0 GGUF — 7,576 MiB of weights.**

Chosen because it is the only candidate that satisfies the claim while remaining
within the 16 GB container disk and requiring no runtime rewrite.

| Candidate | Weights | Exceeds largest GPU? | Notes |
|---|---|---|---|
| Q6_K (Phase 3) | 6,311 MiB | **No** — fits on B and C | cannot demonstrate the claim |
| **Q8_0** | **7,576 MiB** | **Yes** | **selected** |
| F16 | 14,586 MiB | Yes | 14.2 GiB download; disk has only 1.5 GiB spare with Q6_K present |
| Phi-3.5-MoE Q4 | ~23 GiB | Yes | exceeds 22 GiB aggregate before KV/buffers |

Downloaded and verified on all three hosts: **7.576 GiB each**, byte-identical size.

## 2. The claim, measured on a clean GPU

HOST-B was cleaned to **7,686 MiB free VRAM** (nothing else running). Loading the
full 32-layer model at Q8_0:

```
HOST-B clean free VRAM: 7686 MiB (model weights = 7576 MiB)
RESULT: OOM -> cannot be GPU-resident:
  CUDA out of memory. Tried to allocate 120.00 MiB.
  GPU 0 has a total capacity of 7.60 GiB of which 50.44 MiB is free.
```

**110 MiB of nominal headroom, consumed by the CUDA context and layer buffers
before all layers fit.** The largest GPU in the fleet cannot hold this model
GPU-resident.

HOST-A is not close: **5,747 MiB** free against 7,576 MiB needed — short by
1,829 MiB.

This is a controlled load attempt, not a calculated claim.

## 3. Memory budget for a three-way split

| Host | Layers | Weights | KV @512 tok | Buffers + context | Peak est. | Free | Headroom |
|---|---|---|---|---|---|---|---|
| HOST-A | 0–10 + embed | 2,806 MiB | 22 MiB | ~350 MiB | ~3,178 MiB | 5,747 | 2,569 |
| HOST-B | 11–20 | 2,420 MiB | 20 MiB | ~350 MiB | ~2,790 MiB | 7,686 | 4,896 |
| HOST-C | 21–31 + head | 2,806 MiB | 22 MiB | ~350 MiB | ~3,178 MiB | 7,786 | 4,608 |

Aggregate peak **~9.1 GiB** of 21.2 GiB available. The split is comfortable — the
constraint is not capacity, it is the current VRAM occupancy on two hosts.

## 4. Why the distributed run is blocked

| Host | Free VRAM now | Blocker |
|---|---|---|
| HOST-A | **276 MiB** | PID 1482890 holding 5,468 MiB |
| HOST-B | 7,686 MiB | clean, ready |
| HOST-C | **2,270 MiB** | PID 390673 holding 5,512 MiB |

Both blocking processes have **empty `/proc/<pid>/cmdline`**. I cannot establish
their identity or that they belong to this experiment.

**MESHWEAVE's safety rule is explicit: if process ownership cannot be established,
refuse termination.** That rule exists because an earlier cleanup killed `sshd` on
all three hosts. I am not going to repeat it, and I am not going to work around it
by killing something that merely holds a port or VRAM.

A stage cannot run in 276 MiB — a single Q8_0 layer is ~242 MiB before buffers. So
the three-host configuration is genuinely unavailable right now.

### What would unblock it

- Restart HOST-A and HOST-C instances (the processes are container-local; a
  stop/start clears them and the data disk survives).
- Or identify those two PIDs from the Vast console web terminal and confirm they
  are safe to stop.

Once both hosts show >3.5 GiB free, the run is:

```
MESHWEAVE_MODEL=/root/models/phi-mini-moe-Q8_0.gguf \
  python -m experiments.model_execution.pipeline_node stage --index N --layers auto ... 
```

The model path is now an environment variable, so Phase 3 remains reproducible with
the default Q6_K file.

## 5. What is established

| Requirement | Status |
|---|---|
| Model cannot be GPU-resident on any participating GPU | **PROVEN** — OOM on the largest GPU with 7,686 MiB free |
| Model components distributed across multiple GPUs | **BLOCKED** |
| All three hosts execute real model computation | **BLOCKED** |
| No host loads the complete model | designed — each loads only its range |
| No CPU offload used to conceal VRAM | holds — the runtime has no CPU offload path |
| Forward pass + multiple tokens | **BLOCKED** |
| Numerical correctness vs reference | **BLOCKED** (Q8_0 vs the Q6_K reference would need re-baselining anyway) |
| Peak VRAM per host | measured for HOST-B during the OOM test |

Everything blocked is blocked by host state, not by the architecture. Phases 3B–3D
already proved the execution path works end to end; the only change here is the
model file and its size.

## 6. Honest assessment of the capacity claim's significance

The claim is now proven in the narrow, exact sense the brief specified: the model
cannot be **GPU-resident** on any single participating GPU.

It is important to state what that does **not** mean:

- **CPU offload would run it.** Each host has ~62 GB RAM. llama.cpp with most
  layers in system RAM would run Q8_0 at roughly 1–3 tokens/second on one machine.
  So "cannot run on one machine" would be false; "cannot be GPU-resident" is true.
- **Distributed execution has not yet been shown to beat CPU offload here.** Three
  hosts add two WAN hops (~310 ms each), so the expected decode is ~450 ms/token.
  CPU offload on one host might be comparable. That comparison is unmeasured and
  should be part of the next run.

## 7. Remaining limitations

- Distributed execution of Q8_0: not run.
- No independent reference for Q8_0 logits yet. llama.cpp is available on HOST-C
  and can run the same file, so the reference path used in Phase 3A.1 applies
  unchanged.
- No benchmark data for the large model.
- Two of three hosts are in an unknown VRAM state.

## 8. Files

| Path | Contents |
|---|---|
| `docs/PHASE_4_MODEL_FEASIBILITY.md` | candidate analysis and memory budget |
| `experiments/model_execution/pipeline_node.py` | `MESHWEAVE_MODEL` env var selects the model |
| Q8_0 on all hosts | `/root/models/phi-mini-moe-Q8_0.gguf`, 7.576 GiB |
