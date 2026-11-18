# Framework compatibility matrix

Four candidate execution paths, scored against the environment we actually have:
two Turing cards (sm_75) with 6–8 GiB each, 226–228 ms apart, 8–15 Mbps, serving
a 0.49B parameter Qwen2.5 model.

Every cell carries its evidence: **[V]** verified in source, **[D]** documented by
the project, **[U]** unknown, **[M]** measured in this repository.

---

## 1. Summary

| | Native (MESHWEAVE) | Parallax | DeepSpeed-Inference | Dynamo |
|---|---|---|
| **Fits this environment** | yes | blocked (16 GB disk) | no | no |
| **Role in Float16** | default runtime | WAN candidate, pending spike | datacenter class A | production/regional serving |
| **Splits weights across nodes** | yes, by layer | yes, by layer | no (TP only) | no (workers are whole) |
| **Improves single-request tok/s** | n/a (baseline) | no | no | no |
| **Improves aggregate tok/s** | with replicas (not yet built) | with replicas | no | yes, with replicas |
| **Measured here** | 2.93 tok/s split, 19.02 single | not run | not run | not run |

---

## 2. Capability detail

### 2.1 Model support

| | Evidence |
|---|---|
| Native | Qwen2 dense only, config-driven. Refuses other architectures by name. **[V]** |
| Parallax | HF repo or local path; static list includes Qwen2.5 0.5B–72B, Qwen3, DeepSeek, GLM, Kimi, MiniMax, gpt-oss. No GGUF. **[V]** |
| DeepSpeed | HF transformers models via AutoTP; refuses several architectures. **[V]** |
| Dynamo | Whatever its backend supports; vLLM/SGLang/TRT-LLM. **[D]** |

### 2.2 Turing (sm_75)

| | Evidence |
|---|---|
| Native | Built and measured on 7.5 here. fp16 chosen automatically below 8.0. **[M]** |
| Parallax | Absent from its GPU database (only A100/4090/5090); absent from docs. Would use placeholder 50 TFLOPS / 600 GB/s. **[V]** |
| DeepSpeed | Not in a support matrix; open kernel issues at small hidden sizes (#8831, #8828, #8830). **[V]** |
| Dynamo | Platform lists Ampere/Ada/Hopper/Blackwell. vLLM claims 7.5+; TensorRT-LLM lists none; SGLang silent. **[V]** |

### 2.3 Multi-node communication

| | Pattern | Cost per token at 226 ms |
|---|---|
| Native | One activation per hop; last stage returns the token directly | ~310 ms measured **[M]** |
| Parallax | Protobuf forward over Lattica, per micro-batch | unmeasured; one round trip per hop minimum |
| DeepSpeed | One all-reduce **per layer** | ~5.4 s estimated from 24 × 226 ms **[V, reasoned]** |
| Dynamo | Request plane TCP; KV transfer only in disaggregated mode | n/a — it does not split a model |

### 2.4 Maturity

| | Evidence |
|---|---|
| Native | 229 tests, no external framework deps **[V]** |
| Parallax | v0.1.2 (2025-12-02), ~11 months without a release, no GPU CI, no mid-generation recovery (open #411) **[V]** |
| DeepSpeed | v0.20.0 (2026-11-12), mature training library, inference secondary **[V]** |
| Dynamo | v1.6.2 (2026-11-05), production-oriented, breaking changes in v1.6.0 **[V]** |

### 2.5 Security posture

| | Evidence |
|---|---|
| Native | Outbound-only agent, no shell endpoint, per-node and per-deployment tokens, unix-socket control **[V]** |
| Parallax | HTTP ports 3000–3002 have no authentication **[V]** |
| DeepSpeed | No network service of its own; inherits torch.distributed **[V]** |
| Dynamo | Frontend is an HTTP service; auth must be added externally **[V]** |

---

## 3. What each would cost to integrate

| | Effort | What Float16 would have to build |
|---|---|
| Native | already done | placement intelligence, replication, batching |
| Parallax | medium | isolated install, auth in front, a decision on who owns placement, a Turing verdict |
| DeepSpeed | high | per-rank process launching, a datacenter-only environment, kernel validation |
| Dynamo | medium | etcd+NATS for multi-node, a router policy decision, an auth layer |

---

## 4. Verdicts

**Native — default.** The only path with measured two-node execution on this
hardware, and the only one where Float16 owns placement. Its gap is intelligence
(equal splits, no replicas), not capability.

**Parallax — candidate, blocked on this instance class.** Architecturally the
closest match: layer-range sharding, small activations, HF models, Apache-2.0.
The spike could not complete because `sglang[all]` plus torch needs more disk
than a 16 GB Vast container has (8 GB of unpacking space alone; see
`docs/PARALLAX_SPIKE_RESULT.md`). Whether it runs on Turing is therefore still
**UNKNOWN** — not disproven. On a larger instance it remains the most plausible
WAN backend, and the sensible integration would be its single-node serving with
Float16 still choosing the split.

**DeepSpeed-Inference — datacenter only.** Tensor parallelism is the only
multi-node mode and it all-reduces per layer, which is fatal at 226 ms.
Inference has no pipeline parallelism (`RuntimeError` in source), so this is not a
tuning problem. Retained for environment class A, where NVLink makes TP the right
choice and PP unnecessary.

**Dynamo — production and regional serving.** It orchestrates; it does not split a
model or accelerate a single stream. Turing is outside its matrix, multi-node
needs etcd+NATS that assume a stable link, and disaggregation needs RDMA.
Retained for the serving tier once datacenter nodes exist.

---

## 5. What this means for the next milestone

The matrix says the work is not "integrate three frameworks". It is:

1. **Make placement intelligent** — evaluate candidate layer splits against
   measured GPU throughput and free VRAM instead of dividing evenly.
2. **Add replication** — when a model fits one node, run independent replicas and
   route, which is the only lever that raises aggregate throughput here.
3. **Separate the two metrics in every benchmark** — single-request tok/s and
   aggregate tok/s move in opposite directions on this hardware, and reporting
   one number hides that.
4. **Keep the framework doors open** — each is a backend behind a common
   interface, with an explicit `supported` predicate that currently returns false
   for this environment, so the architecture does not have to change when the
   hardware does.