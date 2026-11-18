# NVIDIA Dynamo — technical research

**Repository:** https://github.com/ai-dynamo/dynamo · Apache-2.0
**Latest release:** v1.6.2, 2026-11-05 (docs default v1.6.1)
**Assessed:** 2026-11-18

Evidence tags: **VERIFIED** (read in source or the official docs),
**DOCUMENTED** (vendor claim), **UNKNOWN**.

The important distinction throughout: **Dynamo the platform** has a support
matrix, and **each inference backend** has its own. They do not agree.

---

## 1. Components

**VERIFIED.** Dynamo is an orchestration layer above inference engines, not an
engine.

| Component | Role | Required? |
|---|---|---|
| **Frontend** | OpenAI-compatible HTTP (8000); tokenises, applies the chat template, validates, routes | required for the API path |
| **Router** | Worker selection: `kv`, `round-robin`, `random`, `power-of-two`, `least-loaded`, `device-aware-weighted`, `direct` | functionally required; the *policy* is optional (`direct` means someone else chose) |
| **Workers** | Prefill and/or decode pools wrapping an engine | required |
| **KVBM** | KV offload GPU→CPU→SSD→remote | optional, 🚧 on vLLM |
| **Planner** | SLA autoscaler that sets replica counts | optional, needs an external reconciler |
| **ModelExpress / Grove** | Weight streaming; Kubernetes gang scheduling | optional |

Disaggregated prefill/decode is optional — in aggregated mode one worker does
both.

## 2. Deployment without Kubernetes

**VERIFIED.** PyPI install (`uv pip install "ai-dynamo[vllm]"`), with host
requirements of NVIDIA driver 575.51+/580+, Ubuntu 22.04/24.04, Python 3.10/3.12.
Containers are published (`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0`).

Entry points are `python3 -m dynamo.frontend --http-port 8000` and
`python3 -m dynamo.vllm --model …`.

**etcd and NATS are not mandatory for a single machine** — `--discovery-backend
file` works, with events falling back to ZMQ. They are required for multi-node,
and NATS needs JetStream (`-js`).

## 3. GPU and backend compatibility — the part that matters here

**VERIFIED.** The support matrix lists **Ampere, Ada Lovelace, Hopper and
Blackwell**. **Turing (sm_75) is not listed.**

The backends disagree with each other:

| Backend | Turing support |
|---|---|
| vLLM 0.31.0 | States compute capability 7.5 or higher (T4, RTX 20xx) — **may work** |
| TensorRT-LLM 1.3.0rc29 | Lists SM80/86, Ada, Hopper, Blackwell — **no Turing** |
| SGLang 0.5.21 | No statement either way — **UNKNOWN** |

So a vLLM worker might run on a 2070 SUPER while Dynamo's own CUDA-13 containers,
NIXL paths and KVBM are unvalidated there. That distinction is the honest answer:
the platform does not claim Turing, but one of its backends does.

## 4. Transport

**VERIFIED.** The request plane defaults to `tcp` (`DYN_REQUEST_PLANE`), with
`nats` optional; the event plane is `zmq` or `nats`. NIXL carries GPU-to-GPU KV
transfers in disaggregated mode over NVLink, InfiniBand/UCX or PCIe. The install
guide states that multi-node disaggregated serving needs an RDMA-capable
interconnect. UCX can use TCP, but no TCP fallback or WAN/NAT behaviour is
documented.

**Assessment: built for a datacenter fabric, not for the public internet.**
High-latency WAN is unsupported by design — etcd leases, NATS and KV-event timing
all assume a fast, stable link.

## 5. Effect on the two metrics

**No single-request decode improvement. VERIFIED by absence.**
Every published figure is aggregate: "7× higher throughput per GPU", "2× faster
TTFT", "750× higher throughput". KV-aware routing avoids redundant *prefill* work;
it does not raise one stream's decode tokens/sec, which is set by the engine and
the GPU. Dynamo describes itself as sitting above the engines.

For a deployment serving one request at a time, Dynamo adds a hop and a router
decision and cannot make the tokens arrive faster.

## 6. API surface

**VERIFIED.** SSE streaming; `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/responses`, `/v1/models`, image and video endpoints;
Anthropic Messages (experimental); KServe gRPC v2; OpenAPI at `/openapi.json`.

## 7. Maturity

v1.0 GA in March 2025, described as production-ready and bundled into NIM and
NVIDIA AI Enterprise. v1.6.0 introduced breaking changes: vLLM services renamed
to `worker`/`prefill`/`decode`, stricter multi-node validation,
AIConfigurator→AISimulate, TensorRT-LLM AutoDeploy removed, several router
scoring flags deprecated. An N-2 mixed-version window is supported.

## 8. What integration would mean for Float16

**An addition, not a replacement.** Float16 keeps discovery, placement and its
pipeline runtime. Dynamo would contribute an OpenAI-compatible front end, routing
and optional KV-aware dispatch. Two shapes:

* Float16's coordinator calls a Dynamo frontend; or
* Float16's runtime registers as a Dynamo worker via `@dynamo_worker` /
  `register_model` (documented Python worker API).

If Float16 already places requests, the router should run in `direct` mode to
avoid two policies competing.

## 9. Assessment for this environment

**Not useful for two Turing GPUs 226 ms apart.** Turing is outside the platform's
matrix; CUDA-13 and NIXL are unvalidated there; multi-node needs etcd and NATS
that assume a stable low-latency link; disaggregation is impossible at 8–15 Mbps;
and it cannot improve single-stream decode.

**Retained for the production and regional tiers.** Once the fleet contains
datacenter-class nodes (environment A), Dynamo's routing, KV-aware dispatch and
OpenAI-compatible front end are a sensible serving layer, and its Python worker
API lets Float16's runtime sit underneath without surrendering placement.

**Minimum viable integration, if attempted:** aggregated mode only — vLLM 0.31
directly on the Turing host (it claims sm_75 support), one frontend per node with
`--discovery-backend file`, `--router-mode round-robin`, no KVBM, no planner, no
disaggregation. In practice that is close to the thin HTTP layer Float16 already
has, which is why it is not worth doing on this hardware.

## 10. Open questions

* Whether Dynamo's Python and Rust layers import and run on an sm_75 host.
* WAN behaviour of etcd and NATS at 226 ms (untested, undocumented).
* Whether a vLLM worker on Turing performs acceptably under Dynamo's request
  plane — the backend may run while the platform's assumptions do not hold.
* Real cost of TCP-only KV transfer, if disaggregation is ever attempted.