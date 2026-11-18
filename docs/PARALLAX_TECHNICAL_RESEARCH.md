# Parallax — technical research

**Repository:** https://github.com/GradientHQ/parallax · Apache-2.0 · ~1.4k stars
**Latest release:** v0.1.2, 2025-12-02 (commits continue to 2026-07-01)
**Assessed:** 2026-11-18

Evidence is tagged **VERIFIED** (read in source at the cited path),
**DOCUMENTED** (stated in the project's own docs), **UNKNOWN** (no evidence
either way).

---

## 1. Architecture

**Pipeline parallelism only. VERIFIED.**
Each node serves a contiguous, inclusive layer range
(`src/scheduling/layer_allocation.py`: `allocate(node, start_layer, end_layer)`).
Allocation is water-filling proportional to node compute with a min-heap over
per-layer KV load and a `rebalance_threshold` of 0.25. Nodes take
`--start-layer` / `--end-layer` (`src/parallax/server/server_args.py`).

Tensor and data parallelism exist only *inside* a node (`--tp-size`, `--dp-size`).
There is no cross-node tensor or expert parallelism.

**Peer communication. VERIFIED.**
Node-to-node forward is a protobuf RPC (`src/parallax/p2p/proto/forward.proto`,
`rpc_pp_forward`, `rpc_abort`, `chat_completion`) over **Lattica**, a libp2p-style
layer pinned at `lattica==1.0.21`. Nodes announce in a DHT
(`dht_prefix="gradient"`) and bootstrap through Gradient-operated relays
(`PUBLIC_INITIAL_PEERS`, `PUBLIC_RELAY_SERVERS` in `src/parallax/cli.py`).
Executor↔P2P is ZMQ.

Ports: scheduler 3001, node chat 3002, executor HTTP 3000; Lattica TCP/UDP
default to `0` (ephemeral).

**Coordinator. VERIFIED.**
There is a scheduler (`src/scheduling/scheduler.py`) that joins nodes, heartbeats,
allocates layers and dispatches requests. It is not an open protocol — nodes join
with `parallax join -s <peer-id>` or local discovery. Manual layer assignment
exists (`Node.manual_layer_assignment`) and suppresses global rebalance, so an
external coordinator can pin ranges, but node join is still mediated by Lattica
and its scheduler.

## 2. Hardware

**VERIFIED.** `src/parallax/server/server_info.py` carries a GPU database
containing only **A100-40/80G, RTX 5090 and RTX 4090**. Anything else falls back
to a placeholder `{tflops_fp16: 50.0, bandwidth_gbps: 600.0}` — which is what a
Turing card would get. Non-CUDA, non-Apple silicon raises `NotImplementedError`.

**DOCUMENTED.** `docs/user_guide/install.md` lists Blackwell, Ampere and Hopper
(RTX 50 / RTX 40 / A100 / H100) plus DGX Spark. **Turing is not mentioned
anywhere in the repository.**

**UNKNOWN.** No minimum-VRAM figure is stated. No NVLink or fast-interconnect
requirement is documented. WAN behaviour is undocumented, though relay-based NAT
traversal is a documented mode.

**Relevant to us:** the only memory knobs are `--param-mem-ratio` (0.65) and
`--kvcache-mem-ratio` (0.25).

## 3. Models and precision

**VERIFIED.** Model is an HF repo id or local path (`--model-path`). A static
supported list (`src/backend/server/static_config.py`) includes
`Qwen/Qwen2.5-0.5B-Instruct` through `Qwen2.5-72B-Instruct`, Qwen3, DeepSeek,
GLM, Kimi, MiniMax, gpt-oss, Step.

**No GGUF support** — zero references in the source. Dtypes are
`float16 | bfloat16 | float32`; quantisation exists only on the MLX path. The GPU
path loads full-precision safetensors.

## 4. Maturity

**VERIFIED.** v0.1.2 (2025-12-02), preceded by v0.1.1 (2025-11-26) and v0.1.0.
Roughly eleven months without a tagged release despite continued commits.

CI runs on ubuntu-latest and macOS with `pytest` and coverage, around fifty test
files covering the scheduler, layer allocation, KV cache, shard loader and model
variants. **There is no GPU end-to-end or performance job.**

Open issues relevant to production: #305 fault tolerance and recovery, #411
continue generation after a worker fails, #342 removing nodes and resharding,
#486 hot-swapping models.

## 5. KV cache and batching

**VERIFIED.** Paged KV plus a radix/prefix cache
(`src/parallax/server/cache/`, `block_radix_cache.py`). Continuous batching via
`form_batch()` with `max_batch_size`, `max_num_tokens_per_batch`, a micro-batch
ratio and chunked prefill in `src/parallax/server/scheduler.py`. The README's
claim that continuous batching is for Mac understates the GPU path.

## 6. Fault tolerance

**VERIFIED.** Heartbeat timeout is 30 s
(`src/scheduling/scheduler.py`: `heartbeat_timeout: float = 30.0`), after which
the node is detached and the *whole* fixed pipeline is invalidated
("if any member leaves, the whole fixed pipeline becomes invalid") before a
global rebalance. In-flight requests are **not** resumed (#411 is open).

## 7. Licensing

Apache-2.0.

## 8. Security observations

Parallax's HTTP ports (3000, 3001, 3002) have **no authentication** in the
source. Any node that publishes them exposes inference and cluster control to
whoever can reach them. A deployment behind it needs its own auth layer, and the
ports should not be published.

It also performs a version check against `chatbe-dev.gradient.network` unless
`-u` is passed, and WAN mode depends on Gradient-operated relays.

---

## 9. Assessment for this environment

Two Turing cards (sm_75), 6–8 GiB, 226 ms apart at 8–15 Mbps, Qwen2.5-0.5B.

**Structurally aligned.** Layer-range sharding is exactly what our runtime does,
and the payload is small: hidden 896 → ~1.8 KiB per stage per token at bf16,
trivially within 8–15 Mbps. A 12/12 split needs ~0.3 GB of weights per node.

**Three things argue against adopting it now:**

1. **Turing is undocumented and unlisted.** Its GPU database has no entry, so
   the scheduler would plan with placeholder 50 TFLOPS / 600 GB/s numbers. Its
   stack (SGLang 0.5.x, flashinfer kernels) may or may not run on sm_75. This is
   the one genuinely open question, and it is answerable only by running it.
2. **Maturity.** v0.0.1 through v0.1.2 in eleven months, no GPU CI, and no
   recovery for in-flight work. A 30 s heartbeat on a link with 25 ms jitter
   risks spurious leaves.
3. **It would add a second scheduler.** Node join, heartbeat and layer
   allocation would be Parallax's, not Float16's, which contradicts the brief's
   requirement that Float16 own placement.

**Verdict: candidate, not default.** The spike in `deploy/parallax_spike.sh`
establishes whether it runs on Turing at all. If it does, the realistic
integration is to use its single-node serving and keep Float16's placement, not
to adopt its scheduler.

## 10. What integration would require

* A node-scoped install in an isolated virtualenv (it pulls vLLM, SGLang, Lattica).
* Outbound access to Gradient relays and DHT bootstrap for multi-node mode.
* An auth layer in front of ports 3000–3002.
* A decision about who owns placement: `manual_layer_assignment` lets Float16 pin
  ranges, but node membership still goes through Parallax's scheduler.

**UNKNOWN until measured:** whether it loads a model on sm_75; its per-token
latency at 226 ms; whether continuous batching behaves over a WAN link.