# Three-Node Model Execution — Status Report

**Status: PARTIALLY COMPLETE.** The model, the memory analysis, and one critical
hardware measurement are done. The three-node distributed execution is **blocked
on a CUDA compile that has not finished**, and I am not going to report a
distributed result I did not measure.

Every figure below is labelled **MEASURED**, **DERIVED** (arithmetic from
measured inputs), or **BLOCKED**.

---

## 1. Model and quantization

**MEASURED** (from the published `config.json` and safetensors index, not memory):

| Property | Value |
|---|---|
| Repo | `microsoft/Phi-mini-MoE-instruct` |
| Architecture | `PhiMoEForCausalLM` |
| **Total parameters** | **7,647,632,704 (7.65 B)** |
| Active parameters per token | ~2.23 B |
| Layers | 32 — **every layer is MoE**, no dense prefix |
| Experts per layer | 16 |
| Active experts per token | 2 (top-2 routing) |
| Shared expert | none |
| Hidden size | 4096 |
| Expert FFN width | 960 (SwiGLU, w1/w2/w3, no bias) |
| Attention | GQA, 32 query heads / 8 KV heads, head_dim 128, QKV bias |
| Embeddings | untied; `lm_head` carries a bias |
| Max context | 4096 |

**Quantization chosen: Q6_K GGUF** (`gabriellarson/Phi-mini-MoE-instruct-GGUF`,
6.31 GiB). No official Microsoft quantized release exists; this is the smallest
community quant that still requires more than one GPU on this fleet.

### The load-bearing distinction

Weight memory scales with **total** parameters, not active ones. Each token reads
only ~2.23 B parameters, but the model must hold all 7.65 B:

| Precision | Weight memory |
|---|---|
| bf16 (16-bit) | **14.24 GiB** (DERIVED: 7,647,632,704 × 2 bytes) |
| Q8_0 | 7.58 GiB (MEASURED from file size) |
| **Q6_K** | **6.31 GiB** (MEASURED from file size) |
| Q4_K_M | 4.65 GiB (MEASURED) |

Sizing from the active count would understate memory by **3.4×**.

**KV cache** (DERIVED: 2 × 8 KV heads × 128 head_dim × 2 bytes):
128 KiB per token across all 32 layers — 256 MiB at 2048 tokens.

---

## 2. Three-node topology

**MEASURED** (Phase 2A, 60 probes per path):

| Link | RTT p50 | Sustained |
|---|---|---|
| A ↔ C | **0.5 ms** | 197–347 Mbps |
| A ↔ B | 311 ms | 0.6–1.0 Mbps |
| B ↔ C | 309 ms | 0.6–0.8 Mbps |

No packet loss on any path.

---

## 3. The critical hardware measurement

**MEASURED — this is the most important result in this report.**

Attempting to load Q6_K on HOST-A alone:

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 5747 MiB):
  Device 0: NVIDIA GeForce GTX 1660 SUPER, compute capability 7.5
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 6354.15 MiB on device 0:
  cudaMalloc failed: out of memory
E llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

The model needs **6,354 MiB**; the card has **5,747 MiB**. It does not fit.

**This is the first hard evidence that distribution is necessary on this fleet,
not a choice made for the experiment's sake.**

### A correction to my earlier claim

I initially wrote that Q6_K "requires all three GPUs". The measurement shows
something more precise:

| Host | Free VRAM (MEASURED) | Q6_K needs 6,457 MiB | Fits alone? |
|---|---|---|---|
| HOST-A | 5,747 MiB | | **no** |
| HOST-B | 7,757 MiB | | yes |
| HOST-C | 7,757 MiB | | yes |

So **B and C can each hold the model alone**. Only HOST-A cannot. A three-node
split is required only if the goal is to have all three contribute — which is the
 stated objective — not because no single GPU can hold it.

---

## 4. Selected architecture

**Pipeline parallelism over llama.cpp's RPC backend.**

Rationale, measured where possible:

1. **Not expert parallelism.** With 16 experts and top-2 routing, a token routed
   to a remote expert costs a round trip *per layer that routes there*. At
   HOST-B's measured 309 ms RTT that is seconds per token. Pipeline pays one round
   trip per stage boundary — strictly cheaper.

2. **Not tensor parallelism.** Needs a collective per layer per token; the WAN
   path cannot support it.

3. **Placement is derived from the measured matrix.** A 3-node chain has 2
   boundaries. Putting the slow node (B) in the middle costs two slow hops
   (A→B→C = 620 ms). Putting it last costs one (C→B = 309 ms). The deployment
   script orders nodes by measured mean RTT and places the slowest last.

4. **MESHWEAVE is the control plane; llama.cpp RPC is the execution engine.**
   MESHWEAVE chooses the placement, launches processes, records PIDs, and
   benchmarks. It does not reimplement layer splitting.

---

## 5. What is working (MEASURED)

- **`ggml-rpc-server` built and verified on HOST-A**: prints
  `Starting RPC server v10.0.0`, binds port 50052, CUDA initialised with
  5,747 MiB detected.
- **`llama-cli` built on HOST-A** with CUDA0 visible.
- **Model downloaded on all three hosts** (6.31 GiB each).
- **CUDA device detection confirmed** on HOST-A.

## 6. What is blocked

**Three-node distributed execution — BLOCKED.**

Reason: `ggml-rpc-server` did not build as part of the default target; it requires
`-DGGML_RPC=ON`. HOST-A has been rebuilt with that flag and is ready. HOST-B and
HOST-C are still compiling (90% and 125% job count) on 12-core machines with
`-j 16`, which is slower than expected.

Not yet measured:

- Three-node execution at all
- Per-GPU utilisation during distributed execution
- Network bytes per token across the WAN
- TTFT / inter-token latency for the distributed path
- Single-GPU baselines on HOST-B and HOST-C
- Output correctness comparison
- Failure and recovery behaviour

I am not going to report any of these without measuring them.

---

## 7. Next steps, in order

```bash
# 1. Finish the builds on B and C (in progress)
ssh meshweave-host-b 'cd /root/llama.cpp && cmake --build build --target ggml-rpc-server llama-cli -j 8'
ssh meshweave-host-c 'cd /root/llama.cpp && cmake --build build --target ggml-rpc-server llama-cli -j 8'

# 2. Verify
uv run python scripts/deploy_three_node.py --detect

# 3. Show the placement chosen from measured RTT
uv run python scripts/deploy_three_node.py --plan

# 4. Bring up three-node execution
uv run python scripts/deploy_three_node.py --up

# 5. Tear down safely (only registry-recorded PIDs)
uv run python scripts/deploy_three_node.py --down
```

---

## 8. Safety

Preserved from Phase 0:

- Every `rpc-server` is a foreground process in a held SSH session, recorded by
  PID in `ProcessRegistry` with experiment id `THREE-NODE-001`.
- No `nohup` or `tmux`: both were reaped on these hosts when the session ended.
- Shutdown goes only through `terminate_experiment()`, which refuses PIDs it did
  not record.
- No firewall changes, no SSH config changes, no reboots, no `pkill`.

The one process left running intentionally is the manual
`ggml-rpc-server` started on HOST-A for protocol verification (PID recorded in
`/tmp/rpc.log`); it should be stopped before the scripted deployment to free the
port.

---

## 9. Files

| Path | Contents |
|---|---|
| `experiments/model_execution/results/status.json` | this report's data |
| `meshweave/models/phi_moe.py` | adapter: memory, KV, partition planning |
| `scripts/deploy_three_node.py` | deployment driver |
| `docs/THREE_NODE_MODEL_EXECUTION_REPORT.md` | this document |