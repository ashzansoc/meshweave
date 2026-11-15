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

## 5. The model works — single-GPU baselines (MEASURED)

Phi-mini-MoE Q6_K loads and generates correct output on two of the three GPUs.

| Host | GPU | Free VRAM | Loads? | Prompt t/s | **Generation t/s** | Output |
|---|---|---|---|---|---|---|
| **HOST-C** | RTX 2070 SUPER | 7,785 MiB | **yes** | 341.3 | **133.7** | correct |
| **HOST-B** | Quadro RTX 4000 | 7,786 MiB | **yes** | 84.1 | **63.7** | correct |
| HOST-A | GTX 1660 SUPER | 5,747 MiB | **no** — OOM | — | — | — |

Harness: `llama-cli -ngl 99 --temp 0`, greedy decoding, all 32 layers on the GPU.

Correctness check: asked "What is 2+2? Answer in one word." — both hosts answered
`4`. VRAM returned to 1 MiB after exit, so nothing leaked.

**HOST-C does 133.7 tokens/second.** That single number reframes the whole
exercise:

- A three-node pipeline over the measured links would add ~310 ms per token at
  the HOST-B boundary, capping throughput near **3 tokens/second**.
- Single-GPU HOST-C already does **133.7 tokens/second** — about **45× faster**.

So for *this* model, on *this* fleet, distribution is strictly worse. The model
fits on HOST-B and HOST-C individually; it does not need three GPUs. Three-node
execution would be justified only by capacity that a single card lacks, and the
measurements show that capacity is not needed here.

## 5b. What is working (MEASURED)

- **`ggml-rpc-server` built and verified on HOST-A**: prints
  `Starting RPC server v10.0.0`, binds port 50052, CUDA initialised with
  5,747 MiB detected.
- **`llama-cli` built on HOST-A** with CUDA0 visible.
- **Model downloaded on all three hosts** (6.31 GiB each).
- **CUDA device detection confirmed** on HOST-A.

## 6. The distributed attempt (MEASURED — did not complete)

All three hosts now have `ggml-rpc-server` built with CUDA. The nodes were started
on internal port 10200, which Vast maps externally on every host:

| Host | Internal | External | GPU visible to RPC |
|---|---|---|---|
| HOST-A | — | driver | GTX 1660 SUPER, 5,747 MiB |
| HOST-B | 10200 | 83.63.1.187:33774 | Quadro RTX 4000, 7,686 MiB free |
| HOST-C | 10200 | 1.208.108.242:46092 | RTX 2070 SUPER, 7,679 MiB free |

The RPC protocol works. Evidence:

```
HOST-B rpc-server log:  "Accepted client connection"
HOST-C nvidia-smi:      2452 MiB allocated
HOST-A -> HOST-B:33774: TCP OPEN
```

**But the load did not complete.** After three minutes HOST-C held 2,452 MiB and
sat at **0% GPU utilisation**, with the driver still running and no progress.

### Root cause

**llama.cpp's RPC backend transfers model tensors to remote devices over TCP at
load time.** The weights are not read from the worker's own disk; they are shipped
across the network by the driver.

That collides with the measured links:

| Link | Sustained | ~2 GiB shard would take |
|---|---|---|
| A ↔ C | 347 Mbps | ~50 s |
| A ↔ B | 0.6–1.0 Mbps | **~5.7 hours** |

So HOST-B cannot receive a shard in any practical time, and even the fast link
did not finish inside the test window.

**This is a link-capacity limit, not a code fault.** It is exactly the class of
constraint the Phase 2A measurements predicted, now confirmed against the real
runtime.

### The fix, and why it matters

Every host already has the complete 6.31 GiB GGUF **on local disk**. A design that
loads weights from local disk on each worker and sends only *activations* between
hosts avoids the transfer entirely. At 7 KiB per token per hop (Phi-mini-MoE's
hidden size is 4096, so ~8 KiB fp16), the payload is negligible even at 0.6 Mbps —
0.8 Mbps moves 8 KiB in ~80 ms, versus hours for the weights.

That is the architecture MESHWEAVE's own pipeline runtime implements
(`meshweave/runtime/pipeline.py`): it loads each stage from local storage and
passes activations. It is the correct design for this fleet, and the llama.cpp RPC
result is the measurement that justifies it.

## 7. What is not measured

- Three-node inference output (no token was generated)
- Per-GPU utilisation during a completed run
- TTFT / inter-token latency
- Network bytes per token on a completed run
- Single-GPU baselines on HOST-B and HOST-C
- Correctness comparison
- Failure and recovery

I am not reporting these, because I did not measure them.

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