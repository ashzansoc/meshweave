# DeepSpeed — technical research

**Repository:** https://github.com/deepspeedai/DeepSpeed · Apache-2.0 · ~43k stars
**Latest release:** v0.20.0, 2026-11-12 · **Docs:** 0.20.1
**Assessed:** 2026-11-18

Evidence tags: **VERIFIED** (read in source), **DOCUMENTED**, **UNKNOWN**.
DeepSpeed is a training library with a maintained but secondary inference path;
this assessment is about `deepspeed.InferenceEngine` only.

---

## 1. Parallelism at inference, by capability

Evaluated separately, because the three capabilities have different verdicts.

### 1.1 Tensor parallelism — present, unsuitable for WAN

**VERIFIED.** Three modes exist: a user-supplied injection policy, kernel
injection (`replace_with_kernel_inject`), and AutoTP (`tp_size > 1`,
`deepspeed/module_inject/auto_tp.py`). AutoTP supports Llama, Mistral, Qwen2/3,
Mixtral, Phi3, Falcon, ChatGLM, DeepseekV2/V3 and OPT; it explicitly refuses
deberta, flaubert, fsmt, gpt2, led, longformer, xlm and xlnet.

The cost is structural. Each TP-sharded linear calls `dist.inference_all_reduce`
(`deepspeed/module_inject/layers.py`), routed in `deepspeed/comm/torch.py`:

* a two-rank P2P kernel for messages under 4 MB,
* a shared-memory `torch.ops.deepspeed.inference_all_reduce_` (x86, AVX2/AVX512,
  world-wide only),
* otherwise `torch.distributed.all_reduce` — the only path that works across
  machines.

So a multi-node TP forward performs **one synchronising all-reduce per layer**.
For 24 layers at a 226 ms RTT that is ~5.4 s per token before any compute.
Bandwidth is irrelevant here; the cost is per-layer synchronisation. There is no
overlap with computation and no WAN-specific handling.

**Verdict: unsuitable for the 226 ms environment.** Not a marginal call — it is
two orders of magnitude away from usable.

### 1.2 Pipeline parallelism — not available at inference

**VERIFIED.** `deepspeed/engine.py` raises
`RuntimeError('pipeline parallelism is currently not supported in inference.')`.

DeepSpeed does have pipeline parallelism, but `PipelineModule` and the `PP`
engine are **training** APIs: they schedule forward and backward passes with
micro-batches and gradient accumulation. There is no inference pipeline path to
evaluate for latency. So the correct statement is not "PP is too slow here" — it
is that the inference engine does not offer it.

**Verdict: not applicable at inference, by design.** This is not a criticism of
the design: in a datacenter with NVLink, TP is the better choice and PP is
unnecessary.

### 1.3 Quantisation — present

**VERIFIED.** fp32, fp16, bf16 and int8 via `deepspeed.init_inference` and the
config `dtype`. int8 is a dtype conversion, not a calibrated GPTQ/AWQ path.

---

## 2. Backends and topology assumptions

**VERIFIED.** NCCL or Gloo via `torch.distributed`; no InfiniBand or NVLink
requirement is coded. The shared-memory fast path is x86-only and world-wide,
with no multi-node fallback (issue #7625). Nothing in the source addresses
high-latency links, NAT, or partial connectivity.

**UNKNOWN.** Whether the compiled kernels pass on sm_75. Open issues #8831,
#8828 and #8830 report out-of-bounds reads and writes in the inference v2 norm
kernels for hidden sizes ≤128 under fp16/bf16. Older issue #59 ("Turing NLG")
suggests early Turing use, but there is no supported matrix entry.

**Assessment: Turing is unvalidated, not supported.**

---

## 3. Process model

**VERIFIED.** DeepSpeed-Inference is not an in-process library you call once. It
needs `torch.distributed.init_process_group` (MASTER_ADDR/PORT, RANK, WORLD_SIZE,
LOCAL_RANK), a multi-process launch (torchrun or the DeepSpeed launcher), and then
`deepspeed.init_inference(model, config)` per rank.

An external coordinator would have to spawn and supervise N ranks itself. That is
compatible with the existing agent model in principle — the agent already starts
and supervises processes — but it would be a second runtime with its own launcher
semantics.

---

## 4. Maturity and licence

Apache-2.0. v0.20.0 is current and actively maintained. Inference v1 is stable;
inference v2 (FastGen, ragged batching) is labelled experimental in the docs.
The project's centre of gravity is training.

---

## 5. Where DeepSpeed fits in Float16

**Not in the 226 ms WAN environment.** Tensor parallelism needs a per-layer
all-reduce that this link cannot amortise, and the inference path has no pipeline
parallelism to fall back on.

**Yes, for the datacenter class.** On a node or rack with NVLink, TP with
DeepSpeed's fused kernels is a credible way to serve a model that exceeds one
GPU, and it is a better fit there than a hand-written pipeline. It is retained as
the intended backend for environment class A (datacenter) in
`docs/UNIFIED_INFERENCE_ARCHITECTURE.md`.

**What would have to be true to adopt it:** a measured multi-node all-reduce
latency on the target interconnect that is a small fraction of layer compute, a
kernel pass on the target GPU architecture, and a decision to let DeepSpeed own
rank placement within a node.

## 6. Open questions

* Real kernel pass rate on sm_75 (issues suggest problems at small hidden sizes).
* Measured all-reduce latency for a 0.5B-scale model on a 300 ms link — reasoned
  above, not measured, because installing it to measure a 5 s/token outcome is
  not a good use of a GPU host.
* Whether any hidden pipeline path exists for inference (the explicit
  `RuntimeError` says no).