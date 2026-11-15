# Phase 3A — Numerical Correctness

**Status: NOT ACHIEVED.** The custom runtime executes all 32 layers correctly in
structure but does not yet produce llama.cpp's output. Two real defects were found
and one was fixed; the remaining divergence is not yet isolated.

No distributed execution was attempted, per the phase instruction.

---

## 1. Exact reference configuration

Verified from GGUF metadata, not assumed:

| Field | Value | Matches implementation |
|---|---|---|
| `general.architecture` | `phimoe` | yes |
| `tokenizer.ggml.bos_token_id` | 1 | — |
| `tokenizer.ggml.eos_token_id` | 32000 | — |
| **`tokenizer.ggml.add_bos_token`** | **False** | **BOS hypothesis eliminated** |
| `phimoe.attention.layer_norm_rms_epsilon` | 1e-5 | yes |
| `phimoe.rope.freq_base` | 10000.0 | yes |
| `phimoe.expert_count` | 16 | yes |
| `phimoe.expert_used_count` | 2 | yes |

**Critical: `add_bos_token` is False.** llama.cpp does not prepend BOS for this
model, so the token sequence is identical between both implementations. The BOS
hypothesis from Phase 3 is **wrong** and is now closed.

## 2. Exact token sequence

Prompt: `What is 2+2? Answer in one word.`

```
[1724, 338, 29871, 29906, 29974, 29906, 29973, 673, 297, 697, 1734, 29889]
```

No BOS prefix. Both implementations receive this exact sequence.

## 3. Original numerical failure

| | Phase 3 result |
|---|---|
| llama.cpp next token | `4` |
| Custom top-1 | 29871 (`' '`, a space) |
| Custom top-5 values | 67.4, 66.6, 66.5, 66.4, 65.6 |

The top-5 spread of ~2 points across a vocabulary of 32,064 is nearly uniform. A
correct run has a dominant top-1. That flatness has driven every hypothesis below.

## 4. Defects found

### 4.1 CONFIRMED BUG — MoE router renormalisation (FIXED)

My code renormalised the top-2 router weights to sum to 1:

```python
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
```

Phi's reference `sparsemixer` takes the top-k of a softmax over **all 16** experts
and uses those probabilities **directly**. They sum to well under 1. Renormalising
scaled every MoE output up by `1/sum`, compounding across 32 layers.

llama.cpp exposes an explicit `norm_w` flag in `build_moe_ffn`, confirming that
models differ here.

**Fixed.** The output changed materially (top-5 values went 67→49), which confirms
the fix took effect, but it did not achieve parity on its own.

### 4.2 HYPOTHESIS TESTED AND REJECTED — RoPE convention

`LLAMA_ARCH_PHIMOE` returns `LLAMA_ROPE_TYPE_NEOX` (verified in
`src/llama-model.cpp`). Reading the CUDA kernel:

```c
if (is_neox && i0 < n_dims) {
    ix0 = i0/2;
    ix1 = i0/2 + n_dims/2;
}
```

NEOX rotates **split halves** — pairs `(j, j + head_dim/2)` with frequency index
`j`. That is exactly what `rotate_half` with `cat(freqs, freqs)` implements. My
RoPE was already correct; this hypothesis is closed.

### 4.3 HYPOTHESIS TESTED AND REJECTED — norm biases

The GGUF carries `attn_norm.bias`, `ffn_norm.bias` and `output_norm.bias` with
non-zero values (range roughly −0.75 to +0.77). Because a large constant added to
the residual stream would explain both the inflated magnitude and the flatness,
applying them was tested directly.

**Disabling them alone did not fix it** (top-5 became 26.6, 25.0, 24.1 — still
flat, top token 645). This hypothesis is closed as a *sole* cause, though the
correct treatment of these biases remains unverified.

## 5. Current state after fixes

| Configuration | Top-1 | Top-5 values |
|---|---|---|
| llama.cpp reference | `4` | not exposed |
| Phase 3 (original) | 29871 | 67.4, 66.6, 66.5 |
| + renorm removed | 27137 | 48.9, 47.1, 45.4 |
| + renorm removed, biases off | 645 | 26.6, 25.0, 24.1 |

Still wrong in all three. The value scale moves, the flatness persists.

## 6. Earliest divergence — NOT YET ISOLATED

This is the unfinished work. llama.cpp cannot dump intermediate tensors through
`llama-cli` (checked: no logits or activation flags), so parity has only been
compared at the final logits.

**The required harness was not built.** It needs one of:

1. A small C++ program linked against `libllama` that runs `llama_decode` and
   dumps per-layer tensors, or
2. A second pure-Python reference using `transformers`' native `PhiMoEForCausalLM`
   (supported since 4.46, no `trust_remote_code`), built from a GGUF→state_dict
   conversion held in RAM. This gives inspectable intermediates at every stage and
   is the more practical route on these hosts.

Without that, I am comparing only the endpoint — which the phase brief explicitly
warns against.

## 7. Remaining suspects, ordered

1. **Attention correctness.** A flat distribution is what you get if attention
   contributes little. Specific unverified points: GQA head grouping order, whether
   the o_projection is applied per-head or after reshape, and whether the
   `attn_output` weight orientation matches.
2. **Norm bias semantics.** These biases are unusual for Phi-3-class models.
   Whether they belong inside the norm, after it, or not at all is unresolved.
   Tested only as on/off, never against a reference.
3. **Expert aggregation.** The two selected experts may need summing in a
   different order, or the `ffn_down_exps` orientation may differ from
   `ffn_gate_exps`.
4. **Logit scaling.** A model-specific output scale would explain the uniformly
   large magnitudes (67 → 49 → 27) across all three variants.
5. **Quantisation interpretation.** `Q8_0` expert down-projections vs `Q6_K`
   elsewhere may be decoded with a different block layout than the `gguf` package
   assumes.

## 8. What is verified working

| Component | Evidence |
|---|---|
| GGUF loading | 517 tensors, all shapes validated against geometry |
| Tensor orientation | measured: GGUF arrays are `[out,in]`, normalised to `[in,out]` |
| Layer partitioning | 32 layers split 11/10/11, each stage loads independently |
| Three-stage execution | runs at 4.8/4.6/5.7 GiB peak on one GPU, hidden `[1,12,4096]` maintained |
| No NaN/Inf | checked; none observed |
| Reproducibility | repeated runs give identical top-5 ordering |

## 9. Numerical tolerances

**Not yet defined.** The phase requires tolerances justified by quantisation and
precision rather than chosen to make tests pass. Defining one now, against a
reference that disagrees structurally, would be the exact failure the brief warns
about.

## 10. Regression tests

Added: none for parity, because parity is not achieved. The existing 97 tests
still pass; `ruff` and `mypy --strict` are clean across 53 files.

## 11. Remaining limitations

- No intermediate-tensor comparison harness exists. This is the blocking gap.
- Norm bias treatment unverified.
- No distributed execution attempted (correctly, per the brief).

## 12. Next step

Build the `transformers`-based Python reference. It is the fastest route to
per-layer intermediates on hardware that cannot run a second llama.cpp build
conveniently, and it settles suspects 1–3 by comparison rather than hypothesis.

Until intermediate tensors are compared and the first divergent layer is found,
further changes to the forward pass are guesswork.
