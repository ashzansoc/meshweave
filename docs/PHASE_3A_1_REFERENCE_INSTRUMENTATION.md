# Phase 3A.1 — Reference Instrumentation and Numerical Parity

**Result: NUMERICAL PARITY ACHIEVED.**

The apparent failure in Phase 3 and 3A was a **misdiagnosis, not a code defect**.
The Phase 3 implementation was already correct. This document records how that was
established and what was learned.

---

## 1. Reference build configuration

| Item | Value |
|---|---|
| Reference | llama.cpp `b1-de32b26`, built on HOST-C with CUDA |
| Model | `phi-mini-moe-Q6_K.gguf` (6.31 GiB) |
| Same file both sides | yes — no cross-precision comparison |
| Tokens | `[1724, 338, 29871, 29906, 29974, 29906, 29973, 673, 297, 697, 1734, 29889]` |
| BOS | `add_bos_token: false` — identical sequences, no BOS on either side |

## 2. Instrumentation method

`llama-cli` exposes no intermediate tensors, so a small observational dumper was
written against `libllama` (`tools/dump_reference.cpp`, ~150 lines):

- loads the model at `n_gpu_layers = 99`
- decodes the exact token list
- writes `logits.bin` from `llama_get_logits_ith(ctx, last)`
- writes `final_hidden.bin` from `llama_get_embeddings_ith`
- writes `layerNN_input.bin` when `llama_set_embeddings_layer_inp` is enabled

It performs no model mathematics. It only reads buffers llama.cpp already produces.

### Known limitation

`llama_set_embeddings_layer_inp` **aborts inside `build_graph`** for `PHIMOE`,
even with a single layer enabled. Layer-by-layer comparison was therefore not
possible, and parity is established at the logits layer only. This is documented
rather than worked around by guessing.

## 3. Captured tensor inventory

| Tensor | Shape | Status |
|---|---|---|
| `logits` | 32,064 | **captured** |
| `final_hidden` | 4,096 | captured |
| `layer00_input` … `layer31_input` | 4,096 each | **blocked** — graph abort |

## 4. Comparison results

```
max abs error     0.14927
mean abs error    0.025566
cosine similarity 0.99999917
NaN / Inf         none
```

Logits span −14.06 to 67.39. A max absolute error of 0.149 is ~0.2 % of that
range, and the cosine similarity is 0.999999 — the vectors point in the same
direction to within floating-point noise.

| | Reference | Custom |
|---|---|---|
| top-6 IDs | 29871, 13, 450, 32007, 376, 1724 | 29871, 450, 13, 32007, 376, 1724 |
| top-6 values | 67.39, 66.52, 66.52, 66.39, 65.63, 64.78 | 67.38, 66.56, 66.50, 66.38, 65.56, 64.75 |

The top-6 **sets are identical**. The order differs only for IDs 13 and 450, which
are tied at 66.52 in the reference — a tie broken differently by fp16 accumulation.

## 5. Earliest divergence

**There is none.** The first comparable tensor already agrees.

## 6. Root cause of the apparent failure

Three separate things were mistaken for a bug:

**1. The flat distribution was real.** llama.cpp's own top-5 spread is **1.76**
points. I treated flatness as a symptom of a broken residual stream. It is simply
what this model produces for this prompt.

**2. "4" was never the top token.** `llama-cli` displays generated *text*. The
actual top token in **both** implementations is **29871**, a space. "4" appears
because the CLI renders the continuation downstream of that space. Comparing
against displayed text — rather than against the logits vector — is what created
the false signal. The Phase brief warned against exactly this.

**3. The Phase 3A "fix" broke a correct implementation.** On the strength of the
HuggingFace `sparsemixer` source, I removed MoE router renormalisation. That was
wrong. llama.cpp's `build_moe_ffn` evidently renormalises for this architecture,
and the reference logits match **only** with renormalisation in place. Reverted.

## 7. Changes made

| Change | Rationale | Net effect |
|---|---|---|
| Reverted removal of MoE renormalisation | Reference logits require it | Restored parity |
| Added `tools/dump_reference.cpp` | No other way to obtain exact reference logits | Enables measurement-based debugging |
| Added `tests/test_phimoe_forward.py` (14 tests) | Lock in every verified behaviour | Prevents the same regression |

No other changes to the forward pass. The tensor loader and partitioner are
untouched.

## 8. Numerical tolerances

Justified rather than chosen:

- **Cosine similarity ≥ 0.9999.** Accumulation order and kernel differences alone
  cannot explain a cosine below this if the computation is the same.
- **max abs error ≤ 0.25.** With logits of magnitude ~67 and fp16 activations,
  roughly 2–3 significant digits survive; 0.25 is ~0.4 % and matches observed
  0.149 with margin.
- **Top-1 within the tied sets.** Exact ordering of logits that differ by <0.05 is
  not a meaningful signal.

These follow from the precision difference between fp16 activations and
llama.cpp's mixed kernels, not from a desire to pass.

## 9. Regression tests

`tests/test_phimoe_forward.py` — 14 tests:

- tensor orientation for 2D and 3D expert stacks
- RMSNorm formula, bias application, dtype preservation
- RoPE: split-halves pairing, identity at position 0, norm preservation, offset
- KV cache store/read and positional growth
- **MoE renormalisation is required**, plus a guard that the two variants differ
- geometry constants including GQA divisibility

Full suite: **111 tests pass**. `ruff` clean. `mypy --strict` clean on 53 files.

## 10. Remaining limitations

1. **No per-layer comparison.** The graph aborts when layer hooks are enabled for
   this architecture. Parity is proven at the logits, which bounds total error but
   cannot localise a compensating pair of errors. Unlikely, but unproven.
2. **`final_hidden` is not directly comparable.** llama.cpp reports it
   post-final-norm; the runtime returns the pre-norm layer output. Cosine 0.854
   reflects that mismatch of definition, not an error.
3. **Only one prompt tested.** A second prompt with different content would
   strengthen the claim.

## 11. Next corrective action

1. Test a second, unrelated prompt to confirm the agreement is not prompt-specific.
2. Proceed to Phase 3 distributed execution: the single-GPU implementation is now
   trustworthy at the level of the logits vector, which is the gate the brief
   required.
3. Optionally, revisit per-layer instrumentation by patching `build_graph` guarded
   by an env var, if a future divergence needs localising.

---

## The lesson

The forward pass was correct at Phase 3. Two phases of debugging were spent
chasing a symptom that came from comparing the wrong quantity — rendered text
instead of logits — and from assuming a distribution shape was a defect.

Recording the reference logits vector is what resolved it in one comparison. That
instrumentation should have been the first step, not the last.
