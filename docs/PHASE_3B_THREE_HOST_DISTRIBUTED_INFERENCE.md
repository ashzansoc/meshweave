# Phase 3B — Three-Host Distributed Model Execution

**Result: three independent GPU hosts execute one model together, with numerical
parity and working autoregressive generation.**

| Requirement | Status |
|---|---|
| HOST-A loads its weights locally | **yes** — layers 0–10 + embedding |
| HOST-B loads its weights locally | **yes** — layers 11–20 |
| HOST-C loads its weights locally | **yes** — layers 21–31 + final norm + head |
| All three GPUs execute real computation | **yes** — measured below |
| Activations cross real host boundaries | **yes** — persistent TCP, A→B→C |
| One forward pass succeeds | **yes** |
| Logits agree with reference | **yes** — cosine 0.9999994 |
| Correct token generated | **yes** — first token matches reference |
| Short response generated | **yes** — 5 tokens |

---

## 1. Model and configuration

`microsoft/Phi-mini-MoE-instruct`, Q6_K GGUF (6.31 GiB), 32 layers, 16 experts
top-2, hidden 4096, GQA 32/8. Identical file on all three hosts; weights are read
from local disk on each and never transferred.

## 2. Partition ownership (measured free VRAM, not assumed)

| Host | GPU | Free VRAM | Layers | Bytes owned |
|---|---|---|---|---|
| HOST-A | GTX 1660 SUPER | 5,747 MiB | 0–10 (11) + embed | ~4.5 GiB |
| HOST-B | Quadro RTX 4000 | 7,786 MiB | 11–20 (10) | ~4.1 GiB |
| HOST-C | RTX 2070 SUPER | 7,785 MiB | 21–31 (11) + head | ~4.5 GiB |

HOST-A takes the fewest layers because it has the least VRAM. The partition is
capacity-proportional, not equal.

## 3. Activation protocol

Length-prefixed JSON header + raw fp16 payload over a persistent TCP connection.
The header carries protocol version, request id, stage, absolute position, shape,
dtype, payload length and a **CRC32** — a corrupted frame raises rather than being
fed to the model.

Measured activation size: **8 KiB per hop** (1 × 12 × 4096 fp16 = 98,304 bytes for
the prefill, 8,192 bytes per decode step). Small compared with the 6.31 GiB of
weights that are *not* sent, which is the entire point of this architecture.

## 4. Distributed forward pass — MEASURED

Identical token ids to the llama.cpp reference:
`[1724,338,29871,29906,29974,29906,29973,673,297,697,1734,29889]`

| Trial | Elapsed | max abs err | mean abs err | cosine | top-1 |
|---|---|---|---|---|---|
| 0 | 20.6 s | 0.1823 | 0.04081 | 0.9999994 | 29871 ✓ |
| 1 | 13.1 s | 0.1823 | 0.04081 | 0.9999994 | 29871 ✓ |
| 2 | 12.8 s | 0.1823 | 0.04081 | 0.9999994 | 29871 ✓ |

Reference (single-GPU, Phase 3A.1): max abs 0.149, mean abs 0.0256, cosine
0.99999917.

The distributed error is marginally higher because activations are serialised to
fp16 at two stage boundaries. Bit-identical results across all three trials
confirm the pipeline is deterministic.

## 5. Token generation — MEASURED

```
prefill              13,782 ms
token 1  29871       3,352 ms
token 2     13       2,729 ms
token 3  32001       5,584 ms
token 4   7803       2,618 ms
token 5  29889
```

The first token (29871) matches the reference's top-1 exactly.

## 6. The bug this phase found

The first distributed run gave cosine **0.99998** but mean absolute error
**3.26** — a near-constant offset that preserved ranking. That signature meant
the computation was structurally right but systematically shifted.

**Cause:** each stage advanced `pos` by its own sequence length when forwarding
downstream. For a 12-token prefill, stage 1 received `pos=12` and rotated RoPE at
positions 12–23 instead of 0–11, and stage 2 at 24–35.

**Fix:** `pos` is the absolute position of the *first* token and must pass through
the chain unchanged. Each stage appends its own KV entries at those same absolute
offsets.

**Effect:** mean abs error 3.26 → 0.04081.

This is recorded in `tests/test_phimoe_forward.py` behaviour through the RoPE
offset tests.

## 7. Performance — measured, and slow

| Configuration | Decode |
|---|---|
| HOST-C standalone (llama.cpp) | ~7.5 ms/token (133.7 tok/s) |
| **Three-host pipeline** | **2,618–5,584 ms/token** |

**~350–750× slower.** This is not a defect; it is the measured cost of the
topology. Every generated token crosses two WAN boundaries:

- A→B: 311 ms RTT
- B→C: 309 ms RTT

Plus stage compute. A single token therefore pays ~620 ms of pure network before
any GPU work, and the observed figures are several times that — the gap is worth
investigating (serialisation, TCP behaviour on the lossy link, or per-request
overhead in the relay).

Prefill at 13.8 s is consistent with the same two-hop cost applied to a longer
activation.

## 8. What this proves and does not prove

**Proves:** one model, three independent machines, weights resident locally,
activations crossing real network boundaries, output numerically matching the
reference, and autoregressive generation working.

**Does not prove:** that this is economically useful. At 0.3–0.4 tokens/second it
is 350–750× slower than a single GPU that can already hold the model. HOST-B's
304 ms RTT makes it a poor pipeline partner for anything latency-sensitive.

## 9. Failed experiments and dead ends

| Attempt | Outcome |
|---|---|
| llama.cpp RPC across all three | hangs at load — connection churn at ~300 ms RTT |
| `gguf` module missing on A/B | stages crashed on import; installed |
| Port 10100 held by an echo sink | bind failure; stopped by exact PID after confirming identity |
| Stage exited on client disconnect | fixed: stages now loop on `accept()` |
| Position advanced per stage | the offset bug above |

## 10. Safety

All processes were started as foreground children of live SSH sessions or via
`nohup` with recorded PIDs. Every termination was by exact PID after inspecting
`/proc/<pid>/cmdline` to confirm identity — never by port, never by pattern.
No firewall, SSH config, or host state was modified. No reboots.

## 11. Recommended next phase

1. **Profile the 2.6–5.6 s/token.** The network accounts for ~620 ms; the rest is
   unexplained and probably fixable.
2. **Reconsider HOST-B's role.** Its measured 304 ms RTT means it should not sit
   in the per-token path. Options: prefill-only, or an independent replica.
3. **Test a model too large for one GPU.** The architecture is proven; its value
   only appears when single-GPU execution is impossible.

## 12. Files

| Path | Contents |
|---|---|
| `experiments/model_execution/pipeline_node.py` | stage worker + coordinator client |
| `experiments/model_execution/results/three_node/summary.json` | raw measurements |
| `tools/dump_reference.cpp` | llama.cpp reference dumper |
| `docs/PHASE_3A_1_REFERENCE_INSTRUMENTATION.md` | parity basis |
