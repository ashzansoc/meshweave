# Distributed Deployment

Put one model across several machines from the dashboard: paste a Hugging Face
repository, tick the nodes, deploy. Each node holds a contiguous range of decoder
layers in its own VRAM and the hidden state travels between them.

This document says exactly what runs where, what has been verified, and what has
not.

---

## 1. What is actually distributed

The strategy is **pipeline parallelism**, split at layer boundaries.

For a 24-layer model on two nodes:

| stage | node | layers | holds |
|---|---|---|---|
| 0 | node A | `[0,12)` | embedding + layers 0–11 |
| 1 | node B | `[12,24)` | layers 12–23 + final norm + language head |

Only the hidden state crosses the network: `1 × seq × hidden` in fp16. For
Qwen2.5-0.5B that is 1.8 KiB per decoded token, or about 7 KiB for a 4-token
prompt. The last stage samples the next token from its own logits and returns the
token id **directly** to rank 0, which removes a full round trip per token — the
same fix that took the earlier three-host Phi-MoE run from 2,663 to 453 ms per
token.

Weights are genuinely split. `meshweave/runtime/dense.py` reads only the
tensors for the layers a stage owns, straight out of the safetensors shards on
that machine. Nothing sends weights over the network.

---

## 2. Why not DDP or FSDP

Both were considered, because they are the obvious PyTorch answers.

**DDP does not split weights.** It replicates the full model on every process and
synchronises gradients. There is no gradient descent during inference, so DDP
contributes nothing here and would simply put a full copy on every node.

**FSDP does split weights**, and it is the generic answer for a model that does
not fit on one card. It shards parameters across ranks and all-gathers a layer's
full parameters for each forward pass. That all-gather is the problem on these
links. A single Qwen2.5-0.5B decoder layer is about 30 MB in bf16. At the
measured 0.6–1.5 Mbps between the current hosts that is roughly four minutes per
layer — per token. Even on the 197 Mbps path measured within one provider, one
layer's all-gather is over a second, so a 24-layer model would need about half a
minute to emit a single token.

Pipeline parallelism moves 1.8 KiB per hop instead of 30 MB per layer, and is
the mechanism that the measurements in `docs/PHASE_3C_PERFORMANCE_REPORT.md` and
`docs/PHASE_3D_1_DIRECT_RETURN_REPORT.md` support. If a fast interconnect is ever
available (a real datacenter environment, class A), FSDP becomes a reasonable
alternative and the planner should choose between them on measured bandwidth
rather than preference.

This is a deliberate engineering decision, not a refusal to use torch.distributed:
the numbers are in this repository.

---

## 3. What has been verified

| Claim | Evidence | Status |
|---|---|---|
| Stage maths match the reference implementation | `tools/check_dense_parity.py`: single stage, 3-way split and a decode step all within 1.8e-07 of `transformers`, fp32 | measured |
| A split model generates the same tokens as an unsplit one | `tools/check_pipeline.py`: two real processes, real TCP sockets, identical output | measured |
| Weights are split, not replicated | stage 0 held 322 KiB, stage 1 145 KiB, stage 2 177 KiB of a 644 KiB model | measured |
| The agent starts a stage from a queued command and reports status | deployment API tests drive the real routes | measured (in-process) |
| A node without a published TCP port is refused, by name | live `POST /v1/deployments` against the running coordinator | measured |
| A real multi-host run over the current Vast links | blocked: the nodes do not all publish a TCP port | **blocked** |

The last line is the honest gap. The pipeline works between processes and the
control plane works end to end, but a genuinely cross-machine deployment has not
been run since the current instances were created, because one of them publishes
only SSH.

---

## 4. What a node needs

1. **A published TCP port**, not just SSH. Peers dial `external_ip:mapped_port`;
   a node behind NAT that accepts no connections cannot take part in a
   multi-stage chain. The coordinator picks the lowest published port that is not
   22, so no configuration is needed beyond publishing one.
2. **Disk for the whole checkpoint.** Shards are not split finely enough to
   download a single layer range, so every node downloads the full repository
   (about 1 GiB for Qwen2.5-0.5B in bf16) even though it keeps only its layers in
   VRAM. The planner refuses a node whose free disk is below 90 % of that.
3. **Free VRAM for its own layers**, checked per node against that node's own
   reported free memory with 15 % headroom, plus a KV cache reservation for
   `max_seq` tokens. Totals across nodes are never added up and presented as a
   single GPU.
4. **The current agent.** Re-run the installer to pick up the stage runtime,
   `safetensors`, `tokenizers` and `huggingface_hub`.

---

## 5. Supported models

The runtime implements one architecture family: `Qwen2ForCausalLM`, which covers
the Qwen2 and Qwen2.5 dense models. Anything else is refused by name at deploy
time, with the architecture quoted in the error. It is not attempted with
"close enough" maths.

Two config details are read from the model's own `config.json` and are load-
bearing:

* `rope_theta` lives under `rope_parameters` in files written by transformers v5.
  Reading only the legacy top-level key silently fell back to theta=10000 on a
  1e6 model, which changes every rotary angle while still producing plausible
  output. Both layouts are now read.
* `rope_scaling` and `partial_rotary_factor != 1` are refused rather than
  approximated.

RMSNorm is implemented exactly as the reference does it — narrowing to the model
dtype before applying the weight — because the more accurate fp32 variant is a
different function and the difference is visible in bf16.

---

## 6. Operating it

Launch from the dashboard's **Deploy** view, or:

```bash
curl -s -X POST http://127.0.0.1:8700/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","node_ids":["alpha","beta"]}'
```

The response is the plan: which node runs which layers, how much VRAM each stage
holds, and the address each stage will be dialled at. A refusal is a 409 whose
message names the node and the shortfall.

A prompt is accepted once every stage reports `running`:

```bash
curl -s -X POST http://127.0.0.1:8700/v1/deployments/<id>/prompt \
  -H 'content-type: application/json' \
  -d '{"prompt":"The capital of France is","max_new_tokens":32,"temperature":0}'
```

The result comes back through the agent with token counts, time to first token
and milliseconds per token.

---

## 7. Control surface

The agent still has no shell endpoint and no inbound port.

* The coordinator enqueues `run_stage`, `stop_stage` and `run_prompt`. The agent
  polls `/v1/agents/{node}/work`. Any other command kind is ignored, not
  executed.
* `run_stage` starts `stage_runner.py` as a child process in its own session and
  records the pid in `/var/lib/meshweave/stages.json`.
* `stop_stage` signals **only** a pid recorded there, after confirming
  `/proc/<pid>/cmdline` still contains `stage_runner.py`. A recycled pid is never
  signalled.
* Prompts reach rank 0 through a unix socket at `/run/meshweave/control.sock`,
  mode 0600. It is not bound to any network interface.
* Prompts run on a worker thread so a long generation cannot stall heartbeats
  and get the node declared unreachable while it is working.

---

## 8. Known limits

* One prompt at a time per deployment: rank 0 holds a lock and the KV cache is
  reset per request.
* `max_seq` defaults to 2048; the KV cache is allocated per stage up front.
* A node that dies mid-generation fails the request, not the deployment. Failure
  handling for a lost stage is the next piece of work.
* No quantisation yet: the checkpoint's own dtype is used (bf16 for these
  models).
