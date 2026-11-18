# POC 01 — results

Two independent rented GPU hosts, one model split across them, one endpoint in
front. Everything below was measured on `2026-11-18` against the live machines.

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 0.49B params, bf16 checkpoint) |
| Nodes | `gpu-node-1` — RTX 2070 SUPER, 8 GiB, 12 cores · `brisk-thistle-9ae2` — GTX 1660 SUPER, 6 GiB, 28 cores |
| Precision | float16 (both cards are compute 7.5, which has no bf16 kernels) |
| Cluster | `F16-VDC-001` |
| Prompt | `"The capital of France is"` unless stated |

---

## 1. The model runs, split across two machines

Both nodes held 12 of 24 layers. Neither held the whole model.

| Stage | Node | Layers | Resident VRAM |
|---|---|---|---|
| 0 | gpu-node-1 | `[0,12)` | 0.587 GiB |
| 1 | brisk-thistle-9ae2 | `[12,24)` | 0.587 GiB |

Generated text:

> Paris. It is the largest city in Europe and the third largest city in the
> world. It is located in the south

The single-node run of the same model and prompt produced **byte-identical
text**, which is the check that the split is a transport detail and not a
different computation.

## 2. Both GPUs were busy during the same generation

NVML samples from both agents, taken while the request was in flight:

| Node | Time | GPU utilisation | VRAM in use |
|---|---|---|---|
| gpu-node-1 | 15:12:47 | 1.0 % | 1.17 GiB |
| brisk-thistle-9ae2 | 15:12:48 | 2.0 % | 1.37 GiB |

Both in the same one-second window. Utilisation is low because a distributed
decode step is mostly waiting on the network, not computing — see §4.

## 3. Benchmarks A, B and C

Same model, same prompt, same 24-token generation, same settings.

| Configuration | tok/s | ms/token | TTFT | total |
|---|---|---|---|---|
| A — all on gpu-node-1 | 19.02 | 38.8 | 368 ms | 1.26 s |
| B — all on brisk-thistle-9ae2 | 16.67 | 41.0 | 497 ms | 1.44 s |
| C — split across both | 2.93 | 310.8 | 1053 ms | 8.20 s |

**Distributed is 6.5× slower than the faster single-node configuration.** This
is measured, not projected, and the report does not describe it as an
improvement.

The pipeline round trip was 309.8 ms per token, against a measured link RTT of
226–228 ms. The remainder is the two compute hops and serialisation.

## 4. Where the time goes

A decode step moves 2,090 bytes of activation per token on average (1.8 KiB for
the decode step itself, plus the prompt's hidden states amortised over the
generation). At 8–15 Mbps that transfer is a small fraction of the 310 ms; the
round trip dominates.

This matches the earlier phases on a different model: at 300 ms RTT, a
pipeline's per-token cost is roughly one round trip per hop, and compute is a few
percent of the total.

## 5. The link, measured both directions

| | gpu-node-1 → brisk-thistle | brisk-thistle → gpu-node-1 |
|---|---|---|
| RTT min / p50 / p95 | 226 / 228 / 337 ms | 223 / 224 / 237 ms |
| Jitter | 25.8 ms | 5.3 ms |
| Upload / download | 14.7 / 8.0 Mbps | 14.6 / 8.3 Mbps |
| Failed round trips | 0 of 10 | 0 of 10 |

Measured between the two containers over their published ports. Not ICMP, and
not a provider's advertised figure.

## 6. What this proves, and what it does not

**Proven by execution:** a 0.5B model is partitioned by layer across two
machines; both GPUs run a forward pass during the same request; activations
cross the network in both directions; the output is identical to a single
machine; and the whole thing is addressable through one deployment record.

**Not proven, and not claimed:** that distribution is faster. On these links it
is 6.5× slower for a model that fits on one card. The honest statement is that
distribution is a **capacity** mechanism — it lets a model that no single GPU can
hold run at all — and this fleet's cards are large enough that the mechanism is
not needed for a 0.5B model.

## 7. Reproducing it

```bash
# 1. link measurement
curl -s -X POST http://127.0.0.1:8700/v1/network/measure \
  -H 'content-type: application/json' \
  -d '{"node_ids":["gpu-node-1","brisk-thistle-9ae2"]}'

# 2. the cluster and the deployment
curl -s -X POST http://127.0.0.1:8700/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","cluster_id":"F16-VDC-001"}'

# 3. the three benchmarks
python3 benchmarks/vdc/run_benchmark.py \
  --coordinator http://127.0.0.1:8700 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nodes gpu-node-1 brisk-thistle-9ae2
```

Raw numbers: `benchmarks/vdc/results.json`.

## 8. Known limitations of this run

* One request at a time; rank 0 serialises on its KV cache.
* No streaming; the gateway returns one response.
* No mid-generation recovery. A stage that dies takes the request with it.
* Qwen2 dense only.
* GPU utilisation is a 1-second NVML average sampled every 0.5 s, so a short
  request yields few samples. It shows that the GPU was working, not a detailed
  profile.