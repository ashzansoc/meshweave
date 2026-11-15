# Three-node model execution

Raw results for the Phi-mini-MoE three-node experiment.

| File | Contents |
|---|---|
| `capabilities.json` | per-host binary/model/GPU check |
| `deployment.json` | which node drove, which served, the layer split |
| `single_gpu_*.json` | per-host baseline runs |
| `three_node_*.json` | distributed runs |
| `correctness.json` | output comparison against single-GPU |

Nothing here is written unless a real run produced it. A blocked experiment writes
a `blocked_reason` instead of a number.
