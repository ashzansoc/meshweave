# MESHWEAVE VDC benchmarks

Model: `Qwen/Qwen2.5-0.5B-Instruct`  
Cluster nodes: gpu-node-1, brisk-thistle-9ae2  
Measured: 2026-11-18 20:52:10

| Configuration | tok/s | ms/token | TTFT ms | total s |
|---|---|---|---|---|
| single_1 (gpu-node-1) | 19.024 | 38.8 | 368.0 | 1.262 |
| single_2 (brisk-thistle-9ae2) | 16.671 | 41.0 | 496.6 | 1.44 |
| distributed (gpu-node-1, brisk-thistle-9ae2) | 2.926 | 310.8 | 1053.2 | 8.201 |

## What the numbers say

Distributed is 6.5x **slower** than the faster single-node configuration for the same prompt and length. This is measured, not projected.

Distributed execution is a capacity mechanism on these links: it lets a model that does not fit one GPU run at all. It does not make a model that already fits run faster.
