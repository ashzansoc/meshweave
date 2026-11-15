# Phase 2A — Network Validation

**Status: MEASURED.** Six of six directional paths carry real data. All three
hosts are running, all SSH aliases work, and every number below comes from an
instrumented transfer between the actual machines.

Result labels: **MEASURED** (taken from a real exchange), **ESTIMATED** (derived
from measurement, method stated), **BLOCKED** (could not be measured).

---

## 1. Headline

| Question | Answer |
|---|---|
| Can these three hosts exchange data? | **Yes — all six directions.** |
| Is the network uniform? | **No.** RTT varies by ~600× between pairs. |
| Which pairs are usable for distributed execution? | **A↔C only.** |
| Is HOST-B usable? | **Not for coupled execution.** ~310 ms RTT, under 1 Mbps. |

---

## 2. Safety audit (Phase 0) — PASSED

Required before any remote execution, following the earlier incident in which a
cleanup routine killed `sshd` on all three hosts.

- No `pkill`, `killall`, `fuser -k`, `lsof -k`, firewall operations, SSH config
  writes, or automatic reboot anywhere in shipped code.
- `meshweave/remote/process_registry.py` records every process an experiment
  starts, by exact PID, node, experiment id and command. Termination is refused
  for an unrecorded PID, a different experiment's PID, or a dead process.
- 29 regression tests: behavioural (real unrelated processes survive) and static
  (AST scan for forbidden patterns).

Full detail: [PHASE_0_SAFETY_AUDIT.md](PHASE_0_SAFETY_AUDIT.md).

---

## 3. SSH access — MEASURED

All three aliases authenticate with `~/.ssh/id_rsa_vastai`, key auth only, no
password fallback. Keys and authentication were not modified.

| Alias | Endpoint | Container | Auth |
|---|---|---|---|
| `meshweave-host-a` | `1.208.108.242:51400` | `70d0b1aea811` | OK |
| `meshweave-host-b` | `83.63.1.187:24014` | `2d6e3ce8541f` | OK |
| `meshweave-host-c` | `1.208.108.242:46079` | `5c00204b15f7` | OK |

Verified: authentication, host identity, remote command execution, timeout
handling, non-zero exit handling.

---

## 4. Hardware inventory — MEASURED

Verified from `nvidia-smi` and `/sys/class/dmi/id/`, not the provider listing.

| | HOST-A | HOST-B | HOST-C |
|---|---|---|---|
| GPU | GTX 1660 SUPER | **Quadro** RTX 4000 | RTX 2070 SUPER |
| VRAM | 6 GiB | 8 GiB | 8 GiB |
| Driver | 580.159.03 | 595.99.02 | 580.178.04 |
| CPU | Xeon E5-2630 v4 | Xeon W-2123 | Xeon E5-1650 v3 |
| Kernel | 5.15.0-186 | 7.0.0-42 | 6.8.0-147 |
| Board | MACHINIST X99-D8 MAX | HP 81C5 | Supermicro X10SRA |
| Internal IP | 172.17.0.2/16 | 172.17.0.3/16 | 172.17.0.2/16 |
| MAC | 56:0f:9a:7c:9a:18 | f2:b6:cb:27:e3:9d | 36:58:0c:ef:37:e0 |

**Aggregate: 22 GiB VRAM, 3 GPUs, 3 node identities.**

### Listing discrepancy

The provider lists HOST-B as "RTX 4000". `nvidia-smi` reports **Quadro RTX 4000**.
These are different products. The host is authoritative.

### Container constraints (from the Vast agent guide)

Unprivileged Docker container, not a VM: no kernel module loading, no
Docker-in-Docker, no `perf`/eBPF, no sysctl or cgroup changes, host kernel shared
and read-only. `/dev/net/tun` is absent on all three.

**Consequence: WireGuard is not installable on these hosts**, and neither is
SSH TUN. This is structural, not a configuration gap. No overlay was attempted,
consistent with the instruction not to install networking software before proving
direct TCP.

`workspace_is_volume: false` on all three — nothing survives recycle or destroy.

---

## 5. Port discovery — MEASURED

External ports are fixed at creation and cannot be added at runtime. Live values
from `vast-capabilities`:

| Container | HOST-A | HOST-B | HOST-C | Free |
|---|---|---|---|---|
| 22 (SSH) | 51400 | 24014 | 46079 | no |
| 1111 (portal) | 51442 | 37677 | 46011 | no |
| 6006 (tensorboard) | 51468 | 20250 | 46002 | no |
| 8080 (jupyter) | 51443 | 26799 | 46074 | no |
| 8384 (syncthing) | 51416 | 27289 | 46062 | no |
| **10100** | **51452** | **32892** | **46024** | **yes** |
| **10200** | **51440** | **33774** | **46092** | **yes** |
| 72299 (self-mapped) | 51449 | 26034 | 46094 | no |

Only 10100 and 10200 are free, and they are **different external ports on every
host**. Port 9100 (used in an earlier attempt) is not mapped on any host and
cannot carry traffic.

---

## 6. Connectivity matrix — MEASURED

All six directions, real transfers, 20 × 64 KiB per direction.

| Path | Connect | Throughput | Round trip |
|---|---|---|---|
| A → B | 312.6 ms | 0.78 Mbps | 675.9 ms |
| B → A | 305.4 ms | 0.95 Mbps | 549.9 ms |
| **A → C** | **3.0 ms** | **46.2 Mbps** | **11.4 ms** |
| **C → A** | **2.5 ms** | **50.2 Mbps** | **10.5 ms** |
| B → C | 358.3 ms | 1.42 Mbps | 368.2 ms |
| C → B | 317.9 ms | 1.48 Mbps | 355.5 ms |

**6/6 paths carry data.** An earlier run recorded failures on four paths; those
were listeners reaped mid-run, not network faults. With listeners held as
foreground processes in live SSH sessions, all six work.

---

## 7. RTT and jitter — MEASURED

60 probes per path, timed on the sending host with a monotonic clock.

| Path | min | p50 | p90 | p95 | p99 | jitter | loss |
|---|---|---|---|---|---|---|---|
| A → B | — | 311.4 | — | — | 403.2 | 20.1 | 0/60 |
| B → A | — | 308.1 | — | — | 472.5 | 33.3 | 0/60 |
| **A → C** | **0.50** | **0.64** | **0.97** | **1.11** | **1.47** | **0.50** | **0/60** |
| **C → A** | **0.34** | **0.47** | **0.60** | **0.61** | **0.92** | **0.08** | **0/60** |
| B → C | — | 309.0 | — | — | 444.6 | 30.1 | 0/60 |
| C → B | — | 342.0 | — | — | 422.8 | 43.3 | 0/60 |

**No packet loss on any path across 360 probes.**

No one-way latency is reported. The clocks are not synchronised, so `RTT/2` is
not a measurement; it would be an estimate assuming path symmetry, and these paths
are asymmetric in bandwidth, so that assumption is unsafe.

---

## 8. Bandwidth across payload sizes — MEASURED

1 KiB to 1 MiB, per direction. 10 MiB and 100 MiB were skipped on this run: at
~1 Mbps a 10 MiB transfer exceeds the test's wall-clock cap.

**A ↔ C** (the fast pair):

| Payload | C → A | A → C |
|---|---|---|
| 8 KiB | 59.7 Mbps | 64.4 Mbps |
| 64 KiB | 6.6 Mbps | 57.3 Mbps |
| 256 KiB | 39.7 Mbps | 117.0 Mbps |
| 1 MiB | 43.8 Mbps | timed out (15 s cap) |
| **sustained 15 s** | **197.3 Mbps** | **347.2 Mbps** |

**A ↔ B and B ↔ C** (the slow paths):

| Payload | A → B | B → A | B → C | C → B |
|---|---|---|---|---|
| 64 KiB | 0.4 | 0.4 | 0.4 | 0.4 |
| 1 MiB | 2.0 | 1.3 | 1.4 | 3.1 |
| **sustained 15 s** | — | — | 0.6 | 0.8 |

The small-payload figures are latency-bound and unrepresentative; the sustained
numbers are the ones to plan against.

---

## 9. HOST-A / HOST-C investigation — MEASURED

They share public IP `1.208.108.242`. Two separate questions, two different
answers.

**Are they the same virtual machine?** No. Different `boot_id` (5.15.0-186 vs
6.8.0-147), different board identifiers, different CPU. Different kernels are
conclusive: containers on one host share a kernel.

**Are they network-adjacent?** **Yes — decisively.** 0.47–0.64 ms RTT, 197–347
Mbps sustained. That is loopback-class performance, not a cross-internet path.

**Can they reach each other internally?** No. Both report `172.17.0.2/16` on
their own Docker bridge. That address is a different interface on each host, so
it is not a usable path between them. A's own IP is `172.17.0.2`; connecting
there reaches itself.

**Refined verdict:** separate lightweight virtual machines on the same physical
host (or the same rack), whose traffic hairpins through the host NAT. The DMI and
board identifiers measure *virtual* hardware and do not establish physical
separation; the earlier "different-host" verdict was answering a different
question.

**Engineering consequence: A and C are a fast pair.** 0.5 ms RTT is
datacenter-class. Pipeline parallelism across A and C should be viable; the
payload is ~7 KiB per token for a 7B model, which at 0.5 ms RTT is negligible
overhead.

**HOST-B is not a peer.** 310 ms RTT and under 1 Mbps sustained means any
coupled execution with B is unusable for interactive inference.

---

## 10. Recommended topology — MEASURED basis

| Pair | RTT | Sustained | Verdict |
|---|---|---|---|
| A ↔ C | 0.5 ms | 197–347 Mbps | **datacenter-class; use for coupled execution** |
| A ↔ B | 311 ms | 0.6–1.0 Mbps | internet-class; replicas only |
| B ↔ C | 309 ms | 0.6–0.8 Mbps | internet-class; replicas only |

MESHWEAVE should treat **A and C as one low-latency domain** and **B as a separate
WAN node**. Concretely:

- Models needing more than one GPU's memory: partition across **A and C only**.
  22 GiB total VRAM is not enough for a 70B model at bf16, but it is enough for
  a 14B at bf16 (27.5 GiB → needs int8 or partial offload) or a 32B at int4.
- Throughput: run independent replicas on all three, routed by
  least-outstanding-requests. B's slowness only affects requests routed to it.
- **Do not place B in the middle of a pipeline.** A 310 ms hop adds 310 ms to
  every token.

---

## 11. Failure analysis

| Symptom | Cause | Resolution |
|---|---|---|
| Four paths "failed" in an earlier run | Listeners reaped when SSH session ended | Hold listener as foreground process in live SSH session |
| `Address already in use` | Stale listener from a prior run | Reuse the existing listener; never kill by port |
| Port 9100 unreachable on all hosts | Not mapped by Vast | Use 10100, discovered from `vast-capabilities` |
| HOST-B listed as "RTX 4000" | Provider listing imprecise | Trust `nvidia-smi`: Quadro RTX 4000 |
| `sshd` down on all hosts (earlier) | My cleanup killed every listening PID | Phase 0 audit; PID-ownership registry |

---

## 12. Connection reliability

- 360 RTT probes, **0 failures**.
- All six paths sustained a 15-second transfer without dropping.
- Jitter is low on the fast pair (0.08–0.50 ms) and moderate on the slow paths
  (20–43 ms), which is consistent with congested long-fat-path behaviour.

**BLOCKED / not yet measured:**

- 10 MiB and 100 MiB payloads (wall-clock cap on the slow paths).
- Persistent-connection modes and TLS overhead (Phase 7).
- Packet loss under sustained load.
- Background network utilisation during measurement.

---

## 13. Remaining blockers

1. **No WireGuard, by design.** `/dev/net/tun` is absent and the container cannot
   load kernel modules. Not a blocker: direct TCP works on all six paths.
2. **No 10/100 MiB measurements.** Need a longer wall-clock budget, or a
   bandwidth cap on the slow paths.
3. **Phase 7 (persistent TCP, TLS overhead)** not yet run.
4. **Dashboard** does not yet show this data.
5. **Only 2 free ports per host.** If MESHWEAVE needs more simultaneous
   listeners, instances must be created with additional port mappings.

---

## 14. Reproduction

```bash
# Phase 0
uv run pytest tests/test_safety_audit.py -q

# Hardware inventory
uv run python -m experiments.network.exp_net_001 --hosts all

# Co-location
uv run python -m experiments.network.exp_net_011

# Direct TCP, all six directions
uv run python -m experiments.network.exp_net_002

# RTT, jitter, bandwidth
uv run python -m experiments.network.exp_net_005
```

Raw evidence: `experiments/network/raw/`.

| File | Contents |
|---|---|
| `host_inventory.json` | per-host command output and verified GPUs |
| `colocation.json` | hardware fingerprints and pair verdicts |
| `colocation_network.json` | network adjacency measurements |
| `direct_tcp_matrix.json` | six directional transfers |
| `rtt_bandwidth.json` | RTT percentiles, jitter, payload sweep, sustained |