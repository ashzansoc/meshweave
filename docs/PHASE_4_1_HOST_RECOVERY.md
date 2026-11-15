# Phase 4.1 — Host Recovery

## Status: HOST-B available. HOST-A and HOST-C require operator action.

| Host | GPU | Total | Free | Status |
|---|---|---|---|---|
| HOST-A | GTX 1660 SUPER | 6,144 MiB | 276 MiB | **blocked** |
| HOST-B | Quadro RTX 4000 | 8,192 MiB | 7,786 MiB | **available** |
| HOST-C | RTX 2070 SUPER | 8,192 MiB | 2,270 MiB | **blocked** |

## Investigation performed

I did not refuse on principle — I attempted to establish ownership of the blocking
allocations (pid 1482890 on A, 390673 on C):

| Check | Result |
|---|---|
| `/proc/<pid>` exists | **NO** on both hosts |
| PIDs visible in container namespace | 43 (A), 37 (C) — neither present |
| `ps -p <pid>` | cannot see them |
| `/proc/<pid>/cmdline` | empty |
| `/proc/<pid>/exe`, `cwd`, `status`, start time | unreadable |

**Conclusion:** these allocations are outside this container's PID namespace. They
are host-level or belong to another container on the shared physical machine —
HOST-A and HOST-C are known from Phase 2A to share one host.

They **cannot** be terminated from inside these containers, and ownership cannot be
established. Refusing to act is the correct outcome: the standing rule exists
because an earlier cleanup killed `sshd` on all three hosts.

## What is preserved

Verified present on all three hosts, unchanged:

- `phi-mini-moe-Q6_K.gguf` — 6.306 GiB
- `phi-mini-moe-Q8_0.gguf` — 7.576 GiB
- `llama.cpp` build with CUDA and `ggml-rpc-server`
- All Phase 2A / 3 / 4 reports and benchmark data in the repository

Disk free: 1.4–1.5 GB per host.

## Action required

**Restart the HOST-A and HOST-C instances.** A stop/start clears container-local
processes and GPU contexts; the data disk survives, so nothing listed above is lost.

Do not ask me to reboot or stop instances — that requires your explicit authorisation
and I will not do it autonomously.

After restart, verify:

```bash
ssh meshweave-host-a 'nvidia-smi --query-gpu=memory.free --format=csv,noheader'  # expect >3500 MiB
ssh meshweave-host-c 'nvidia-smi --query-gpu=memory.free --format=csv,noheader'  # expect >3500 MiB
```

Conservative requirement: each blocked host needs at least **3.5 GiB free** for its
assigned Q8_0 layers (~3.18 GiB peak plus headroom).

## Then

```bash
MESHWEAVE_MODEL=/root/models/phi-mini-moe-Q8_0.gguf \
  python -m experiments.model_execution.pipeline_node stage --index N --layers auto ...
```
