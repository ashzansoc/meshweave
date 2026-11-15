# Phase 0 — Safety Audit

Status: **PASSED.** Required before any remote execution.

This audit responds to the incident in which a cleanup routine killed `sshd` on
all three Vast hosts. The cause was a defect I introduced: a helper that parsed
every `pid=` from `ss -tlnp` and killed them whenever the target port appeared
anywhere in the output. It terminated every listening process on the machine.

---

## Findings

### Dangerous patterns removed

| Pattern | Status |
|---|---|
| `pkill` | absent from shipped code |
| `killall` | absent |
| `pkill -f <pattern>` | absent |
| PID extraction from unrestricted `ss` output | **removed** |
| Port-based process termination | **removed** — no such API exists |
| `fuser -k`, `lsof -k` | absent |
| Firewall operations (`iptables`, `ufw`, `nft`) | absent |
| SSH configuration writes (`sshd_config`, `authorized_keys`) | absent |
| Automatic reboot/shutdown | absent |

### Remaining process termination, and why it is safe

Two call sites terminate a process. Both act only on a process this run started.

| Location | Target | Scope |
|---|---|---|
| `meshweave/remote/ssh.py` | the local `ssh` client subprocess | spawned by `run()` on timeout |
| `experiments/network/exp_net_002.py` | the local `ssh` client subprocess | spawned by `EchoServer.start()` |

These terminate the **local** ssh client. The remote command dies with its parent
session, which is the correct semantics for a process we own. Neither can reach a
remote daemon.

`EchoServer.stop()` additionally refuses to terminate a listener marked
``reused``, so a listener from a previous run is never killed.

### The Vast reboot API is defined but never called

`meshweave/providers/vast/client.py` defines `reboot()`, mirroring the documented
REST endpoint. A grep confirms **no caller anywhere in the codebase**. It is an
explicit API surface a human must invoke, not an automatic action, so it is
permitted. `test_vast_reboot_is_never_auto_invoked` enforces that it stays that
way.

---

## Controls added

### `meshweave/remote/process_registry.py`

Every experiment-created process is recorded at creation with its exact PID, node
id, experiment id, command, and start time.

`ProcessRegistry.terminate()` refuses when:

- the PID was not recorded by MESHWEAVE,
- the PID belongs to a different experiment,
- the process no longer exists (returns False rather than raising),
- permission is denied.

There is no method to kill by port, by name, or in bulk. The API surface itself is
the control.

### `tests/test_safety_audit.py` — 29 tests

**Behavioural tests** spawn a real unrelated process and assert it survives:

- an unrecorded PID cannot be terminated
- a recorded process from another experiment cannot be terminated
- `terminate_experiment` leaves other experiments' processes running
- terminating a dead process is not an error
- invalid PIDs are refused
- registries are isolated from each other

**Static tests** parse the AST of every shipped file and assert:

- no `pkill`, `killall`, `killall5`, `fuser -k`, `lsof -t -k`
- no `systemctl`, `service ssh`, `iptables`, `nft`, `ufw`, `firewall-cmd`
- no `shutdown -h`, `shutdown now`, `init 0`, `init 6`
- no reads of the private key path
- no writes to `authorized_keys`, `sshd_config`, `/etc/ssh/`
- `reboot()` is never invoked outside its definition

AST parsing rather than grep is deliberate: it does not match the *word* `pkill`
inside a comment that explains why it is forbidden.

---

## Two incidents, one root cause

This is the second process-management error in this phase. The first was
`pkill -f mw_echo`, which matched the SSH command line itself and killed its own
session.

Both are the same mistake: **acting on a name pattern instead of a specific PID.**
The registry exists so that pattern cannot recur — there is no pattern-based path
left to take.

---

## Rules now enforced

1. MESHWEAVE terminates only a process it started, by recorded PID.
2. Ownership must be established or termination is refused.
3. No firewall modification.
4. No SSH configuration modification.
5. No instance stop, restart, or reboot.
6. No destructive command without explicit authorisation.

Verified by `uv run pytest tests/test_safety_audit.py -q` → 29 passed.