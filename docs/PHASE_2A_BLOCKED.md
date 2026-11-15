# Phase 2A — Blocked at P0: SSH authentication

**Status: BLOCKED.** No further phase can run until a public key is authorised on
the three instances. Everything downstream depends on remote command execution.

This document exists because the brief requires blocked results to be documented
rather than papered over.

---

## What was established

### The hosts are reachable

All three endpoints complete a TCP connection and an SSH protocol handshake. This
is measured, not assumed: `ssh` reached the authentication stage and received the
server's list of accepted methods.

```
host-a  1.208.108.242:51400  ->  Permission denied (publickey)
host-b  83.63.1.187:24014    ->  Permission denied (publickey)
host-c  1.208.108.242:46079  ->  Permission denied (publickey)
```

Exit code 255 with `Permission denied (publickey)` means: the TCP connection
succeeded, the SSH handshake succeeded, the server offered `publickey` as the only
accepted method, our key was offered, and the server rejected it.

**This is an authorisation problem, not a connectivity problem.** That distinction
matters: the network path to all three hosts is fine.

### The server accepts publickey only

Captured from a verbose handshake:

```
debug1: Authentications that can continue: publickey
debug1: Next authentication method: publickey
debug1: Offering public key: /Users/ashutosh/.ssh/id_rsa_vastai RSA SHA256:hdPjtTogxjxkIb80426q8BbDEOb1+4r8qdfC2h11bp4 explicit
debug1: Authentications that can continue: publickey
root@1.208.108.242: Permission denied (publickey).
```

Password authentication is not offered, so there is no fallback path.

### Keys tried, and the result

| Key | Fingerprint | Result |
|---|---|---|
| `id_rsa_vastai` | `SHA256:hdPjtTogxjxkIb80426q8BbDEOb1+4r8qdfC2h11bp4` | rejected |
| `id_ed25519` | — | rejected |
| `id_rsa` | — | rejected |
| `aic-gpu-codmw-4.pem` | — | rejected |
| `google_compute_engine` | — | rejected |

No `vastai` CLI is installed and no `~/.config/vastai` directory exists, so there
is no provider-side credential that could mint or attach a key automatically.

---

## The key that needs installing

This is the public half of the key MESHWEAVE will use. It is safe to publish; the
private half stays on this machine.

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCgogaI1sqTAykLdFhyO7Uioy5w1GLP1ZLuN+IOR6Q6F+on+Se5CQZpUWxwoOyQs3DIDggfFGoZVpkCeTSDitKwLkhDIh37HEExDvqEz5WLZsonrfFMwoYTOPnNnP5/cA2FnB42X1THzde5BLK3U//A2dPOMwkMZsL8VV7RDOcufx1iXIslSmOq7dDlXS+DEPm51kYdVo9vwds/libO6HM7S53xCwkxJvn1gdOIlcLcuGG/3CMe/LM+VRs6G7h4jP4k75ff6rff8ZOvGTVnKN8rxg6O2m5U0Vx8QFFV2thJ/trEm18x5/XN/LU0qAwTqwXgcN2gnzlcnYWPpjEERyTvnEollGMQd/3Kgk9IVkUEmuj2keQokLz0QpJRRexYsyeuAfKtWAItcgDe7qciiKtBNDByCCngJemr9EmDVxsDyrSt+ta1cuqp67CFCi1HQUV4z6nV6LJOA8/xQ2Byqd/REsoRgY48ByvQgXoFSUDLzIivlkMijWar7/XzMp3rGcedWX/C9H9ae2SC2lgm22Crd+S27/z2Pg3pqZolMEyAAyKiL+CdZFOcNP9tuXcDpqzwcJ00AI3op/n3Glwz8xVZFYJsh9UNACihrq/Ci+S+nVJB8RovDQvnoZf8Bmc5nuJ0rS7c09LXXNL/HHyv+6zReGhPoWVy7ShshLOEnGI6Ow== vastai-Ashutoshs-MacBook-Pro.local-20261115
```

Fingerprint: `SHA256:hdPjtTogxjxkIb80426q8BbDEOb1+4r8qdfC2h11bp4`

---

## How to unblock

### Option A — install the key via the Vast console (simplest)

For each instance, add the public key above to its SSH keys. On Vast, **account
keys only apply to instances created afterwards**, so for already-running
instances use the per-instance key attach in the console, or:

```bash
# per-instance attach, for each instance id
vastai attach-ssh <instance_id> ~/.ssh/id_rsa_vastai.pub
```

### Option B — install the key from a working session

If you can reach any one of the instances another way (console terminal, Vast's
web SSH, or an existing key), append the public key:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCgogaI1sqT...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Option C — supply a different private key

If the instances were created with a different key pair, give me that key and I
will use it instead. No code change is needed; the key path is a parameter.

---

## What is already built and waiting

Phase 2A infrastructure is complete and tested; it only needs credentials to run.

| Module | Purpose | Tested |
|---|---|---|
| `meshweave/remote/ssh.py` | async SSH exec, `BatchMode`, per-command timeout, exit-code surfaced | yes |
| `meshweave/remote/hosts.py` | the three hosts as data, node-id identity | yes |
| `meshweave/network/colocation.py` | evidence-based co-location verdicts | yes |
| `meshweave/network/classify.py` | link classes from measured RTT and loss | yes |
| `meshweave/network/overlay/addresses.py` | subnet allocation + conflict detection | yes |
| `meshweave/network/overlay/wgconfig.py` | WireGuard config generation | yes |

`tests/test_phase2a_network.py` — 21 tests, all passing.

One design decision already settled by those tests: **a shared public IP is not
evidence of co-location.** The classifier only upgrades a verdict when strong
hardware identifiers agree *and* measured RTT is consistent with a local link.
That matters here because HOST-A and HOST-C share `1.208.108.242`.

---

## What happens once the key works

Per the approved plan, in order:

1. **P0** — verify auth on all three hosts, write `~/.ssh/config` aliases.
2. **P1** — hardware inventory via SSH; verify GPU model from `nvidia-smi` rather
   than the Vast listing; capture `dmidecode`, CPU, PCI IDs, interfaces, routes.
3. **P2** — direct TCP connectivity across all six directions. **This gates the
   overlay**: if direct TCP is reliable, MESHWEAVE uses it and WireGuard becomes
   optional. No networking software is installed before this test.
4. **P3** — UDP capability, `/dev/net/tun`, `PermitTunnel`.
5. **EXP-NET-011** — co-location verdict for A↔C with evidence.
6. **P4–P8** — overlay if justified, benchmarks, persistent transport, dashboard,
   `docs/PHASE_2A_NETWORK_VALIDATION.md`.

The first command I will run after unblocking:

```bash
uv run python -m experiments.network.exp_net_001 --hosts all
```