# Phase 2A — Blocked at P0: SSH authorisation

**Status: BLOCKED.** All three hosts are reachable, but no key on this machine is
authorised, and the Vast API key supplied cannot authorise one either.

Nothing after P0 can run until one of the unblock paths below is completed.

---

## What is working

### Host reachability (measured)

| Host | Endpoint | TCP + SSH handshake | Auth result |
|---|---|---|---|
| HOST-A | `1.208.108.242:51400` | succeeds | `Permission denied (publickey)` |
| HOST-B | `83.63.1.187:24014` | succeeds | `Permission denied (publickey)` |
| HOST-C | `1.208.108.242:46079` | succeeds | `Permission denied (publickey)` |

The distinction matters: **the network path is fine.** This is authorisation, not
connectivity.

The server offers `publickey` as the only accepted method, so there is no password
fallback:

```
debug1: Authentications that can continue: publickey
debug1: Offering public key: /Users/ashutosh/.ssh/id_rsa_vastai RSA SHA256:hdPjtTogx...
root@1.208.108.242: Permission denied (publickey).
```

### Keys tried, all rejected

| Key | Result |
|---|---|
| `id_rsa_vastai` | rejected |
| `id_ed25519` | rejected |
| `id_rsa` | rejected |
| `aic-gpu-codmw-4.pem` | rejected |
| `google_compute_engine` | rejected |

---

## The supplied Vast API key: what it can and cannot do

The key was tested against every endpoint MESHWEAVE needs. The results are not
uniform, and the distinction is the whole story.

| Endpoint | Result | Meaning |
|---|---|---|
| `POST /api/v0/bundles/` | **400** (validation) | **auth passed** — marketplace search works |
| `GET /api/v0/users/current/` | **401** | 2FA required |
| `GET /api/v1/instances/` | **401** | 2FA required |
| `GET /api/v0/instances/{id}` | **401** | 2FA required |
| `POST /api/v0/ssh` | **401** | 2FA required |

The failure message is identical and specific:

```json
{"success": false,
 "msg": "Authorization Error. Your key lacks proper privileges to perform this
         action as it requires you to have logged in using Two Factor Authentication"}
```

So the key is **valid** and **can** search the marketplace. It **cannot** list
instances, read instance detail, or create SSH keys, because Vast gates those
behind a 2FA-verified session.

### Consequence

MESHWEAVE cannot attach its public key through the API. That needs either a
2FA-verified session or manual action in the console.

Note also that `GET /api/v0/users/current/` failing means MESHWEAVE cannot even
confirm which account the key belongs to, so it is not possible to tell from here
whether the three instances belong to this account at all.

---

## The key to install

Public half — safe to publish:

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCgogaI1sqTAykLdFhyO7Uioy5w1GLP1ZLuN+IOR6Q6F+on+Se5CQZpUWxwoOyQs3DIDggfFGoZVpkCeTSDitKwLkhDIh37HEExDvqEz5WLZsonrfFMwoYTOPnNnP5/cA2FnB42X1THzde5BLK3U//A2dPOMwkMZsL8VV7RDOcufx1iXIslSmOq7dDlXS+DEPm51kYdVo9vwds/libO6HM7S53xCwkxJvn1gdOIlcLcuGG/3CMe/LM+VRs6G7h4jP4k75ff6rff8ZOvGTVnKN8rxg6O2m5U0Vx8QFFV2thJ/trEm18x5/XN/LU0qAwTqwXgcN2gnzlcnYWPpjEERyTvnEollGMQd/3Kgk9IVkUEmuj2keQokLz0QpJRRexYsyeuAfKtWAItcgDe7qciiKtBNDByCCngJemr9EmDVxsDyrSt+ta1cuqp67CFCi1HQUV4z6nV6LJOA8/xQ2Byqd/REsoRgY48ByvQgXoFSUDLzIivlkMijWar7/XzMp3rGcedWX/C9H9ae2SC2lgm22Crd+S27/z2Pg3pqZolMEyAAyKiL+CdZFOcNP9tuXcDpqzwcJ00AI3op/n3Glwz8xVZFYJsh9UNACihrq/Ci+S+nVJB8RovDQvnoZf8Bmc5nuJ0rS7c09LXXNL/HHyv+6zReGhPoWVy7ShshLOEnGI6Ow== vastai-Ashutoshs-MacBook-Pro.local-20261115
```

Fingerprint: `SHA256:hdPjtTogxjxkIb80426q8BbDEOb1+4r8qdfC2h11bp4`

---

## Three ways to unblock

### A. Install the key from the Vast console (most reliable)

For each instance: open it, go to **SSH Keys**, and attach the key above. If the
console offers a web terminal instead, paste the line into
`~/.ssh/authorized_keys` and `chmod 600 ~/.ssh/authorized_keys`.

This bypasses the 2FA restriction entirely, because the console session is
already authenticated.

### B. Supply a different private key

If the instances were created with a key pair you still hold, point me at it. No
code change is needed — the key path is a parameter:

```bash
uv run meshweave agent --node-id ... --key ~/.ssh/your_key
```

Vast's "create ssh-key" flow produces exactly this, and the private key is
downloadable once.

### C. Create an unrestricted API key after a 2FA login

If the account's 2FA can be completed and a key minted in that session, the
instance-management endpoints may open up. The 401 suggests the current key was
created without that privilege, so a new one may behave differently.

---

## What is already built

Phase 2A infrastructure is complete, tested, and waiting on credentials. All of
it runs the moment auth works.

| Module | Purpose | Tested |
|---|---|---|
| `meshweave/remote/ssh.py` | async SSH exec, `BatchMode`, exit-255 vs command-failure distinction, MOTD stripping | yes |
| `meshweave/remote/hosts.py` | three hosts as data; node-id identity, never IP | yes |
| `meshweave/network/colocation.py` | evidence-based co-location verdicts | yes |
| `meshweave/network/classify.py` | link classes from measured RTT and loss | yes |
| `meshweave/network/overlay/addresses.py` | subnet allocation + conflict detection | yes |
| `meshweave/network/overlay/wgconfig.py` | WireGuard config generation | yes |
| `experiments/network/exp_net_001.py` | discovery harness, verifies GPU from `nvidia-smi` | runs, blocks cleanly |

21 new tests, 65 total, all passing. `ruff` clean, `mypy --strict` clean on 49
files.

### One decision already settled by the tests

**A shared public IP is not evidence of co-location.** HOST-A and HOST-C both
report `1.208.108.242`, but the classifier never treats that as proof. It only
raises a verdict when strong hardware identifiers agree *and* the measured RTT is
consistent with a local link. The test named
`test_shared_ip_is_not_evidence_by_itself` exists specifically to keep that true.

---

## First commands once unblocked

```bash
# P0 — verify auth, write ~/.ssh/config aliases
uv run python -m experiments.network.exp_net_001 --hosts all

# P1 — hardware inventory, GPU verified against nvidia-smi
cat experiments/network/raw/host_inventory.json
```

Then P2 (direct TCP across all six directions) gates whether any overlay software
is needed at all. No WireGuard is installed before that test.

---

## Security note

The API key supplied for this phase was used in memory only, via an environment
variable, and was never written to a file, a log, or the repository. Verified with
a repository-wide search before committing.