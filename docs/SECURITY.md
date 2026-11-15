# Security

MESHWEAVE is a research environment, but it runs on machines rented from strangers.
This document describes the threat model, what is enforced in code, and — just as
importantly — what is *not* protected, and why that is a deliberate choice.

---

## 1. The core assumption

**Every independently hosted GPU node is potentially untrusted.** This comes
directly from the project brief and it drives every decision below.

A Vast host is a machine owned by someone else, running a hypervisor we did not
configure, with a GPU we cannot audit. It could inspect model weights, log prompts,
return wrong results, or disappear at any moment.

MESHWEAVE's position: we can verify *computational correctness* (did the model
produce the right output for these weights and this input?) separately from
*data confidentiality* (can we stop the host from seeing what we sent it?).
MESHWEAVE addresses the first. It does not solve the second, and it does not
pretend to.

---

## 2. What is enforced

### 2.1 Node identity

Every agent has an X.509 certificate with the node id in its CN
(`meshweave-node:<node-id>`). The coordinator binds a TLS connection to that
identity.

Generate with:

```bash
uv run meshweave certs --node-id node-a --out ./secrets
```

This creates a local CA (`ca.crt`/`ca.key`) and a node certificate
(`node-a.crt`/`node-a.key`). In production, use a real CA; the local one exists so
a research cluster can be stood up without one.

**Known gap:** the coordinator does not yet verify that the CN matches a registered
node id on every request. Registration records the node; the binding is by
certificate possession. Tightening this is P8 work.

### 2.2 Workload-scoped credentials

Each workload gets a short-lived HMAC-signed token, valid for 15 minutes, scoped to
one workload on one node (`meshweave/communication/auth.py`). Tokens are verified by
signature and expiry; a token for node A cannot be used on node B.

### 2.3 Closed control surface

The agent's control API is a fixed set of routes (`ping`, `assign`, `activation`,
`sample`, `memory`, `cancel`, `drain`). There is:

- no shell execution endpoint,
- no file upload,
- no arbitrary code execution,
- no generic "run this" primitive.

The coordinator cannot make an agent do anything outside that set. This is enforced
by the absence of the route, not by a permission check that could be bypassed.

### 2.4 No credential leakage

Vast credentials come from `VAST_API_KEY` or a secrets file excluded by
`.gitignore`. They are never read from the source tree, never logged, and never
returned by an API. `.gitignore` excludes `*.pem`, `*.key`, `*.crt`, `secrets/`,
and `creds/`.

### 2.5 TLS everywhere

Both the control plane and the activation path run over TLS. `httpx` clients verify
certificates; the agent can be configured with `--tls-ca` to pin the coordinator's
CA.

**Known gap:** local development defaults to no TLS for the coordinator (plain
HTTP on 127.0.0.1). That is acceptable for a laptop and not acceptable for a real
cluster. The CLI requires `--tls-cert`/`--tls-key` for any non-loopback bind; this
is documented but not yet enforced in code.

### 2.6 The echo endpoint is harmless by construction

The network benchmark needs a raw TCP endpoint to measure throughput. It opens port
9000 on each agent. That endpoint echoes bytes and does nothing else — no parsing,
no state, no execution. An attacker who reaches it can only make the host echo bytes
they already sent. It needs no authentication because it cannot do anything worth
authenticating.

### 2.7 Resource quotas

Not implemented. A malicious node could report a huge VRAM figure and cause the
planner to place a model it cannot hold. The mitigation is that the agent's
reported inventory is treated as a claim, not a fact, and the planner records
per-node reliability. Full quota enforcement is P8 work.

---

## 3. What is not protected

Stated explicitly, because a security document that overclaims is worse than one
that admits its limits.

### 3.1 A node can read the weights it is given

If MESHWEAVE sends a model shard to node A, node A's operator can copy it. There is
no trusted execution environment, no confidential computing, no enclave.

**Consequence:** MESHWEAVE is suitable for open-weight models. It is not suitable
for proprietary weights on untrusted hardware without additional protection that
this project does not implement.

### 3.2 A node can read the prompts routed to it

The same applies to inference inputs. A node processing a request sees the prompt.

**Consequence:** do not send sensitive prompts through untrusted nodes.

### 3.3 A node can return wrong results

A malicious node could return plausible but incorrect tokens. MESHWEAVE verifies
that output *format* is valid and that the pipeline completed; it does not verify
that the computation was performed correctly.

Detecting this would require either replication with comparison (expensive) or a
cryptographic proof of correct inference (not practical for transformers at this
scale). Not implemented.

### 3.4 A node can lie about its hardware

The registry trusts reported telemetry. A node claiming 80 GiB of VRAM will be
planned against as if it had 80 GiB. This is a liveness problem as much as a
security one: the plan fails at load time rather than at planning time.

### 3.5 The coordinator's secret is a bearer token

Anyone with the coordinator's signing secret can mint workload tokens. The secret
is stored in `reports/.coordinator_secret` with mode 0600. Protect that file.

### 3.6 No rate limiting on agent endpoints

A malicious node could hammer the coordinator with heartbeats. No rate limiting is
implemented.

---

## 4. Threat model summary

| Threat | Mitigated? | How |
|---|---|---|
| Unauthenticated node joins the cluster | Partially | certificate required; CN binding not yet enforced |
| Node impersonates another node | Partially | certificate identity; workload tokens are node-scoped |
| Credential theft from logs | Yes | credentials never logged; gitignore excludes secrets |
| Coordinator remote code execution via agent | Yes | closed control API, no shell primitive |
| Malicious node reads model weights | **No** | accepted; open weights only |
| Malicious node reads prompts | **No** | accepted; no sensitive prompts |
| Malicious node returns wrong results | **No** | accepted; format checked, correctness not |
| Malicious node lies about hardware | **No** | accepted; fails at load time |
| Denial of service via heartbeat flood | **No** | not implemented |

---

## 5. Deployment checklist

Before running MESHWEAVE on real rented hosts:

- [ ] Generate node certificates from a CA you control, not the local dev CA.
- [ ] Enable TLS on the coordinator (`--tls-cert`/`--tls-key`).
- [ ] Set `MESHWEAVE_COORD_ALLOW_INSECURE_REGISTRATION=false`.
- [ ] Restrict the coordinator to a private network or a WireGuard overlay.
- [ ] Restrict the agent port to the minimum required; do not expose 9000 publicly
      beyond the benchmark window.
- [ ] Confirm every node's reported telemetry matches `nvidia-smi` before trusting
      a plan.
- [ ] Only run open-weight models on untrusted hardware.

---

## 6. Reporting a vulnerability

This is a research project without a security team. Open an issue describing the
issue and the affected component. Do not open a public issue for an exploitable
vulnerability in a deployed instance; describe the impact and contact the
maintainer privately.