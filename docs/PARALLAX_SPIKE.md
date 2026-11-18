# Parallax Turing spike

The research documents settle everything about Parallax that source can settle.
One question remains, and only hardware can answer it: **does it load and run a
model on a Turing GPU (sm_75)?**

Parallax's GPU database contains only A100, RTX 4090 and RTX 5090, and its docs
list Blackwell, Ampere and Hopper. Turing is absent from both. Everything else
about the assessment — architecture, maturity, licensing — is in
`docs/PARALLAX_TECHNICAL_RESEARCH.md`.

## Running it

On a GPU host, as root:

```bash
curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/parallax_spike.sh \
  | sudo bash
```

**Install method matters.** Parallax is not published on PyPI under this name —
the `parallax` package there is an unrelated SSH fan-out tool (krig/parallax,
v1.0.6). The supported install is a clone plus the project's own `./install.sh`,
which uses `uv` to create the venv. An earlier revision of this script used
`pip install parallax` and therefore tested entirely the wrong software; that run
is discarded and must be repeated.

It is deliberately contained:

* installs into `/opt/parallax-spike/venv`, never touching `/opt/meshweave`
* runs **one node** with no `--join`, so no ports are published and no peer
  traffic is needed
* binds `127.0.0.1` only
* writes its verdict to `/var/log/parallax-spike.json`
* `sudo bash deploy/parallax_spike.sh --uninstall` removes the environment, the
  process and the result

Parallax ships no authentication on its HTTP ports (3000–3002). That is the
reason the spike is single-node and local: a published port would be an open
inference endpoint.

## What it records

| Field | Meaning |
|---|---|
| `gpu` | `nvidia-smi` name, compute capability, VRAM, driver |
| `parallax_version` | what pip installed |
| `server_ready` | whether it bound the port and answered `/v1/models` |
| `text` | the generated completion, if any |
| `error` | the failure, if any |
| `server_log_tail` | the last 25 lines, for the real reason |

## Reading the result

* **`"ok": true`** — it loads and generates on Turing. Parallax becomes a real
  candidate for the WAN tier, and the next decision is whether to adopt its
  scheduler or only its single-node serving.
* **Fails at install** — dependency incompatibility on this platform; record it
  and move on.
* **Fails at load or generation** — most likely a kernel that does not support
  sm_75. Record the exact error; that is the answer.

Whatever the outcome, it goes into the compatibility matrix as a measured result
rather than an assumption.

## Not covered by this spike

Multi-node operation, which needs its scheduler, Lattica and either direct
connectivity or Gradient's relays. That is a separate decision, and only worth
taking if the single-node result is positive.