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
v1.0.6). Two earlier revisions of this script therefore tested the wrong
software, and then called an `install.sh` that only exists on `main`, not on the
v0.1.2 tag. Both runs are discarded.

The tag's documented source install is an editable install of the `gpu` extra:

```sh
git clone --depth 1 --branch v0.1.2 https://github.com/GradientHQ/parallax.git
cd parallax
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gpu]'      # sglang[all]==0.5.5, torch, mlx — several GB
```

That is what the script now does. It requires Python 3.11–3.13, and it pulls
several GB of CUDA wheels — `sglang[all]`, torch, `nvidia_cusolver_cu12` and
friends.

**Expect download trouble.** A first attempt on `gpu-node-1` failed six times on
a single 267 MB wheel (`nvidia_cusolver_cu12`) that stalled at 253.8 MB, which is
a connectivity limit on that host, not a hardware incompatibility. The script now
uses `uv` with a 600 s timeout and retries the install up to four times, since uv
resumes partial downloads. If all four attempts fail, the recorded reason is
`dependency install failed (network or disk)` and the log is in
`/opt/parallax-spike/install.log`.

That distinction matters: an install failure says nothing about whether Parallax
runs on Turing, and must not be recorded as if it did.

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

## Outcome

Run on `gpu-node-1` (RTX 2070 SUPER, sm_75): **blocked on disk**. The instance
has 16 GB with 6.4 GB of base image, and the install needs about 8 GB of
unpacking space plus several GB installed. It failed with
`No space left on device` while installing `networkx`, after successfully
downloading torch, cuDNN and cuBLAS.

That is a node-specification result, not a hardware one: it says this instance
class cannot host a stack of this size, not that Parallax fails on Turing. The
full record is in `docs/PARALLAX_SPIKE_RESULT.md`.

## Not covered by this spike

Multi-node operation, which needs its scheduler, Lattica and either direct
connectivity or Gradient's relays. That is a separate decision, and only worth
taking if the single-node result is positive.