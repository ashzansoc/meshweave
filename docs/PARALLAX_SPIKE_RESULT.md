# Parallax spike — result

**Date:** 2026-11-18 · **Host:** `gpu-node-1` (RTX 2070 SUPER, sm_75, 8 GiB)
**Verdict: BLOCKED on disk. The Turing question is still open.**

This is an environmental result, not a hardware one. It says the instance class
cannot host Parallax's GPU stack, which is a different finding from "Parallax
does not run on Turing" — and it must not be recorded as if it were the latter.

---

## What was attempted

The supported source install for the `v0.1.2` tag, in an isolated environment:

```sh
git clone --depth 1 --branch v0.1.2 https://github.com/GradientHQ/parallax.git
python3 -m venv .venv
pip install -e '.[gpu]'      # sglang[all]==0.5.5, torch, mlx
```

Nothing outside `/opt/parallax-spike` was touched. `uv` was used with a 600 s
HTTP timeout, up to four attempts, and its cache and temp directory confined to
the spike directory.

## What happened

**Disk, measured on the instance:**

| | |
|---|---|
| Overlay size | 16 GB |
| Base image in use before the spike | 13 GB (3.6 GB free) |
| Reclaimed by clearing `/root/.cache/pip` | 6.1 GB |
| Free before install | 9.6 GB |
| uv temp usage during install | **7.9 GB** |
| Free at failure | ~1.6 GB |

**The failure:**

```
Prepared 141 packages in 5m 24s
error: Failed to install: networkx-3.7-py3-none-any.whl (networkx==3.7)
  cause: Failed to copy to .../site-packages/networkx/algorithms/tree/branchings.py
  cause: No space left on device (os error 28)
```

The download phase completed — torch, cuDNN and cuBLAS all arrived — so PyPI
connectivity was adequate once retries and resume were in place. The wall was
storage: `sglang[all]` plus torch needs roughly 8 GB of unpacking space *and*
several GB installed, and the container has 16 GB with 6.4 GB already used by the
base image.

Two earlier attempts failed for unrelated reasons that are recorded here so the
sequence is not misread: one installed an unrelated PyPI package called
`parallax` (an SSH tool), and one called an `install.sh` that exists on `main`
but not on the `v0.1.2` tag.

## What this does and does not tell us

**Does tell us:**

* Parallax's GPU stack is heavy: `sglang[all]==0.5.5` + torch, roughly 8 GB of
  unpacking and several GB resident.
* A 16 GB Vast container with a 6–8 GiB GPU **cannot host it**. This is a node
  specification constraint, and it applies to any serving stack of this size —
  vLLM and TensorRT-LLM are comparable.
* Clearing caches is not enough: even with 9.6 GB free the install needs more.

**Does not tell us:**

* Whether Parallax runs on sm_75. That needs an instance with enough disk, not a
  different GPU.
* Whether its scheduler or Lattica layer works over a 226 ms link.

SGLang's own documentation makes no statement about Turing either way — no
support matrix entry, no minimum compute capability. So the framework-level
answer remains **UNKNOWN**, and the compatibility matrix records it as such
rather than guessing.

## What would unblock it

Any one of:

* A Vast instance with a larger disk (the 16 GB images are the constraint, not
  the GPU).
* A smaller execution backend for the spike — Parallax's `vllm` extra instead of
  `gpu`, if vLLM 0.11 installs in the space available. That would test a
  different serving layer, so it answers a narrower question.
* Accepting the unknown and relying on the native runtime, which is verified on
  this hardware and adds no dependency.

## Cleanup

`/opt/parallax-spike` was removed from the host, along with the pip cache that
was cleared to make room. No MESHWEAVE files, services or model caches were
touched; the node remained healthy throughout.

## Effect on the compatibility matrix

Parallax moves from "unproven, spike pending" to **"unproven; blocked on this
instance class (16 GB disk)"**. It stays a candidate for a larger instance and
remains out of the default path.