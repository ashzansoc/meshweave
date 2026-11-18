# MESHWEAVE Agent

Register any Linux machine as a MESHWEAVE compute host with one command.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/install.sh \
  | sudo bash -s -- \
      --coordinator https://<coordinator-host>:8700 \
      --token <registration-token> \
      --enroll
```

That is the whole command, and it needs no node name. One is generated
(`silver-heron-3f9c`) and written to `/etc/meshweave/node_id`, then reused by
every later run, so re-running the installer updates that node rather than
adding a second one. Pass `--node-id my-gpu-box` only if you want to choose the
name yourself.

Then verify:

```bash
journalctl -u meshweave-agent -f
```

## What it reports

Real telemetry only — read from NVML and /proc. Nothing is estimated.

| Category | Fields |
|---|---|
| GPU | name, UUID, VRAM total/free/used, temperature, power draw, utilisation |
| CPU | model, core count |
| RAM | total, available |
| Disk | total, available |
| Network | hostname, private IP, external IP, published TCP port mappings |
| Agent | version, collection timestamp, health |

If no NVIDIA driver is present the agent registers anyway and reports `gpu_error`.
It never invents a device.

## Running a model across several nodes

The installer also places the stage runtime at `/opt/meshweave/stage_runner.py`.
A node needs the agent from this version to take part in a deployment: re-run the
same command to update it in place.

To join a multi-node deployment, the instance must publish a TCP port other than
SSH. That port is how the neighbouring stage connects; a node that only exposes
SSH cannot be part of a chain. See `docs/DEPLOYMENTS.md`.

## What it does not do

- **No shell endpoint.** The coordinator cannot run a command on your machine.
- **No file transfer.** No upload, no download.
- **No inbound port.** The agent dials out only; nothing dials it.
- **No firewall or SSH changes.** The installer touches neither.
- **No kernel or driver installation.**

The control API is a fixed set of routes. Anything outside that set does not exist,
which is enforced by the absence of the route rather than by a permission check.

## Uninstall

```bash
curl -fsSL <this-script> | sudo bash -s -- --uninstall
```

Removes the service, the agent and the credentials, including the generated
node name. A later install gets a new identity rather than reusing the old one.

## Security note

Running this makes the machine visible to your coordinator and lets it assign
model-execution work. The agent runs as a hardened systemd service
(`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`). Treat the coordinator as
trusted — it can see your hardware inventory and send work.

The registration token is stored at `/etc/meshweave/agent.json` with mode 0600 and
is never logged.