"""Register the three real Vast hosts with a running coordinator.

Reads the verified hardware inventory produced by EXP-NET-001 and posts a
registration, one heartbeat, and the known hourly rate for each host, so the
coordinator's registry reflects what the machines actually report.

This is a convenience for development and demos. In a real deployment the agent
runs on each host and registers itself.

Usage:
    uv run python scripts/register_vast_nodes.py
    uv run python scripts/register_vast_nodes.py --coordinator http://127.0.0.1:8700
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import meshweave  # noqa: E402
from meshweave.core import (  # noqa: E402
    GpuDevice,
    GpuInventory,
    Heartbeat,
    HostResources,
    NetworkInfo,
    NodeRegistration,
)
from meshweave.remote import HOSTS, Host  # noqa: E402

GIB = 1024**3
INVENTORY = Path("experiments/network/raw/host_inventory.json")

# Hourly rates for these instance types, from the Vast listing. Recorded as data
# rather than guessed; a node with no entry stays unknown.
HOURLY_RATES = {
    "meshweave-host-a": 0.19,
    "meshweave-host-b": 0.29,
    "meshweave-host-c": 0.22,
}


def parse_vram(text: object) -> int:
    """Parse an nvidia-smi memory string such as ``6144 MiB`` into bytes."""
    if text is None:
        return 0
    raw = str(text).strip().upper()
    for suffix, multiplier in (
        ("GIB", 1024**3),
        ("MIB", 1024**2),
        ("KIB", 1024),
        ("B", 1),
    ):
        if raw.endswith(suffix):
            try:
                return int(float(raw[: -len(suffix)]) * multiplier)
            except ValueError:
                return 0
    try:
        # A bare number from nvidia-smi is MiB.
        return int(float(raw) * 1024**2)
    except ValueError:
        return 0


def build_inventory(record: dict, host: Host) -> GpuInventory | None:
    """Build a GpuInventory from one host's recorded discovery output."""
    gpus = record.get("verified_gpus") or []
    if not gpus:
        return None

    devices = []
    for index, gpu in enumerate(gpus):
        total = parse_vram(gpu.get("memory_total"))
        free = parse_vram(gpu.get("memory_free"))
        devices.append(
            GpuDevice(
                index=index,
                uuid=str(gpu["uuid"]),
                name=str(gpu["name"]),
                driver_version=gpu.get("driver_version"),
                vram_total_bytes=total,
                vram_free_bytes=free,
                vram_used_bytes=max(0, total - free),
            )
        )

    os_release = (
        (record.get("commands", {}).get("os_release", {}).get("stdout") or "")[:80]
    )
    return GpuInventory(
        gpus=devices,
        host=HostResources(os_name=os_release.strip() or None),
        network=NetworkInfo(
            hostname=host.node_id,
            external_ip=host.host,
            nat=True,
            mapped_ports={},
        ),
        agent_version=f"phase2a+{meshweave.__version__}",
    )


def load_inventory() -> dict:
    """Read the inventory file before the event loop starts.

    Path reads are blocking; doing them inside a coroutine would stall the loop.
    """
    if not INVENTORY.exists():
        raise FileNotFoundError(INVENTORY)
    return {h["node_id"]: h for h in json.loads(INVENTORY.read_text())}


async def main(coordinator: str) -> int:
    try:
        records = load_inventory()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"missing or unreadable {INVENTORY}: {exc}", file=sys.stderr)
        return 2

    registered = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        for host in HOSTS:
            record = records.get(host.node_id)
            if not record or not record.get("auth_ok"):
                print(f"  {host.node_id}: no verified inventory, skipped")
                continue

            inventory = build_inventory(record, host)
            if inventory is None:
                print(f"  {host.node_id}: no GPUs reported, skipped")
                continue

            registration = NodeRegistration(
                node_id=host.node_id,
                hostname=host.node_id,
                agent_version=inventory.agent_version,
                provider="vast",
            )
            await client.post(
                f"{coordinator}/v1/nodes/register",
                json=registration.model_dump(mode="json"),
            )
            await client.post(
                f"{coordinator}/v1/nodes/{host.node_id}/heartbeat",
                json=Heartbeat(
                    node_id=host.node_id, inventory=inventory
                ).model_dump(mode="json"),
            )

            rate = HOURLY_RATES.get(host.node_id)
            if rate is not None:
                await client.post(
                    f"{coordinator}/v1/nodes/{host.node_id}/cost",
                    json={"hourly_cost_usd": rate},
                )

            registered += 1
            total_gib = sum(d.vram_total_bytes for d in inventory.gpus) / GIB
            rate_text = f", ${rate}/hr" if rate is not None else ""
            print(f"  {host.node_id}: {len(inventory.gpus)} GPU, {total_gib:.1f} GiB{rate_text}")

    print(f"\nregistered {registered}/{len(HOSTS)} hosts")
    return 0 if registered else 1


def cli() -> int:
    parser = argparse.ArgumentParser(description="Register the Vast hosts")
    parser.add_argument("--coordinator", default="http://127.0.0.1:8700")
    args = parser.parse_args()
    return asyncio.run(main(args.coordinator))


if __name__ == "__main__":
    raise SystemExit(cli())
