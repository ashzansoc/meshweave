"""The three Vast.ai hosts for Phase 2A.

Node identity is the node id, never the IP: HOST-A and HOST-C share a public IP
and differ only by mapped port. Treating the IP as the identity would silently
merge two machines into one.

Hardware fields here are *claims from the provider listing*. They are verified
against ``nvidia-smi`` at runtime and the report records which is which. Anything
unverified is marked as such.
"""

from __future__ import annotations

from pathlib import Path

from meshweave.remote.ssh import DEFAULT_KEY, Host

HOST_A = Host(
    node_id="meshweave-host-a",
    host="1.208.108.242",
    port=51400,
    user="root",
    key=DEFAULT_KEY,
    label="HOST-A",
)

HOST_B = Host(
    node_id="meshweave-host-b",
    host="83.63.1.187",
    port=24014,
    user="root",
    key=DEFAULT_KEY,
    label="HOST-B",
)

HOST_C = Host(
    node_id="meshweave-host-c",
    host="1.208.108.242",
    port=46079,
    user="root",
    key=DEFAULT_KEY,
    label="HOST-C",
)

HOSTS: list[Host] = [HOST_A, HOST_B, HOST_C]

# What the Vast listing claims. Verified against nvidia-smi before being trusted.
CLAIMED_SPECS: dict[str, dict[str, object]] = {
    "meshweave-host-a": {
        "gpu_name": "NVIDIA GeForce GTX 1660 SUPER",
        "vram_gib": 6.0,
    },
    "meshweave-host-b": {
        "gpu_name": "NVIDIA RTX 4000",
        "vram_gib": 8.0,
    },
    "meshweave-host-c": {
        "gpu_name": "NVIDIA GeForce RTX 2070 SUPER",
        "vram_gib": 8.0,
    },
}

# Directory for all Phase 2A raw evidence.
NETWORK_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "network"
RAW_DIR = NETWORK_DIR / "raw"


def by_id(node_id: str) -> Host:
    for host in HOSTS:
        if host.node_id == node_id:
            return host
    raise KeyError(f"unknown host {node_id!r}; known: {[h.node_id for h in HOSTS]}")


def shared_ip_groups() -> list[list[str]]:
    """Group node ids that share a public IP.

    Returns groups with more than one member. Sharing an IP is a signal worth
    investigating; it is not proof of co-location.
    """
    by_ip: dict[str, list[str]] = {}
    for host in HOSTS:
        by_ip.setdefault(host.host, []).append(host.node_id)
    return [ids for ids in by_ip.values() if len(ids) > 1]
