"""Overlay address allocation with subnet-conflict detection.

Allocates private addresses from 10.10.0.0/24 and refuses to proceed when a
proposed subnet collides with an interface already present on a host. A conflict
would silently black-hole traffic, so it is detected before any interface is
created rather than discovered afterwards.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from dataclasses import dataclass

from meshweave.errors import MeshweaveError

DEFAULT_PREFIX = "10.10.0.0/24"


class SubnetConflictError(MeshweaveError):
    """A proposed overlay address overlaps an existing interface."""


@dataclass
class Interface:
    """One network interface as observed on a host."""

    name: str
    address: str
    prefix: int
    is_loopback: bool = False
    is_ours: bool = False


@dataclass
class Allocation:
    """The address plan for one host."""

    node_id: str
    address: str
    prefix: int
    is_hub: bool = False

    @property
    def cidr(self) -> str:
        return f"{self.address}/{self.prefix}"


def detect_conflicts(
    existing: list[Interface], proposed_cidr: str
) -> list[str]:
    """Return the names of interfaces that overlap ``proposed_cidr``.

    Only non-loopback interfaces count: the loopback range overlapping a private
    overlay is expected and harmless.
    """
    try:
        proposed = ipaddress.ip_network(proposed_cidr, strict=False)
    except ValueError as exc:
        raise SubnetConflictError(f"invalid proposed network {proposed_cidr!r}: {exc}") from exc

    conflicts: list[str] = []
    for iface in existing:
        if iface.is_loopback:
            continue
        try:
            other = ipaddress.ip_interface(f"{iface.address}/{iface.prefix}")
        except ValueError:
            continue
        if other.ip in proposed:
            conflicts.append(iface.name)
    return conflicts


def allocate(
    node_ids: list[str],
    *,
    base: str = DEFAULT_PREFIX,
    existing_by_node: dict[str, list[Interface]] | None = None,
    is_hub: Callable[[str], bool] | None = None,
) -> list[Allocation]:
    """Allocate one address per node from ``base``.

    Raises ``SubnetConflictError`` if the base network overlaps any interface
    already present on any host. That check runs before allocation so a conflict
    is reported with the offending interface named.
    """
    existing_by_node = existing_by_node or {}
    hub = is_hub or (lambda _: False)

    for node_id, interfaces in existing_by_node.items():
        conflicts = detect_conflicts(interfaces, base)
        if conflicts:
            raise SubnetConflictError(
                f"proposed overlay {base} overlaps existing interface(s) "
                f"{', '.join(conflicts)} on {node_id}. Choose a different --base."
            )

    network = ipaddress.ip_network(base, strict=True)
    # Reserve .1 for the hub so the numbering is predictable and stable.
    allocations: list[Allocation] = []
    for index, node_id in enumerate(node_ids, start=1):
        address = str(network.network_address + index)
        allocations.append(
            Allocation(
                node_id=node_id,
                address=address,
                prefix=network.prefixlen,
                is_hub=hub(node_id),
            )
        )
    return allocations


def parse_interfaces(ip_addr_output: str) -> list[Interface]:
    """Parse ``ip -o addr show`` output into interfaces.

    Only the fields MESHWEAVE needs are extracted; unparseable lines are skipped
    rather than aborting the whole inventory.
    """
    interfaces: list[Interface] = []
    for line in ip_addr_output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[1]
        cidr = parts[3]
        try:
            iface = ipaddress.ip_interface(cidr)
        except ValueError:
            continue
        interfaces.append(
            Interface(
                name=name,
                address=str(iface.ip),
                prefix=iface.network.prefixlen,
                is_loopback=iface.ip.is_loopback,
            )
        )
    return interfaces
