"""WireGuard configuration generation.

Pure functions: given a set of nodes, their private addresses, and their public
endpoints, produce the ``wg0.conf`` for each. Nothing here touches a host, so the
whole thing is unit-testable without SSH.

Design notes:

  * ``AllowedIPs`` is restricted to the mesh subnet. A WireGuard peer should only
    route mesh traffic, never become a default gateway for the host.
  * ``PersistentKeepalive`` is set on peers behind NAT. Without it, an idle tunnel
    loses its NAT mapping and the next transfer pays a full re-handshake.
  * MTU is set to 1280 because a lower MTU that works is far better than a higher
    one that silently black-holes large packets over a path with reduced MTU.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from meshweave.network.overlay.addresses import Allocation

DEFAULT_MTU = 1280
DEFAULT_LISTEN_PORT = 51820
KEEPALIVE_S = 25


@dataclass(frozen=True)
class PeerSpec:
    """One remote peer as far as this host is concerned."""

    node_id: str
    public_key: str
    endpoint: str | None  # "host:port", or None when the peer is behind NAT
    allowed_ips: str


@dataclass(frozen=True)
class InterfaceSpec:
    """This host's own WireGuard interface."""

    node_id: str
    private_key: str
    address: str
    listen_port: int = DEFAULT_LISTEN_PORT
    mtu: int = DEFAULT_MTU


def render_config(
    interface: InterfaceSpec,
    peers: list[PeerSpec],
    *,
    mtu: int = DEFAULT_MTU,
    keepalive_s: int = KEEPALIVE_S,
) -> str:
    """Render a complete ``wg0.conf`` for one host.

    ``peers`` is every *other* node in the mesh. The generated file is suitable
    for ``wg-quick up`` once placed at ``/etc/wireguard/wg0.conf``.
    """
    lines = [
        "[Interface]",
        f"PrivateKey = {interface.private_key}",
        f"Address = {interface.address}/24",
        f"ListenPort = {interface.listen_port}",
        f"MTU = {mtu}",
        "",
    ]

    for peer in peers:
        lines.append("[Peer]")
        lines.append(f"PublicKey = {peer.public_key}")
        if peer.endpoint:
            lines.append(f"Endpoint = {peer.endpoint}")
        lines.append(f"AllowedIPs = {peer.allowed_ips}")
        lines.append(f"PersistentKeepalive = {keepalive_s}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _is_subnet(child: object, parent: object) -> bool:
    """``subnet_of`` with the types narrowed to the same address family."""
    import ipaddress as _ip

    if not isinstance(child, _ip.IPv4Network) or not isinstance(parent, _ip.IPv4Network):
        if not isinstance(child, _ip.IPv6Network) or not isinstance(parent, _ip.IPv6Network):
            return False
        return child.subnet_of(parent)
    return child.subnet_of(parent)


def mesh_allowed_ips(base: str = "10.10.0.0/24") -> str:
    """The address range that should be routed through the mesh."""
    return base


def endpoint_for(host: str, port: int) -> str:
    """Build a WireGuard ``Endpoint`` value.

    The port must be an externally mapped UDP port on the host's public IP. On
    Vast that mapping only exists if it was requested at rent time, which is why
    this value is a parameter rather than an assumption.
    """
    return f"{host}:{port}"


def validate_peer_endpoints(peers: list[PeerSpec], mesh_cidr: str) -> list[str]:
    """Check that each peer's allowed IPs fall inside the mesh network.

    Returns a list of problems. An empty list means the configuration is
    internally consistent.
    """
    problems: list[str] = []
    network = ipaddress.ip_network(mesh_cidr, strict=False)

    for peer in peers:
        try:
            peer_net = ipaddress.ip_network(peer.allowed_ips, strict=False)
        except ValueError:
            problems.append(f"{peer.node_id}: invalid AllowedIPs {peer.allowed_ips!r}")
            continue
        # Compare only within the same address family; mixing IPv4 and IPv6
        # raises rather than returning False.
        if peer_net.version != network.version:
            problems.append(
                f"{peer.node_id}: AllowedIPs {peer.allowed_ips} is a different "
                f"address family from mesh {mesh_cidr}"
            )
        elif not _is_subnet(peer_net, network):
            problems.append(
                f"{peer.node_id}: AllowedIPs {peer.allowed_ips} is outside mesh {mesh_cidr}"
            )
        if peer.endpoint and ":" not in peer.endpoint:
            problems.append(f"{peer.node_id}: endpoint {peer.endpoint!r} is not host:port")

    return problems


def build_configs(
    allocations: list[Allocation],
    public_keys: dict[str, str],
    private_keys: dict[str, str],
    endpoints: dict[str, str | None],
    *,
    mesh_cidr: str = "10.10.0.0/24",
    mtu: int = DEFAULT_MTU,
    listen_port: int = DEFAULT_LISTEN_PORT,
) -> dict[str, str]:
    """Build one config per node, keyed by node id.

    ``endpoints`` maps a node id to ``"host:port"`` or ``None`` when that node has
    no reachable UDP endpoint. A ``None`` endpoint omits the ``Endpoint`` line, so
    the peer is passive and waits for the other side to initiate.
    """
    configs: dict[str, str] = {}
    allowed = mesh_allowed_ips(mesh_cidr)

    # Validate every node's keys up front so a missing peer key is reported as a
    # clear message rather than a KeyError from deep inside the peer list.
    for allocation in allocations:
        node_id = allocation.node_id
        for source, name in ((public_keys, "public"), (private_keys, "private")):
            if node_id not in source:
                raise ValueError(
                    f"missing {name} key for {node_id}; every node in the mesh "
                    "needs a keypair before configs can be generated"
                )

    for allocation in allocations:
        node_id = allocation.node_id
        if node_id not in public_keys or node_id not in private_keys:
            raise ValueError(
                f"missing keys for {node_id}; every node needs a keypair before "
                "configs can be generated"
            )

        peers = [
            PeerSpec(
                node_id=other.node_id,
                public_key=public_keys[other.node_id],
                endpoint=endpoints.get(other.node_id),
                allowed_ips=allowed,
            )
            for other in allocations
            if other.node_id != node_id
        ]

        interface = InterfaceSpec(
            node_id=node_id,
            private_key=private_keys[node_id],
            address=allocation.address,
            listen_port=listen_port,
            mtu=mtu,
        )
        configs[node_id] = render_config(interface, peers, mtu=mtu)

    return configs
