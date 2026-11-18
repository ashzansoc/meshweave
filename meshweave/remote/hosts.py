"""Host definitions.

Hosts are **discovered, not hardcoded**. A MESHWEAVE cluster is whatever agents
have registered with the coordinator; there is no static inventory of machines.

The earlier version of this module hardcoded three Vast instances by IP and port.
That was wrong twice over: those instances no longer exist, and any hardcoded list
guarantees the code drifts from reality. Nodes are identified by the node id the
agent presents, never by IP — two containers can share a public IP and differ only
by mapped port.

For scripts that need to reach a specific host by name, resolve it from the
environment:

    MESHWEAVE_HOSTS='node-a=1.2.3.4:22 node-b=5.6.7.8:2222'

Nothing here is required for the coordinator, the agent or the dashboard. It exists
for operator convenience only.
"""

from __future__ import annotations

import os

from meshweave.remote.ssh import DEFAULT_KEY, Host

# Optional operator-supplied map of node id -> "host:port". Empty by default.
HOSTS_ENV = "MESHWEAVE_HOSTS"


def hosts_from_env() -> list[Host]:
    """Parse ``MESHWEAVE_HOSTS`` into Host objects.

    Format: ``node-a=1.2.3.4:22 node-b=5.6.7.8:2222``. Malformed entries are
    skipped rather than aborting, because a typo in an optional convenience
    variable should not break anything.
    """
    raw = os.environ.get(HOSTS_ENV, "").strip()
    if not raw:
        return []

    hosts: list[Host] = []
    for entry in raw.split():
        node_id, _, addr = entry.partition("=")
        if not node_id or not addr:
            continue
        host, _, port = addr.partition(":")
        if not host:
            continue
        try:
            port_num = int(port) if port else 22
        except ValueError:
            continue
        hosts.append(
            Host(node_id=node_id, host=host, port=port_num, user="root", key=DEFAULT_KEY)
        )
    return hosts


def by_id(node_id: str) -> Host:
    """Resolve one host by node id from the environment.

    Raises rather than guessing: an unknown node id is an operator error, and
    silently returning a wrong endpoint would be worse than failing.
    """
    for host in hosts_from_env():
        if host.node_id == node_id:
            return host
    raise KeyError(
        f"node {node_id!r} is not in {HOSTS_ENV}. Hosts are discovered from "
        "registered agents; set the variable to reach one by name from a script."
    )


# There is no built-in fleet. Scripts that imported these names get empty values,
# which is the honest answer: no hosts are known until agents register.
HOSTS: list[Host] = []
CLAIMED_SPECS: dict[str, dict[str, object]] = {}
