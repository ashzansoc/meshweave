"""Shared fixtures for the MESHWEAVE test suite."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

# The registry persists to the repository's reports/ directory, which is also
# where the live coordinator keeps its state. Without this override a test run
# would rewrite that snapshot with fixtures, and node-a..node-d would reappear in
# the real dashboard after the next restart.
_SNAPSHOT_DIR = tempfile.mkdtemp(prefix="meshweave-test-registry-")
os.environ.setdefault(
    "MESHWEAVE_REGISTRY_SNAPSHOT", os.path.join(_SNAPSHOT_DIR, "registry_snapshot.json")
)

import pytest
from meshweave.agent.synthetic import synthetic_inventory
from meshweave.core import Heartbeat, NodeRegistration
from meshweave.registry.store import ResourceRegistry


@pytest.fixture
def registry() -> Iterator[ResourceRegistry]:
    yield ResourceRegistry()


def register_node(
    registry: ResourceRegistry, node_id: str, gpu: str, cost: float | None = None
) -> None:
    inventory = synthetic_inventory(gpu)
    inventory.gpus[0].uuid = node_id
    registry.register(
        NodeRegistration(
            node_id=node_id, hostname=node_id, agent_version="test", provider="vast"
        )
    )
    registry.record_heartbeat(Heartbeat(node_id=node_id, inventory=inventory))
    if cost is not None:
        registry.set_hourly_cost(node_id, cost)


@pytest.fixture
def four_nodes(registry: ResourceRegistry) -> ResourceRegistry:
    register_node(registry, "node-a", "RTX 4090", 0.35)
    register_node(registry, "node-b", "RTX 4090", 0.29)
    register_node(registry, "node-c", "RTX 3090", 0.22)
    register_node(registry, "node-d", "RTX 4080", 0.19)
    return registry
