"""Shared fixtures for the MESHWEAVE test suite."""

from __future__ import annotations

from collections.abc import Iterator

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
