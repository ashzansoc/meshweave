"""Virtual clusters: naming a group of physical nodes as one logical unit.

A cluster is a label plus a validation result. It does **not** create a bigger
GPU. Combined VRAM here is aggregate capacity across separate machines whose
memory is physically distributed, and every response says so, because that
distinction is the whole reason model execution has to be partitioned.

Validation is split deliberately:

  * **static** checks run at cluster creation -- is the node alive, does it have
    a GPU, is its agent new enough, can a peer dial it, has the link been
    measured
  * **memory** checks run at deploy time, when the model is known

A cluster cannot prove that an unnamed model fits. Claiming otherwise would make
the creation step look more capable than it is.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshweave.coordinator.deployments import agent_can_deploy, pick_port

GIB = 1024**3


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Cluster:
    cluster_id: str
    name: str
    node_ids: list[str]
    created_at: str = field(default_factory=utcnow)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_members(
    records: list[Any], topology: Any | None = None
) -> ValidationResult:
    """Static compatibility of a proposed cluster."""
    result = ValidationResult()

    if not records:
        result.errors.append("a cluster needs at least one node")
        return result
    if len({r.node_id for r in records}) != len(records):
        result.errors.append("the same node appears more than once")

    multi = len(records) > 1
    for record in records:
        inventory = record.inventory
        if inventory is None:
            result.errors.append(f"{record.node_id} has not reported its inventory")
            continue
        if not inventory.gpus:
            result.errors.append(f"{record.node_id} reports no GPU")
        if not agent_can_deploy(inventory.agent_version):
            result.errors.append(
                f"{record.node_id} runs agent {inventory.agent_version or 'unknown'}; "
                "0.4.0 or newer is required to take deployments"
            )
        if multi and pick_port(inventory) is None:
            result.errors.append(
                f"{record.node_id} publishes no TCP port other than SSH, so the other "
                "members cannot reach it"
            )
        if record.status.value in ("UNREACHABLE", "REGISTERING"):
            result.errors.append(f"{record.node_id} is {record.status.value.lower()}")

    capabilities = [
        tuple(g.compute_capability)
        for r in records
        if r.inventory and r.inventory.gpus
        for g in r.inventory.gpus
        if g.compute_capability
    ]
    if capabilities and min(capabilities) < (8, 0) and max(capabilities) >= (8, 0):
        result.warnings.append(
            "members differ in bfloat16 support; deployments will use float16, "
            "which costs some numerical headroom but keeps every member able to run"
        )

    if multi:
        measured = _measured_pairs(topology)
        for left in records:
            for right in records:
                if left.node_id == right.node_id:
                    continue
                if (left.node_id, right.node_id) not in measured:
                    result.warnings.append(
                        f"link {left.node_id}->{right.node_id} has not been measured; "
                        "distributed execution will run without a network suitabilit"
                        "y assessment"
                    )

    return result


def _measured_pairs(topology: Any | None) -> set[tuple[str, str]]:
    if topology is None:
        return set()
    pairs: set[tuple[str, str]] = set()
    for link in getattr(topology, "links", []) or []:
        if getattr(link, "connection_success", False) and getattr(link, "rtt_ms_min", None) is not None:
            pairs.add((link.source, link.target))
    return pairs


def cluster_status(cluster: Cluster, resolve: Any, availability) -> dict[str, Any]:
    """Capacity and health, computed from members right now."""
    members = [resolve(node_id) for node_id in cluster.node_ids]
    # Capacity counts only members that can currently serve. An unreachable
    # member's VRAM is not available for anything, and counting it would make a
    # degraded cluster look healthier than the sum of its parts.
    present = [
        m for m in members if m is not None and availability(m) != "OFFLINE"
    ]
    registered = [m for m in members if m is not None]

    gpus = sum(len(m.inventory.gpus) for m in present if m.inventory)
    vram = sum(
        g.vram_total_bytes
        for m in present
        if m.inventory
        for g in m.inventory.gpus
    )
    free = sum(
        g.vram_free_bytes for m in present if m.inventory for g in m.inventory.gpus
    )
    cores = sum(
        (m.inventory.host.cpu_cores or 0) for m in present if m.inventory and m.inventory.host
    )
    ram = sum(
        (m.inventory.host.ram_total_bytes or 0)
        for m in present
        if m.inventory and m.inventory.host
    )

    states = {m.node_id: availability(m) for m in registered}
    missing = [n for n in cluster.node_ids if n not in {m.node_id for m in registered}]

    if not present or all(state == "OFFLINE" for state in states.values()):
        health = "OFFLINE"
    elif missing or any(state in ("OFFLINE", "DEGRADED", "MAINTENANCE") for state in states.values()):
        health = "DEGRADED"
    else:
        health = "HEALTHY"

    return {
        "cluster_id": cluster.cluster_id,
        "name": cluster.name,
        "members": cluster.node_ids,
        "member_state": states,
        "missing": missing,
        "health": health,
        "capacity": {
            "gpus": gpus,
            "vram_total_bytes": vram,
            "vram_free_bytes": free,
            "vram_total_gib": round(vram / GIB, 2),
            "vram_free_gib": round(free / GIB, 2),
            "cpu_cores": cores,
            "ram_total_bytes": ram,
            "kind": "aggregate distributed capacity",
            "note": (
                "These GPUs are separate devices on separate machines. The sum is "
                "not one pool of addressable VRAM and cannot hold a model that no "
                "single member can hold; execution is partitioned across members."
            ),
        },
        "warnings": cluster.warnings,
        "created_at": cluster.created_at,
    }


class ClusterStore:
    """Clusters, persisted so a restart does not forget the logical grouping."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._items: dict[str, Cluster] = {}
        self._counter = 0
        if path is not None and path.exists():
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for item in raw.get("clusters", []):
            cluster = Cluster(**item)
            self._items[cluster.cluster_id] = cluster
            self._counter = max(self._counter, int(cluster.cluster_id.rsplit("-", 1)[-1]) or 0)

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {"saved_at": utcnow(), "clusters": [c.to_dict() for c in self._items.values()]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)

    def all(self) -> list[Cluster]:
        return list(self._items.values())

    def get(self, cluster_id: str) -> Cluster | None:
        return self._items.get(cluster_id)

    def by_name(self, name: str) -> Cluster | None:
        return next((c for c in self._items.values() if c.name == name), None)

    def create(self, name: str, node_ids: list[str], warnings: list[str]) -> Cluster:
        self._counter += 1
        cluster = Cluster(
            cluster_id=f"F16-VDC-{self._counter:03d}",
            name=name,
            node_ids=list(node_ids),
            warnings=list(warnings),
        )
        self._items[cluster.cluster_id] = cluster
        self._save()
        return cluster

    def update_warnings(self, cluster: Cluster, warnings: list[str]) -> None:
        cluster.warnings = list(warnings)
        self._save()

    def remove(self, cluster_id: str) -> bool:
        existed = self._items.pop(cluster_id, None) is not None
        if existed:
            self._save()
        return existed
