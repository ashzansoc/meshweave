"""Live resource registry.

Holds every participating node, its real inventory, and its earned reliability.
Reliability starts unknown (None) and only becomes a number once there are
enough observations. A node that has never been seen is not "100% reliable".

Thread-safe: the coordinator runs this inside an async event loop but the
registry is also used from sync test code.
"""

from __future__ import annotations

import threading
from datetime import timedelta

from meshweave.config import CoordinatorSettings
from meshweave.core import (
    GpuDevice,
    Heartbeat,
    NodeRecord,
    NodeRegistration,
    NodeStatus,
    utcnow,
)

# A node must be observed at least this many times before it is assigned a
# reliability score. Below this, "unknown" is the honest answer.
MIN_OBSERVATIONS = 10


class ResourceRegistry:
    def __init__(self, settings: CoordinatorSettings | None = None) -> None:
        self.settings = settings or CoordinatorSettings()
        self._lock = threading.RLock()
        self._nodes: dict[str, NodeRecord] = {}
        self._last_seen: dict[str, float] = {}
        self.degrade_reasons: dict[str, str] = {}

    # ---------------------------------------------------------------- writes

    def register(self, registration: NodeRegistration) -> NodeRecord:
        with self._lock:
            record = self._nodes.get(registration.node_id)
            if record is None:
                record = NodeRecord(
                    node_id=registration.node_id, registration=registration
                )
                self._nodes[registration.node_id] = record
            else:
                record.registration = registration
            return record

    def record_heartbeat(self, heartbeat: Heartbeat) -> NodeRecord:
        with self._lock:
            record = self._nodes.get(heartbeat.node_id)
            if record is None:
                raise KeyError(f"heartbeat from unregistered node {heartbeat.node_id}")
            record.inventory = heartbeat.inventory
            record.status = NodeStatus.HEALTHY
            record.last_heartbeat_at = heartbeat.sent_at
            record.missed_heartbeats = 0
            record.observations += 1
            return record

    def mark_degraded(self, node_id: str, reason: str) -> None:
        self.degrade_reasons[node_id] = reason
        with self._lock:
            record = self._nodes.get(node_id)
            if record is not None:
                record.status = NodeStatus.DEGRADED

    def mark_unreachable(self, node_id: str) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is not None:
                record.status = NodeStatus.UNREACHABLE

    def set_hourly_cost(self, node_id: str, usd: float) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is not None:
                record.hourly_cost_usd = usd

    def retire(self, node_id: str) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is not None:
                record.status = NodeStatus.RETIRED

    # ----------------------------------------------------------------- reads

    def get(self, node_id: str) -> NodeRecord | None:
        with self._lock:
            return self._nodes.get(node_id)

    def all(self) -> list[NodeRecord]:
        with self._lock:
            return list(self._nodes.values())

    def healthy(self) -> list[NodeRecord]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status == NodeStatus.HEALTHY]

    def available(self) -> list[NodeRecord]:
        """Nodes that can accept work right now."""
        return [n for n in self.healthy() if n.gpus]

    def with_free_vram(self, required_bytes: int) -> list[NodeRecord]:
        return [n for n in self.available() if n.vram_free_bytes >= required_bytes]

    # ------------------------------------------------------------ aggregates

    def total_vram_bytes(self) -> int:
        return sum(n.vram_total_bytes for n in self.available())

    def total_free_vram_bytes(self) -> int:
        return sum(n.vram_free_bytes for n in self.available())

    def total_gpu_count(self) -> int:
        return sum(len(n.gpus) for n in self.available())

    def hourly_cost_usd(self) -> float | None:
        """Sum of known hourly costs. None if any node's cost is unknown."""
        costs = [n.hourly_cost_usd for n in self.available()]
        if any(c is None for c in costs):
            return None
        return float(sum(c for c in costs if c is not None))

    def all_gpus(self) -> list[tuple[str, GpuDevice]]:
        out: list[tuple[str, GpuDevice]] = []
        for node in self.available():
            for gpu in node.gpus:
                out.append((node.node_id, gpu))
        return out

    # -------------------------------------------------------------- reaping

    def reap(self, now: float | None = None) -> list[str]:
        """Mark nodes whose heartbeat has gone stale. Returns the node ids."""
        now = now if now is not None else utcnow().timestamp()
        changed: list[str] = []
        with self._lock:
            for node_id, record in self._nodes.items():
                last = record.last_heartbeat_at
                if last is None:
                    continue
                age = now - last.timestamp()
                if age > self.settings.heartbeat_grace_s and record.status == NodeStatus.HEALTHY:
                    record.status = NodeStatus.UNREACHABLE
                    record.missed_heartbeats += 1
                    changed.append(node_id)
        return changed

    def reliability(self, node_id: str) -> float | None:
        with self._lock:
            return self._nodes[node_id].reliability_score if node_id in self._nodes else None

    def recompute_reliability(self, node_id: str) -> float | None:
        """Beta-style score from heartbeat observations.

        ``None`` until at least ``MIN_OBSERVATIONS`` heartbeats exist, so a
        freshly-registered node is never reported as reliable.
        """
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None or record.observations < MIN_OBSERVATIONS:
                return None
            missed = record.missed_heartbeats
            total = record.observations + missed
            score = max(0.0, 1.0 - (missed / total))
            record.reliability_score = round(score, 4)
            record.uptime_fraction = score
            return record.reliability_score


def staleness(record: NodeRecord) -> timedelta | None:
    if record.last_heartbeat_at is None:
        return None
    return utcnow() - record.last_heartbeat_at
