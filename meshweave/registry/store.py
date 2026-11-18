"""Live resource registry.

Holds every participating node, its real inventory, and its earned reliability.
Reliability starts unknown (None) and only becomes a number once there are
enough observations. A node that has never been seen is not "100% reliable".

Thread-safe: the coordinator runs this inside an async event loop but the
registry is also used from sync test code.
"""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from pathlib import Path

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
        # Measured compute times, kept for the link assessment. Session-only: it
        # describes what this coordinator observed, and a restart begins again.
        self._compute_samples: list[float] = []
        self.degrade_reasons: dict[str, str] = {}
        # Snapshot loading is explicit, not automatic. A test that constructs a
        # registry must get an empty one; inheriting a previous run's cluster
        # would make results depend on what ran before.
        self._loaded_snapshot = False

    # ------------------------------------------------------------ persistence

    def _snapshot_path(self) -> Path:
        """Snapshot location, anchored to the repo rather than the cwd.

        The coordinator may be launched from any directory; a relative path would
        silently write the snapshot somewhere unexpected and the restart would find
        nothing.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        return repo_root / "reports" / "registry_snapshot.json"

    def load_snapshot(self) -> int:
        """Restore the registry from disk. Returns the number of nodes restored.

        Called explicitly by the coordinator at startup. Not called from the
        constructor: a test must start empty.
        """
        self._loaded_snapshot = True
        self._load_snapshot()
        return len(self._nodes)

    def _load_snapshot(self) -> None:
        path = self._snapshot_path()
        if not path.exists():
            return
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        for raw in data.get("nodes", []):
            try:
                record = NodeRecord.model_validate(raw)
            except Exception:  # noqa: BLE001 - a corrupt entry must not block startup
                continue
            # A restored node has not been heard from in this process. Until it
            # sends a heartbeat it is UNREACHABLE, whatever the snapshot said.
            # Trusting the snapshot's status would present a dead cluster as live.
            record.status = NodeStatus.UNREACHABLE
            record.missed_heartbeats += 1
            self._nodes[record.node_id] = record

    def _save_unlocked(self) -> None:
        payload = {
            "saved_at": utcnow().isoformat(),
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
        }
        path = self._snapshot_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)
        except OSError:
            # A failed snapshot must never break a heartbeat.
            pass

    def save_snapshot(self) -> None:
        """Persist the registry so a restart does not forget the cluster."""
        with self._lock:
            payload = {
                "saved_at": utcnow().isoformat(),
                "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            }
        path = self._snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)

    # ---------------------------------------------------------------- writes

    @staticmethod
    def _is_synthetic(record: NodeRecord) -> bool:
        """A node is synthetic when its agent version or GPU UUID says so.

        Synthetic nodes are test fixtures. They must never be presented as real
        hardware, and they are excluded from cluster aggregates.
        """
        inv = record.inventory
        if inv is not None:
            if "synthetic" in (inv.agent_version or "").lower():
                return True
            for gpu in inv.gpus:
                if gpu.uuid.startswith("synthetic-"):
                    return True
        return "synthetic" in (record.registration.agent_version or "").lower()

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
            result = record.model_copy(deep=True)
            self._save_unlocked()
        return result

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

    def forget(self, node_id: str) -> bool:
        """Remove a node entirely.

        Used for nodes that were registered in error, or synthetic fixtures left
        in a snapshot. Removal is explicit; nothing is dropped automatically.
        """
        with self._lock:
            existed = self._nodes.pop(node_id, None) is not None
            self.degrade_reasons.pop(node_id, None)
            if existed:
                self._save_unlocked()
            return existed

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
        """Real nodes that can accept work right now.

        Synthetic nodes are excluded: a test fixture must never be counted as
        cluster capacity.
        """
        return [
            n for n in self.healthy() if n.gpus and not self._is_synthetic(n)
        ]

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

    def fastest_compute_ms_per_token(self) -> float | None:
        """The best per-token compute time any deployment has recorded.

        Used as the denominator when judging a link. Deliberately the *fastest*
        measurement: a link that cannot keep up with the best this hardware does
        is not going to keep up with a slower run either.
        """
        values: list[float] = []
        for record in getattr(self, "_compute_samples", []):
            values.append(float(record))
        return min(values) if values else None

    def note_compute_ms_per_token(self, value: float | None) -> None:
        """Record a measured compute time for later link assessment."""
        if value is None or value <= 0:
            return
        samples = getattr(self, "_compute_samples", [])
        samples.append(float(value))
        # Keep the fastest 50: older slower runs add nothing to the comparison.
        self._compute_samples = sorted(samples)[:50]

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
            if changed:
                # Without this the reaped state lived only in memory and a
                # restart quietly resurrected a dead node as HEALTHY.
                self._save_unlocked()
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
