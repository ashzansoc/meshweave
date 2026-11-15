"""Failure detection and recovery.

Independent GPU hosts fail. The coordinator must know which execution stages are
affected and recover according to the strategy, rather than restarting everything.

Recovery semantics per strategy:

  SINGLE_GPU     : the request failed. Nothing to recover; retry elsewhere.
  REPLICATED     : drop the failed replica. In-flight requests on it fail; new
                   requests route to survivors. No cluster-wide impact.
  PIPELINE       : the chain is broken. A request mid-flight cannot continue past
                   the missing stage, so it fails. Recovery means rebuilding the
                   stage on another node and replaying from the last checkpoint.
  TENSOR         : the collective is broken; the whole group fails.

For autoregressive inference there is a hard limit worth stating plainly: if a
stage holding layers 14-28 dies, the KV cache for those layers lived in that
process. It cannot be reconstructed from the surviving stages, because the
hidden states entering that stage are not recoverable from later stages. The
request must be restarted from the prompt. MESHWEAVE records this rather than
pretending exact continuation is possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from meshweave.core import ExecutionRecord, NodeStatus, ParallelStrategy


@dataclass
class FailureEvent:
    node_id: str
    workload_id: str
    detected_at: float
    kind: str  # heartbeat_timeout | connection_refused | transfer_error | oom | crash
    detail: str
    affected_stages: list[int] = field(default_factory=list)


@dataclass
class RecoveryOutcome:
    workload_id: str
    strategy: ParallelStrategy
    recovered: bool
    restarted_from_prompt: bool
    requests_lost: int
    requests_survived: int
    action_taken: str
    explanation: str
    kv_cache_recoverable: bool


class FailureDetector:
    """Tracks node health transitions and decides what to do about them."""

    def __init__(self, heartbeat_grace_s: float = 30.0) -> None:
        self.grace_s = heartbeat_grace_s
        self._last_seen: dict[str, float] = {}
        self._failures: list[FailureEvent] = []

    def observe(self, node_id: str, now: float | None = None) -> None:
        self._last_seen[node_id] = now if now is not None else time.time()

    def is_alive(self, node_id: str, now: float | None = None) -> bool:
        last = self._last_seen.get(node_id)
        if last is None:
            return False
        current = now if now is not None else time.time()
        return (current - last) <= self.grace_s

    def record_failure(self, event: FailureEvent) -> None:
        self._failures.append(event)

    def failures(self) -> list[FailureEvent]:
        return list(self._failures)

    def silent_nodes(self, known: list[str], now: float | None = None) -> list[str]:
        return [n for n in known if not self.is_alive(n, now)]


def plan_recovery(
    event: FailureEvent,
    record: ExecutionRecord,
    stage_for_node: dict[str, int],
    surviving_nodes: list[str],
) -> RecoveryOutcome:
    """Decide and describe the recovery for one failed workload.

    This function does not restart anything itself. It returns the decision so
    the coordinator can execute it and the experiment can report it.
    """
    strategy = record.strategy
    affected = stage_for_node.get(event.node_id)
    if affected is not None and affected not in event.affected_stages:
        event.affected_stages.append(affected)

    if strategy == ParallelStrategy.SINGLE_GPU:
        return RecoveryOutcome(
            workload_id=record.workload_id,
            strategy=strategy,
            recovered=False,
            restarted_from_prompt=False,
            requests_lost=1,
            requests_survived=0,
            action_taken="fail request; no retry on the same node",
            explanation="the only replica is gone; the request cannot continue",
            kv_cache_recoverable=False,
        )

    if strategy == ParallelStrategy.REPLICATED:
        return RecoveryOutcome(
            workload_id=record.workload_id,
            strategy=strategy,
            recovered=True,
            restarted_from_prompt=False,
            requests_lost=1,
            requests_survived=0,
            action_taken="drop replica; route new requests to survivors",
            explanation=(
                f"{len(surviving_nodes)} replica(s) remain and can serve new requests; "
                "the in-flight request on the failed replica is lost"
            ),
            kv_cache_recoverable=False,
        )

    if strategy == ParallelStrategy.PIPELINE:
        # Could the missing stage be rebuilt and the request replayed?
        rebuildable = len(surviving_nodes) > 0 and affected is not None
        return RecoveryOutcome(
            workload_id=record.workload_id,
            strategy=strategy,
            recovered=rebuildable,
            restarted_from_prompt=True,
            requests_lost=1,
            requests_survived=0,
            action_taken=(
                "rebuild the stage on a surviving node and replay the request from the prompt"
                if rebuildable
                else "abort the request; not enough surviving nodes to rebuild the chain"
            ),
            explanation=(
                f"stage {affected} held layers whose KV cache is not reconstructable "
                "from the surviving stages; exact continuation is impossible, so the "
                "request restarts from the prompt"
            ),
            kv_cache_recoverable=False,
        )

    return RecoveryOutcome(
        workload_id=record.workload_id,
        strategy=strategy,
        recovered=False,
        restarted_from_prompt=False,
        requests_lost=1,
        requests_survived=0,
        action_taken="abort; collective group is broken",
        explanation=f"{strategy.value} requires every member; losing one fails the group",
        kv_cache_recoverable=False,
    )


def detect_oom(detail: str) -> bool:
    lowered = detail.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered


def node_status_after_failure(node_id: str, detector: FailureDetector) -> NodeStatus:
    if detector.is_alive(node_id):
        return NodeStatus.HEALTHY
    return NodeStatus.UNREACHABLE


def recovery_summary(outcomes: list[RecoveryOutcome]) -> dict[str, Any]:
    return {
        "total": len(outcomes),
        "recovered": sum(1 for o in outcomes if o.recovered),
        "unrecovered": sum(1 for o in outcomes if not o.recovered),
        "requests_lost": sum(o.requests_lost for o in outcomes),
        "restarted_from_prompt": sum(1 for o in outcomes if o.restarted_from_prompt),
        "kv_cache_recoverable_any": any(o.kv_cache_recoverable for o in outcomes),
    }
