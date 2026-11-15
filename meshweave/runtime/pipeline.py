"""Pipeline execution runtime.

Runs one model split across nodes as an ordered chain of stages. Each stage:

  * receives an activation (or the token ids, for stage 0),
  * runs its own layer range,
  * sends the resulting activation to the next stage.

The final stage samples a token and streams it back along the chain.

Why this shape: it is the minimum that proves the MESHWEAVE hypothesis. There is
no tensor parallelism, no expert routing, no microbatching scheduler. Those are
experiments, not architecture.

Instrumentation is not optional. Every hop records bytes, latency and stage, so
an experiment can state what the network actually cost rather than asserting it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from meshweave.communication.wire import HttpEndpoint
from meshweave.core import (
    Environment,
    ExecutionRecord,
    StageTransfer,
)
from meshweave.errors import ExecutionError, NodeUnavailableError
from meshweave.models.base import Partition
from meshweave.runtime.transport import (
    Activation,
    StageConnection,
    TransferStats,
    measure_round_trip,
)


@dataclass
class StageRuntime:
    """One stage of a pipeline, as seen by the driver."""

    index: int
    node_id: str
    endpoint: HttpEndpoint
    layer_start: int
    layer_end: int
    is_first: bool
    is_last: bool
    connection: StageConnection | None = None
    failure: str | None = None

    def describe(self) -> str:
        return (
            f"stage {self.index} on {self.node_id} "
            f"(layers {self.layer_start}-{self.layer_end})"
        )


@dataclass
class PipelinePlan:
    """A resolved pipeline: ordered stages plus the shared context."""

    stages: list[StageRuntime]
    model: str
    workload_id: str
    hidden_size: int
    precision_bytes: float = 2.0
    partition: Partition | None = None
    environment: Environment = Environment.C_INTERNET
    per_hop_rtt_ms: float | None = None
    tokens: list[str] = field(default_factory=list)
    transfers: list[StageTransfer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PipelineRuntime:
    """Drives a distributed forward pass and a streaming decode loop."""

    def __init__(self, plan: PipelinePlan) -> None:
        self.plan = plan
        self._wires()

    def _wires(self) -> None:
        """Establish the persistent connection between adjacent stages."""
        for i, stage in enumerate(self.plan.stages[:-1]):
            nxt = self.plan.stages[i + 1]
            stage.connection = StageConnection(
                endpoint=nxt.endpoint,
                node_id=nxt.node_id,
                stage_index=i,
                workload_id=self.plan.workload_id,
            )

    # ------------------------------------------------------------- preflight

    async def preflight(self) -> dict[str, Any]:
        """Probe every hop and record the real latency.

        Returns the measured per-hop latency so the caller can compare it with
        the planner's estimate. A plan whose estimate disagrees with measurement
        is a finding, not a failure.
        """
        probe_bytes = 8 * 1024  # a 7B-class single-token activation
        results: list[dict[str, Any]] = []
        for stage in self.plan.stages:
            if stage.connection is None:
                continue
            try:
                measured = await measure_round_trip(
                    stage.connection.endpoint, b"\x00" * probe_bytes
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                stage.failure = str(exc)
                results.append({"stage": stage.index, "error": str(exc)})
                continue
            results.append({"stage": stage.index, **measured})

        latencies = [float(r["p50_ms"]) for r in results if r.get("p50_ms") is not None]
        if latencies:
            self.plan.per_hop_rtt_ms = sum(latencies) / len(latencies)
        return {
            "hops": results,
            "measured_per_hop_ms": self.plan.per_hop_rtt_ms,
            "boundaries": max(0, len(self.plan.stages) - 1),
        }

    # ----------------------------------------------------------- execution

    async def run(
        self, token_ids: list[int], max_new_tokens: int
    ) -> AsyncIterator[str]:
        """Execute the pipeline and yield generated tokens as they arrive.

        Raises ``ExecutionError`` when a stage fails. Partial output is still
        yielded before the failure, so a caller can report what was produced.
        """
        if not self.plan.stages:
            raise ExecutionError("pipeline has no stages")

        failed = [s for s in self.plan.stages if s.failure]
        if failed:
            raise ExecutionError(
                "preflight failed for " + ", ".join(s.describe() for s in failed)
            )

        self.plan.stages[0]
        last = self.plan.stages[-1]

        activation = Activation(
            dtype="int64",
            shape=(1, len(token_ids)),
            data=_encode_ints(token_ids),
        )

        # ---- prefill: push the prompt down the chain, once -------------
        for stage in self.plan.stages:
            if stage.connection is None:
                continue
            nxt = self.plan.stages[stage.index + 1]
            stats = await stage.connection.send(activation, nxt.index)
            self._record(stats, forward=True)

        # ---- decode: one token at a time -------------------------------
        generated: list[int] = []
        for step in range(max_new_tokens):
            try:
                token = await self._sample_next_token(last, generated)
            except (ExecutionError, NodeUnavailableError) as exc:
                raise ExecutionError(
                    f"stage {last.index} ({last.node_id}) failed at token {step}: {exc}"
                ) from exc

            if token is None:
                break

            generated.append(token)
            yield _decode_token(token)

            # Feed the new token back to stage 0 and run the chain again.
            activation = Activation(
                dtype="int64", shape=(1, 1), data=_encode_ints([token])
            )
            cursor: Activation = activation
            for stage in self.plan.stages:
                if stage.connection is None:
                    continue
                nxt = self.plan.stages[stage.index + 1]
                stats = await stage.connection.send(cursor, nxt.index)
                self._record(stats, forward=True)

    async def _sample_next_token(
        self, last: StageRuntime, generated: list[int]
    ) -> int | None:
        import httpx

        try:
            async with last.endpoint.client() as client:
                resp = await client.post(
                    "/agent/sample",
                    json={
                        "generated": generated,
                        "max_new_tokens": 1,
                        "workload_id": self.plan.workload_id,
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPError as exc:
            raise ExecutionError(f"sampling failed on {last.node_id}: {exc}") from exc

        token = body.get("token_id")
        if token is None:
            return None
        if body.get("finished"):
            return int(token)
        return int(token)

    def _record(self, stats: TransferStats, forward: bool) -> None:
        self.plan.transfers.append(
            stats.to_stage_transfer() if forward else stats.to_stage_transfer()
        )

    # -------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        transfers = self.plan.transfers
        if not transfers:
            return {"transfers": 0, "warnings": self.plan.warnings}

        forward = [t for t in transfers if t.direction == "forward"]
        latencies = sorted(t.send_ms for t in forward)
        total_bytes = sum(t.bytes_sent for t in forward)
        boundaries = max(1, len(self.plan.stages) - 1)

        return {
            "transfers": len(forward),
            "boundaries": boundaries,
            "total_bytes": total_bytes,
            "mean_bytes_per_transfer": total_bytes / len(forward) if forward else 0,
            "mean_send_ms": sum(latencies) / len(latencies) if latencies else None,
            "p50_send_ms": latencies[len(latencies) // 2] if latencies else None,
            "max_send_ms": latencies[-1] if latencies else None,
            "measured_per_hop_ms": self.plan.per_hop_rtt_ms,
            "network_bytes_per_token": total_bytes / max(1, len(latencies)),
            "warnings": self.plan.warnings,
        }

    def record_for(self, record: ExecutionRecord) -> ExecutionRecord:
        record.transfers = self.plan.transfers
        return record


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _encode_ints(values: list[int]) -> bytes:
    import struct

    return struct.pack(f">{len(values)}q", *values)


def _decode_ints(data: bytes) -> list[int]:
    import struct

    count = len(data) // 8
    return list(struct.unpack(f">{count}q", data[: count * 8]))


def _decode_token(token: int) -> str:
    """Render a token id as text without a tokenizer dependency.

    The runtime does not ship a tokenizer: token *ids* are the unit of the
    experiment (they are what the network sees). The caller supplies a tokenizer
    when it wants text, and the adapter's ``render`` hook does that.
    """
    return f"[{token}]"


def build_plan(
    stages: list[dict[str, Any]],
    model: str,
    workload_id: str,
    hidden_size: int,
    precision_bytes: float = 2.0,
) -> PipelineRuntime:
    """Assemble a PipelinePlan from stage dictionaries.

    ``stages`` is ordered: index, node_id, endpoint URL, layer_start, layer_end.
    """
    runtimes: list[StageRuntime] = []
    for i, spec in enumerate(stages):
        endpoint = HttpEndpoint(base_url=spec["endpoint"])
        runtimes.append(
            StageRuntime(
                index=i,
                node_id=spec["node_id"],
                endpoint=endpoint,
                layer_start=int(spec["layer_start"]),
                layer_end=int(spec["layer_end"]),
                is_first=(i == 0),
                is_last=(i == len(stages) - 1),
            )
        )
    return PipelineRuntime(
        PipelinePlan(
            stages=runtimes,
            model=model,
            workload_id=workload_id,
            hidden_size=hidden_size,
            precision_bytes=precision_bytes,
        )
    )
