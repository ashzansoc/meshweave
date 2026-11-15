"""Activation transport.

The one thing a distributed transformer needs from the network: move a hidden
state from stage N to stage N+1, and get the next token back.

Design notes that matter for the experiment:

* An activation is ``batch x seq_len x hidden``. During decode with a single
  request that is a *tiny* payload (7 KiB for a 7B model). Bandwidth is not the
  problem; the round trip is. This module therefore exposes both a *send* path
  and a *latency* measurement, so an experiment can compare the two rather than
  assume one dominates.

* Connections are persistent and pre-established. A fresh TLS handshake per
  token would add a full RTT to every token for no benefit.

* Every transfer records bytes and elapsed time. The runtime writes those into
  ``StageTransfer`` so the report can prove what crossed the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from meshweave.communication.wire import HttpEndpoint
from meshweave.core import StageTransfer
from meshweave.errors import TransferError

# Frame header: magic (4 bytes) + payload length (4 bytes) + dtype/ndim header.
_MAGIC = b"MWAX"
_HEADER = struct.Struct(">4sII")  # magic, payload_len, header_len


@dataclass
class Activation:
    """A tensor crossing a stage boundary.

    ``dtype`` and ``shape`` are recorded so the receiving side can validate the
    payload instead of trusting it.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def n_elements(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def describe(self) -> str:
        return f"{self.dtype}{list(self.shape)} = {self.nbytes} B"


def encode_activation(act: Activation) -> bytes:
    import json

    header = json.dumps({"dtype": act.dtype, "shape": list(act.shape)}).encode()
    return _HEADER.pack(_MAGIC, len(act.data), len(header)) + header + act.data


def decode_activation(frame: bytes) -> Activation:
    import json

    if len(frame) < _HEADER.size:
        raise TransferError("activation frame truncated")
    magic, payload_len, header_len = _HEADER.unpack(frame[: _HEADER.size])
    if magic != _MAGIC:
        raise TransferError("bad activation magic")
    header_bytes = frame[_HEADER.size : _HEADER.size + header_len]
    payload = frame[_HEADER.size + header_len :]
    if len(payload) != payload_len:
        raise TransferError(
            f"activation payload length mismatch: header says {payload_len}, got {len(payload)}"
        )
    meta = json.loads(header_bytes.decode())
    return Activation(dtype=str(meta["dtype"]), shape=tuple(meta["shape"]), data=payload)


@dataclass
class TransferStats:
    """What one activation transfer actually cost."""

    from_stage: int
    to_stage: int
    bytes_sent: int
    send_ms: float
    round_trip_ms: float | None = None
    bytes_per_second: float | None = None
    direction: str = "forward"

    def to_stage_transfer(self) -> StageTransfer:
        return StageTransfer(
            from_stage=self.from_stage,
            to_stage=self.to_stage,
            bytes_sent=self.bytes_sent,
            send_ms=self.send_ms,
            wire_rtt_ms=self.round_trip_ms,
            direction=self.direction,
        )


@dataclass
class StageConnection:
    """A persistent connection to the next pipeline stage."""

    endpoint: HttpEndpoint
    node_id: str
    stage_index: int
    workload_id: str = ""
    stats: list[TransferStats] = field(default_factory=list)

    async def send(self, act: Activation, to_stage: int) -> TransferStats:
        """Send one activation and wait for the stage to acknowledge receipt.

        The acknowledgement is what makes the latency measurement honest: a
        one-way send would not tell us when the bytes actually landed.
        """
        frame = encode_activation(act)
        started = time.perf_counter()

        import httpx

        try:
            async with self.endpoint.client() as client:
                resp = await client.post(
                    "/agent/activation",
                    content=frame,
                    headers={
                        "content-type": "application/octet-stream",
                        "x-meshweave-workload": self.workload_id,
                    },
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TransferError(
                f"activation to {self.node_id} (stage {to_stage}) failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats = TransferStats(
            from_stage=self.stage_index,
            to_stage=to_stage,
            bytes_sent=act.nbytes,
            send_ms=elapsed_ms,
            round_trip_ms=elapsed_ms,
            bytes_per_second=(act.nbytes / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else None,
        )
        self.stats.append(stats)
        return stats

    def summary(self) -> dict[str, Any]:
        if not self.stats:
            return {"transfers": 0}
        latencies = [s.send_ms for s in self.stats]
        latencies.sort()
        return {
            "transfers": len(self.stats),
            "mean_send_ms": sum(latencies) / len(latencies),
            "p50_send_ms": latencies[len(latencies) // 2],
            "max_send_ms": latencies[-1],
            "mean_bytes": sum(s.bytes_sent for s in self.stats) / len(self.stats),
        }


async def measure_round_trip(
    endpoint: HttpEndpoint, payload: bytes, samples: int = 5
) -> dict[str, Any]:
    """Measure the real per-hop latency to a stage.

    This is the number the whole pipeline-latency argument depends on, so it is
    measured rather than assumed. It includes TLS and HTTP overhead, which is
    the honest cost of the path MESHWEAVE actually uses.
    """
    import httpx

    samples_list: list[float] = []
    async with endpoint.client() as client:
        for _ in range(samples):
            start = time.perf_counter()
            try:
                resp = await client.post(
                    "/agent/ping", json={}, timeout=endpoint.timeout_s
                )
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            samples_list.append((time.perf_counter() - start) * 1000.0)

    if not samples_list:
        raise TransferError(f"round-trip probe to {endpoint.base_url} failed")

    samples_list.sort()
    return {
        "samples": samples_list,
        "p50_ms": samples_list[len(samples_list) // 2],
        "min_ms": samples_list[0],
        "max_ms": samples_list[-1],
        "payload_bytes": len(payload),
    }


def payload_for(spec_hidden: int, batch: int = 1, seq_len: int = 1, dtype_size: float = 2.0) -> int:
    """Bytes of one activation tensor -- the thing that actually crosses the wire."""
    return int(batch * seq_len * spec_hidden * dtype_size)


async def aclose(*writers: asyncio.StreamWriter) -> None:
    for writer in writers:
        with contextlib.suppress(Exception):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
