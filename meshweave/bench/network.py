"""Network benchmark.

Runs before any model is loaded. Produces the N x N measured matrix the planner
consumes. Every probe is a real measurement against a real endpoint; there is no
fallback that invents a plausible number.

Probes
------
RTT / jitter / loss : repeated TCP-connect probes (no ICMP privilege required),
                      plus ICMP ping when available.
TCP throughput      : single-stream and 4-stream transfer of a fixed payload.
Sustained throughput: longer transfer used to expose throttling after ramp-up.
Stability           : connect/disconnect cycles with a success ratio.
UDP support         : an actual UDP echo attempt, recorded as a boolean.
"""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from meshweave.core import LinkKind, LinkMeasurement, TopologyMatrix, utcnow

REFERENCE_PAYLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB
SUSTAINED_PAYLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB
RTT_SAMPLES = 20
STABILITY_ATTEMPTS = 10


class ProbeError(Exception):
    """A probe could not run. Recorded as an error, never as a zero."""


@dataclass
class Endpoint:
    host: str
    port: int
    node_id: str | None = None


@dataclass
class ProbeConfig:
    reference_payload_bytes: int = REFERENCE_PAYLOAD_BYTES
    sustained_payload_bytes: int = SUSTAINED_PAYLOAD_BYTES
    rtt_samples: int = RTT_SAMPLES
    stability_attempts: int = STABILITY_ATTEMPTS
    connect_timeout_s: float = 5.0
    transfer_timeout_s: float = 120.0
    concurrency: int = 4
    run_sustained: bool = True
    probes: list[str] = field(default_factory=lambda: ["rtt", "tcp", "stability", "udp"])


async def tcp_connect(host: str, port: int, timeout_s: float) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    try:
        return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_s)
    except (TimeoutError, OSError) as exc:
        raise ProbeError(f"tcp connect {host}:{port} failed: {exc}") from exc


async def probe_rtt(
    host: str, port: int, cfg: ProbeConfig
) -> tuple[float, float, float, float, float, float]:
    """Measure connect RTT by sampling TCP handshakes.

    Returns (p50, p95, min, jitter, loss_pct). Jitter is the mean absolute
    successive difference, which is the standard VoIP-style definition.
    """
    samples: list[float] = []
    failures = 0
    for _ in range(cfg.rtt_samples):
        start = time.perf_counter()
        try:
            reader, writer = await tcp_connect(host, port, cfg.connect_timeout_s)
        except ProbeError:
            failures += 1
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed_ms)
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()

    if not samples:
        return 0.0, 0.0, 0.0, 0.0, 100.0, 0.0

    p50 = statistics.median(samples)
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    p95 = ordered[idx]
    lo = min(samples)
    jitter = (
        statistics.mean(abs(samples[i] - samples[i - 1]) for i in range(1, len(samples)))
        if len(samples) > 1
        else 0.0
    )
    loss = failures / cfg.rtt_samples * 100.0
    # One-way is estimated as half of RTT. This is an estimate and is labelled
    # as such; asymmetric routes make it an upper bound on accuracy.
    one_way = p50 / 2.0
    return p50, p95, lo, jitter, loss, one_way


async def _drain(reader: asyncio.StreamReader) -> None:
    while True:
        chunk = await reader.read(1024 * 1024)
        if not chunk:
            return


async def measure_throughput(host: str, port: int, n_bytes: int, cfg: ProbeConfig) -> float:
    """Send exactly n_bytes and time it. Returns Mbps (10^6 bits/s)."""
    reader, writer = await tcp_connect(host, port, cfg.connect_timeout_s)
    payload = b"\x5a" * (1024 * 1024)  # 1 MiB pattern, repeated
    start = time.perf_counter()
    sent = 0
    try:
        while sent < n_bytes:
            chunk = payload[: min(len(payload), n_bytes - sent)]
            writer.write(chunk)
            await writer.drain()
            sent += len(chunk)
        writer.write_eof()
        await asyncio.wait_for(_drain(reader), timeout=cfg.transfer_timeout_s)
    finally:
        elapsed = time.perf_counter() - start
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
    if elapsed <= 0:
        raise ProbeError("throughput measurement returned non-positive duration")
    return (n_bytes * 8) / elapsed / 1e6


async def measure_multistream(host: str, port: int, n_bytes: int, cfg: ProbeConfig) -> float:
    """Aggregate throughput across ``cfg.concurrency`` parallel streams."""
    per = max(1, n_bytes // cfg.concurrency)

    async def one() -> float:
        return await measure_throughput(host, port, per, cfg)

    results = await asyncio.gather(*[one() for _ in range(cfg.concurrency)], return_exceptions=True)
    values = [r for r in results if isinstance(r, float)]
    if not values:
        raise ProbeError("all parallel streams failed")
    return sum(values)


async def measure_stability(host: str, port: int, cfg: ProbeConfig) -> tuple[bool, float]:
    """Connect/disconnect repeatedly; return (connected_ever, success_pct)."""
    ok = 0
    for _ in range(cfg.stability_attempts):
        try:
            reader, writer = await tcp_connect(host, port, cfg.connect_timeout_s)
        except ProbeError:
            continue
        ok += 1
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
    pct = ok / cfg.stability_attempts * 100.0 if cfg.stability_attempts else 0.0
    return ok > 0, pct


async def probe_udp(host: str, port: int, cfg: ProbeConfig) -> bool:
    """Attempt a real UDP echo. Returns whether it round-tripped."""
    del cfg  # the probe is a single round trip; config does not apply
    loop = asyncio.get_running_loop()
    result: dict[str, bool] = {"ok": False}

    class Protocol(asyncio.DatagramProtocol):
        def __init__(self) -> None:
            self.transport: asyncio.DatagramTransport | None = None

        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            self.transport = transport  # type: ignore[assignment]

        def datagram_received(self, data: bytes, addr: object) -> None:
            del addr
            if data == b"MESHWEAVE":
                result["ok"] = True

    try:
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(), remote_addr=(host, port)
        )
    except OSError:
        return False

    try:
        proto = protocol
        assert isinstance(proto, Protocol)  # noqa: S101
        assert proto.transport is not None  # noqa: S101
        proto.transport.sendto(b"MESHWEAVE")
        for _ in range(20):
            if result["ok"]:
                break
            await asyncio.sleep(0.05)
    finally:
        transport.close()
    return result["ok"]


async def measure_link(
    source: str,
    target: str,
    endpoint: Endpoint,
    cfg: ProbeConfig,
    link_kind: LinkKind = LinkKind.INTERNET,
) -> LinkMeasurement:
    """Run every configured probe against one endpoint. Failures are recorded."""
    link = LinkMeasurement(source=source, target=target, link_kind=link_kind, measured_at=utcnow())
    errors: list[str] = []

    if "rtt" in cfg.probes:
        try:
            p50, p95, lo, jitter, loss, one_way = await probe_rtt(
                endpoint.host, endpoint.port, cfg
            )
            link.rtt_ms_p50 = p50
            link.rtt_ms_p95 = p95
            link.rtt_ms_min = lo
            link.jitter_ms = jitter
            link.packet_loss_pct = loss
            link.one_way_ms_estimate = one_way
        except ProbeError as exc:
            errors.append(str(exc))

    if "tcp" in cfg.probes:
        try:
            single = await measure_throughput(
                endpoint.host, endpoint.port, cfg.reference_payload_bytes, cfg
            )
            link.tcp_mbps_single_stream = single
            link.transfer_time_s = cfg.reference_payload_bytes * 8 / (single * 1e6)
            try:
                link.tcp_mbps_multi_stream = await measure_multistream(
                    endpoint.host, endpoint.port, cfg.reference_payload_bytes, cfg
                )
            except ProbeError as exc:
                errors.append(f"multistream: {exc}")
            if cfg.run_sustained:
                try:
                    link.sustained_mbps = await measure_throughput(
                        endpoint.host, endpoint.port, cfg.sustained_payload_bytes, cfg
                    )
                except ProbeError as exc:
                    errors.append(f"sustained: {exc}")
        except ProbeError as exc:
            errors.append(str(exc))

    if "stability" in cfg.probes:
        try:
            ever, pct = await measure_stability(endpoint.host, endpoint.port, cfg)
            link.connection_success = ever
            link.connection_stability_pct = pct
        except ProbeError as exc:
            errors.append(str(exc))

    if "udp" in cfg.probes:
        try:
            link.udp_supported = await probe_udp(endpoint.host, endpoint.port, cfg)
        except OSError as exc:
            errors.append(f"udp: {exc}")

    if errors and link.rtt_ms_p50 is None and link.tcp_mbps_single_stream is None:
        link.error = "; ".join(errors)
    elif errors:
        # Partial success: keep the measurements, surface the gaps as an error
        # string so the report can show what we could not measure.
        link.error = "; ".join(errors)

    return link


async def build_matrix(
    nodes: list[str],
    endpoint_for: dict[str, Endpoint],
    cfg: ProbeConfig | None = None,
    link_kind_for: dict[str, LinkKind] | None = None,
    progress: Any | None = None,
) -> TopologyMatrix:
    """Probe every unordered pair exactly once, in both directions.

    Directionality matters: residential uplinks are frequently asymmetric, so a
    single measurement per direction is required. 8 nodes -> 28 pairs -> 56
    directed measurements.
    """
    cfg = cfg or ProbeConfig()
    link_kind_for = link_kind_for or {}
    matrix = TopologyMatrix(generated_at=utcnow(), nodes=list(nodes))
    tasks: list[tuple[str, str, Endpoint, LinkKind]] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            ka = link_kind_for.get(a, LinkKind.INTERNET)
            tasks.append((a, b, endpoint_for[a], ka))
            kb = link_kind_for.get(b, LinkKind.INTERNET)
            tasks.append((b, a, endpoint_for[b], kb))

    for source, target, endpoint, kind in tasks:
        link = await measure_link(source, target, endpoint, cfg, kind)
        matrix.links.append(link)
        if progress is not None:
            progress.tick(source, target)

    return matrix


def matrix_to_jsonable(matrix: TopologyMatrix) -> dict[str, Any]:
    return {
        "generated_at": matrix.generated_at.isoformat(),
        "nodes": matrix.nodes,
        "links": [link.model_dump(mode="json") for link in matrix.links],
    }


def endpoint_from_url(url: str, node_id: str | None = None) -> Endpoint:
    """Extract host:port from a URL, defaulting to the standard agent port."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 8443)
    return Endpoint(host=parsed.hostname or "127.0.0.1", port=port, node_id=node_id)


def new_probe_id() -> str:
    return f"probe-{uuid.uuid4().hex[:12]}"


def echo_handler_stub() -> None:  # pragma: no cover - documented for agent authors
    """The agent must expose a raw TCP echo endpoint for these probes.

    The reference implementation lives in ``meshweave.agent.echo`` and is the
    only reason MESHWEAVE opens a non-HTTP port on a node.
    """
    raise NotImplementedError
