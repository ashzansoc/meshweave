"""TCP echo and probe, used to measure one link between two nodes.

This measures what a pipeline will actually experience: a TCP connection between
those two machines, over whatever NAT and provider routing sits in between. It is
not ICMP, and it is not a bandwidth estimate from a provider's marketing page.

What it can and cannot report:

  * round-trip time, from repeated small round trips on one connection
  * jitter, as the mean absolute difference between consecutive round trips
  * upload and download throughput, from a single large transfer each way
  * failures, counted when a round trip times out or the connection drops

Packet loss is **not** measurable here. TCP retransmits, so a lost packet shows up
as latency rather than loss, and ICMP is not available inside these containers.
The failure count is reported instead, and called a failure count.

Wire format is deliberately trivial: a 4-byte big-endian length followed by that
many bytes, echoed back unchanged.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field

LENGTH = struct.Struct(">I")
MAX_FRAME = 16 * 1024 * 1024


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #


class EchoServer:
    """A bounded TCP echo listener. Started by the agent, stopped by its ttl."""

    def __init__(self, host: str, port: int, *, ttl_s: float = 300.0) -> None:
        self.host = host
        self.port = port
        self.ttl_s = ttl_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.accepted = 0
        self.frames = 0

    def start(self) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(4)
        server.settimeout(1.0)
        self._sock = server
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.port

    def _run(self) -> None:
        assert self._sock is not None
        deadline = time.monotonic() + self.ttl_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.accepted += 1
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        self.stop()

    def _serve_conn(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            while not self._stop.is_set():
                try:
                    (length,) = LENGTH.unpack(_recv_exact(conn, 4))
                    if length > MAX_FRAME:
                        return
                    payload = _recv_exact(conn, length)
                    conn.sendall(LENGTH.pack(length) + payload)
                    self.frames += 1
                except (ConnectionError, OSError, struct.error):
                    return

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    ok: bool = False
    target: str = ""
    connect_ms: float | None = None
    rtt_count: int = 0
    rtt_min_ms: float | None = None
    rtt_p50_ms: float | None = None
    rtt_p95_ms: float | None = None
    rtt_max_ms: float | None = None
    rtt_mean_ms: float | None = None
    jitter_ms: float | None = None
    failures: int = 0
    upload_mbps: float | None = None
    download_mbps: float | None = None
    bulk_bytes: int = 0
    error: str | None = None
    samples_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        # Keep the series small: the summary is the evidence, the raw samples
        # are enough to show the distribution is not a single lucky packet.
        payload["samples_ms"] = [round(v, 3) for v in self.samples_ms[:40]]
        return payload


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def probe(
    host: str,
    port: int,
    *,
    rtt_count: int = 20,
    rtt_bytes: int = 64,
    bulk_bytes: int = 4 * 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
    timeout_s: float = 20.0,
) -> ProbeResult:
    """Measure one direction of one link: this host -> host:port."""
    result = ProbeResult(target=f"{host}:{port}", bulk_bytes=bulk_bytes)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(timeout_s)
    try:
        connect_start = time.perf_counter()
        sock.connect((host, port))
        connect_ms = (time.perf_counter() - connect_start) * 1000.0
    except OSError as exc:
        result.error = f"connect failed: {exc}"
        sock.close()
        return result

    try:
        payload = b"\x5a" * rtt_bytes
        for _ in range(rtt_count):
            start = time.perf_counter()
            try:
                sock.sendall(LENGTH.pack(len(payload)) + payload)
                (length,) = LENGTH.unpack(_recv_exact(sock, 4))
                _recv_exact(sock, length)
            except (OSError, ConnectionError):
                result.failures += 1
                continue
            result.samples_ms.append((time.perf_counter() - start) * 1000.0)

        if result.samples_ms:
            samples = result.samples_ms
            result.rtt_count = len(samples)
            result.rtt_min_ms = round(min(samples), 3)
            result.rtt_p50_ms = round(_percentile(samples, 0.50), 3)
            result.rtt_p95_ms = round(_percentile(samples, 0.95), 3)
            result.rtt_max_ms = round(max(samples), 3)
            result.rtt_mean_ms = round(sum(samples) / len(samples), 3)
            if len(samples) > 1:
                diffs = [abs(b - a) for a, b in zip(samples, samples[1:], strict=False)]
                result.jitter_ms = round(sum(diffs) / len(diffs), 3)

        if bulk_bytes > 0:
            block = b"\xa5" * min(chunk_bytes, bulk_bytes)
            sent = 0
            upload_start = time.perf_counter()
            while sent < bulk_bytes:
                piece = block[: min(len(block), bulk_bytes - sent)]
                try:
                    sock.sendall(LENGTH.pack(len(piece)) + piece)
                except OSError as exc:
                    result.error = f"upload failed after {sent} bytes: {exc}"
                    break
                sent += len(piece)
            upload_s = time.perf_counter() - upload_start

            received = 0
            download_start = time.perf_counter()
            while received < sent:
                try:
                    (length,) = LENGTH.unpack(_recv_exact(sock, 4))
                    _recv_exact(sock, length)
                except (OSError, ConnectionError) as exc:
                    result.error = f"download failed after {received} bytes: {exc}"
                    break
                received += length
            download_s = time.perf_counter() - download_start

            if sent and upload_s > 0:
                result.upload_mbps = round((sent * 8) / upload_s / 1e6, 2)
            if received and download_s > 0:
                result.download_mbps = round((received * 8) / download_s / 1e6, 2)

        result.ok = result.rtt_count > 0 and result.error is None
        result.connect_ms = round(connect_ms, 3)
    finally:
        try:
            sock.close()
        except OSError:
            pass

    return result
