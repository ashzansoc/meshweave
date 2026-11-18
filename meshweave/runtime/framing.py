"""Length-prefixed framing for stage-to-stage traffic.

A frame is a 4-byte big-endian JSON length, the JSON header, then the raw
payload. The header carries a CRC32 of the payload so a truncated or corrupted
transfer fails loudly instead of being decoded as activations.

This is deliberately small and separate from ``experiments/model_execution/
pipeline_node.py``: that file is a working, measured Phi-MoE path and is left
alone rather than refactored under an experiment that is still proving itself.
"""

from __future__ import annotations

import json
import socket
import struct
import zlib

HEADER_LEN = struct.Struct(">I")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 1 << 30  # 1 GiB -- larger than any activation we send


class FramingError(RuntimeError):
    """A peer sent something that is not a well-formed frame."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise FramingError(f"peer closed after {n - remaining} of {n} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> int:
    """Send one frame. Returns the number of payload bytes on the wire."""
    header = dict(header)
    header["payload_bytes"] = len(payload)
    header["crc32"] = zlib.crc32(payload) & 0xFFFFFFFF
    blob = json.dumps(header).encode()
    sock.sendall(HEADER_LEN.pack(len(blob)) + blob)
    if payload:
        sock.sendall(payload)
    return len(payload)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    """Receive one frame, verifying its length bounds and checksum."""
    (header_len,) = HEADER_LEN.unpack(_recv_exact(sock, 4))
    if header_len > MAX_HEADER_BYTES:
        raise FramingError(f"header length {header_len} exceeds {MAX_HEADER_BYTES}")
    header = json.loads(_recv_exact(sock, header_len))
    payload_len = int(header.get("payload_bytes", 0))
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise FramingError(f"payload length {payload_len} out of range")
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    expected = int(header.get("crc32", 0))
    if (zlib.crc32(payload) & 0xFFFFFFFF) != expected:
        raise FramingError("payload checksum mismatch")
    return header, payload
