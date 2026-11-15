"""Wire protocol: length-prefixed JSON over TLS.

Deliberately minimal. There is no RPC introspection, no shell endpoint, and no
generic "run this on the host" primitive. The coordinator may send a
``WorkloadAssignment`` or a ``CommandRequest`` from a closed enum; nothing else.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from dataclasses import dataclass
from typing import Any

import httpx

from meshweave.errors import MeshweaveError, NodeUnavailableError

MAX_FRAME_BYTES = 64 * 1024 * 1024  # 64 MiB hard ceiling on any single frame


class ProtocolError(MeshweaveError):
    pass


def encode_frame(obj: Any) -> bytes:
    """Serialise a JSON-able object into a length-prefixed frame."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large: {len(payload)} > {MAX_FRAME_BYTES}")
    return len(payload).to_bytes(8, "big") + payload


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    header = await reader.readexactly(8)
    length = int.from_bytes(header, "big")
    if length <= 0 or length > MAX_FRAME_BYTES:
        raise ProtocolError(f"invalid frame length {length}")
    body = await reader.readexactly(length)
    parsed: dict[str, Any] = json.loads(body.decode("utf-8"))
    return parsed


async def write_frame(writer: asyncio.StreamWriter, obj: Any) -> None:
    writer.write(encode_frame(obj))
    await writer.drain()


@dataclass
class HttpEndpoint:
    """Where to reach a peer agent's HTTP API."""

    base_url: str
    ca_bundle: str | None = None
    timeout_s: float = 30.0

    def client(self) -> httpx.AsyncClient:
        verify: str | bool = self.ca_bundle if self.ca_bundle else True
        return httpx.AsyncClient(base_url=self.base_url, verify=verify, timeout=self.timeout_s)


def server_ssl_context(certfile: str, keyfile: str, ca: str | None = None) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if ca:
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=ca)
    return ctx


class AgentClient:
    """Coordinator-side client for one agent's control API.

    Every call maps to a closed set of agent endpoints. There is no passthrough.
    """

    def __init__(self, node_id: str, endpoint: HttpEndpoint) -> None:
        self.node_id = node_id
        self.endpoint = endpoint

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self.endpoint.client() as client:
                resp = await client.post(path, json=body)
                resp.raise_for_status()
                return dict(resp.json())
        except httpx.HTTPError as exc:
            raise NodeUnavailableError(f"{self.node_id}: {path} failed: {exc}") from exc

    async def ping(self) -> dict[str, Any]:
        return await self._post("/agent/ping", {})

    async def assign(self, assignment: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/agent/assign", assignment)

    async def cancel(self, workload_id: str) -> dict[str, Any]:
        return await self._post("/agent/cancel", {"workload_id": workload_id})

    async def memory(self) -> dict[str, Any]:
        return await self._post("/agent/memory", {})

    async def drain(self) -> dict[str, Any]:
        return await self._post("/agent/drain", {})
