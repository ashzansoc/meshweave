"""Communication layer: wire framing, TLS transport, node identity."""

from __future__ import annotations

from meshweave.communication.auth import (
    WorkloadToken,
    fingerprint_cert,
    issue_workload_token,
    load_or_create_secret,
)
from meshweave.communication.wire import (
    AgentClient,
    HttpEndpoint,
    encode_frame,
    read_frame,
    server_ssl_context,
    write_frame,
)

__all__ = [
    "AgentClient",
    "HttpEndpoint",
    "WorkloadToken",
    "encode_frame",
    "fingerprint_cert",
    "issue_workload_token",
    "load_or_create_secret",
    "read_frame",
    "server_ssl_context",
    "write_frame",
]
