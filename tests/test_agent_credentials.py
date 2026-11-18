"""The agent must never send its node token where an enrollment token is expected.

This produced a 401 on every restart once the agent began storing a node token:
the bearer and the enrollment secret are different credentials, and swapping them
looks correct in code while failing on the wire.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

AGENT_PY = Path(__file__).resolve().parent.parent / "deploy" / "agent.py"


def _load_agent():
    spec = importlib.util.spec_from_file_location("mw_agent_cred", AGENT_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Handler(BaseHTTPRequestHandler):
    seen: dict[str, str] = {}

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        _Handler.seen = {
            "path": self.path,
            "enrollment": self.headers.get("x-meshweave-enrollment", ""),
            "authorization": self.headers.get("authorization", ""),
            "body": json.loads(body or b"{}").get("node_id", ""),
        }
        payload = json.dumps({"node_token": "node-token-abc"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence the test server
        pass


def _serve() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_the_bearer_is_the_node_token_and_the_enrollment_header_is_not() -> None:
    agent = _load_agent()
    server, base = _serve()
    try:
        client = agent.Coordinator(
            base,
            token="node-token-from-a-previous-run",
            node_id="alpha",
            enroll=True,
            enrollment_token="shared-bootstrap-secret",
        )
        client.register({"network": {"hostname": "alpha"}})
    finally:
        server.shutdown()

    assert _Handler.seen["enrollment"] == "shared-bootstrap-secret"
    assert _Handler.seen["authorization"] == "Bearer node-token-from-a-previous-run"

    # Registration handed back a node token; it becomes the bearer for everything
    # that follows, and the enrollment secret stays where it belongs.
    assert client.token == "node-token-abc"
    assert client.enrollment_token == "shared-bootstrap-secret"


def test_an_agent_without_enrollment_sends_no_header() -> None:
    agent = _load_agent()
    server, base = _serve()
    try:
        client = agent.Coordinator(base, token="node-token", node_id="beta", enroll=False)
        client.register({"network": {"hostname": "beta"}})
    finally:
        server.shutdown()

    assert _Handler.seen["enrollment"] == ""


def test_a_stored_node_token_does_not_become_the_enrollment_secret() -> None:
    """The exact failure: config holds both, and register must still succeed."""
    agent = _load_agent()
    server, base = _serve()
    try:
        client = agent.Coordinator(
            base,
            token="stored-node-token",
            node_id="alpha",
            enroll=True,
            enrollment_token="enroll-demo-token",
        )
        assert client.enrollment_token == "enroll-demo-token"
        response = client.register({"network": {"hostname": "alpha"}})
    finally:
        server.shutdown()

    assert response["node_token"] == "node-token-abc"
    assert _Handler.seen["enrollment"] == "enroll-demo-token"