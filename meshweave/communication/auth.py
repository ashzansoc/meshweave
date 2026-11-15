"""Node identity: certificates and workload tokens.

Every agent has a certificate. Every workload gets a short-lived token scoped to
that workload and that node. Tokens are never reused across workloads.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from meshweave.errors import AuthError

TOKEN_TTL_S = 900  # 15 minutes


@dataclass(frozen=True)
class WorkloadToken:
    workload_id: str
    node_id: str
    issued_at: int
    expires_at: int
    nonce: str
    signature: str

    def encode(self) -> str:
        body = {
            "workload_id": self.workload_id,
            "node_id": self.node_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return f"{base64.urlsafe_b64encode(raw).decode()}.{self.signature}"

    @classmethod
    def decode(cls, token: str, secret: bytes) -> WorkloadToken:
        try:
            body_b64, signature = token.split(".", 1)
            body = json.loads(base64.urlsafe_b64decode(body_b64.encode()))
        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise AuthError("malformed workload token") from exc

        expected = _sign(body_b64, secret)
        if not hmac.compare_digest(expected, signature):
            raise AuthError("workload token signature mismatch")
        if int(body.get("expires_at", 0)) < int(time.time()):
            raise AuthError("workload token expired")

        return cls(
            workload_id=str(body["workload_id"]),
            node_id=str(body["node_id"]),
            issued_at=int(body["issued_at"]),
            expires_at=int(body["expires_at"]),
            nonce=str(body["nonce"]),
            signature=signature,
        )


def _sign(body_b64: str, secret: bytes) -> str:
    digest = hmac.new(secret, body_b64.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def issue_workload_token(workload_id: str, node_id: str, secret: bytes) -> WorkloadToken:
    now = int(time.time())
    nonce = secrets.token_hex(16)
    body = {
        "workload_id": workload_id,
        "node_id": node_id,
        "issued_at": now,
        "expires_at": now + TOKEN_TTL_S,
        "nonce": nonce,
    }
    body_b64 = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return WorkloadToken(
        workload_id=workload_id,
        node_id=node_id,
        issued_at=now,
        expires_at=now + TOKEN_TTL_S,
        nonce=nonce,
        signature=_sign(body_b64, secret),
    )


def load_or_create_secret(path: str) -> bytes:
    """Load the coordinator's token-signing secret, creating it if absent."""
    if os.path.exists(path):  # noqa: PTH110
        with open(path, "rb") as fh:  # noqa: PTH123
            return fh.read().strip()
    secret = secrets.token_bytes(32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:  # noqa: PTH123
        fh.write(secret)
    os.chmod(path, 0o600)  # noqa: PTH101
    return secret


def fingerprint_cert(cert_pem: str) -> str:
    """SHA-256 of a PEM certificate, for pinning in the registry."""
    from cryptography import x509

    cert = x509.load_pem_x509_certificate(cert_pem.encode())
    der = cert.public_bytes(__import__("cryptography").hazmat.primitives.serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()
