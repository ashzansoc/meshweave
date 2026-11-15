"""Local certificate authority for development and test clusters.

MESHWEAVE requires every node to have a certificate and every workload to carry
a short-lived token. In production you would use a real CA; for a research
cluster on rented hosts a local CA is the right size of solution.

The generated node certificate carries the node id in its CN, so the coordinator
can bind a TLS connection to a registered node without a separate lookup.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from meshweave.errors import MeshweaveError


def _key(path: Path) -> rsa.RSAPrivateKey:
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)  # type: ignore[return-value]
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return key


def ensure_ca(out_dir: Path) -> tuple[Path, Path]:
    """Create a CA if absent. Returns (cert_path, key_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / "ca.crt"
    key_path = out_dir / "ca.key"

    if cert_path.exists():
        return cert_path, key_path

    key = _key(key_path)
    subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "MESHWEAVE Local CA")]
    )
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def issue_node_certificate(
    out_dir: Path, node_id: str, days: int = 365
) -> dict[str, Any]:
    """Issue a node certificate signed by the local CA."""
    ca_cert_path, ca_key_path = ensure_ca(out_dir)
    ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
    ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)

    node_key_path = out_dir / f"{node_id}.key"
    node_cert_path = out_dir / f"{node_id}.crt"

    key = _key(node_key_path)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, f"meshweave-node:{node_id}"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MESHWEAVE"),
        ]
    )
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(node_id)]), critical=False)
        .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
    )
    node_cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    node_cert_path.chmod(0o644)

    return {
        "node_id": node_id,
        "certificate": str(node_cert_path),
        "key": str(node_key_path),
        "ca_certificate": str(ca_cert_path),
        "subject": f"meshweave-node:{node_id}",
        "expires_days": days,
    }


def load_node_cert(cert_path: Path) -> x509.Certificate:
    try:
        return x509.load_pem_x509_certificate(cert_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise MeshweaveError(f"cannot load certificate {cert_path}: {exc}") from exc


def common_name(cert: x509.Certificate) -> str | None:
    for attr in cert.subject:
        if attr.oid == NameOID.COMMON_NAME:
            return str(attr.value)
    return None
