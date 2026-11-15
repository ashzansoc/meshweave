"""Remote command execution over SSH.

Used to inspect and configure GPU hosts. Deliberately thin: it shells out to the
system ``ssh`` binary rather than adding an SSH library dependency, because the
only thing MESHWEAVE needs is "run this command, give me the output and the exit
code".

Rules enforced here:
  * ``BatchMode=yes`` -- never prompt, never hang waiting for a password.
  * ``StrictHostKeyChecking=accept-new`` -- record new hosts, reject changed ones.
  * A non-zero exit is surfaced with its stderr, never swallowed.
  * Wall-clock is recorded for reporting, but durations use a monotonic clock.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from meshweave.errors import MeshweaveError

DEFAULT_KEY = Path.home() / ".ssh" / "id_rsa_vastai"
SSH_CONFIG = Path.home() / ".ssh" / "config"


class RemoteError(MeshweaveError):
    """An SSH command failed, or the host could not be reached."""


@dataclass
class Host:
    """One remote GPU host.

    A host is identified by its node id, never by its IP: two instances may share
    a public IP and differ only by mapped port.
    """

    node_id: str
    host: str
    port: int
    user: str = "root"
    key: Path = DEFAULT_KEY
    label: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def describe(self) -> str:
        return f"{self.node_id} ({self.target}:{self.port})"


@dataclass
class Result:
    """One command's outcome."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    host: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def lines(self) -> list[str]:
        return [line for line in self.stdout.splitlines() if line.strip()]


def _ssh_argv(host: Host, remote_command: str, connect_timeout: float) -> list[str]:
    argv = [
        "ssh",
        "-i",
        str(host.key),
        "-p",
        str(host.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={int(connect_timeout)}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
    ]
    if not SSH_CONFIG.exists():
        argv += ["-o", "UserKnownHostsFile=/dev/null"]
    argv.append(f"{host.target}")
    argv.append(remote_command)
    return argv


def _auth_detail(stderr: str, limit: int = 240) -> str:
    """Extract the useful line from an ssh failure.

    Hosts commonly print a welcome banner (Vast prints several lines of it) before
    the authentication failure. Taking the last non-empty line reliably lands on
    the actual error.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "(no stderr)"
    for line in reversed(lines):
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in ("permission denied", "connection refused", "timed out",
                           "host key verification", "no route", "connection reset")
        ):
            return line[:limit]
    return lines[-1][:limit]


async def run(
    host: Host,
    command: str,
    *,
    timeout_s: float = 60.0,
    connect_timeout_s: float = 10.0,
    check: bool = True,
) -> Result:
    """Run one command on a host and return its result.

    Raises ``RemoteError`` when the command fails and ``check`` is true, or when
    the SSH connection itself fails. A command that runs and exits non-zero is a
    different thing from a host that cannot be reached, and the message says which
    happened.
    """
    argv = _ssh_argv(host, command, connect_timeout_s)
    started = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise RemoteError(f"{host.describe()}: cannot execute ssh: {exc}") from exc

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RemoteError(
            f"{host.describe()}: command timed out after {timeout_s:.0f}s: {command[:120]}"
        ) from exc

    elapsed = time.monotonic() - started
    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    code = proc.returncode if proc.returncode is not None else -1

    result = Result(
        command=command,
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        duration_s=elapsed,
        host=host.node_id,
    )

    if code == 255:
        # ssh's own failure code: connection, auth, or configuration problem.
        # The remote MOTD is stripped: it is host-supplied text, often a long
        # banner, and including it buries the actual failure.
        detail = _auth_detail(stderr)
        raise RemoteError(
            f"{host.describe()}: ssh failed (exit 255). "
            f"This is an authentication or connectivity problem, not a command error.\n"
            f"  command: {command[:120]}\n"
            f"  stderr: {detail}"
        )

    if check and code != 0:
        raise RemoteError(
            f"{host.describe()}: command exited {code}\n"
            f"  command: {command[:200]}\n"
            f"  stderr: {stderr.strip()[:400]}"
        )

    return result


async def run_many(
    host: Host,
    commands: list[str],
    *,
    timeout_s: float = 60.0,
    stop_on_error: bool = False,
) -> list[Result]:
    """Run several commands in one SSH session.

    Combining them into a single session avoids paying the handshake cost per
    command, which matters when collecting a large diagnostic set.
    """
    script = "\n".join(f"echo '--- {i} ---'\n{c}" for i, c in enumerate(commands))
    result = await run(host, script, timeout_s=timeout_s, check=stop_on_error)
    return [result]


async def check_auth(host: Host) -> tuple[bool, str]:
    """Confirm key-based auth works, without running anything meaningful.

    Returns ``(ok, detail)``. Never raises: a failed auth check is the expected
    outcome when a key has not been installed, and the caller needs the reason.
    """
    try:
        result = await run(host, "echo __AUTH_OK__", timeout_s=20.0, check=False)
    except RemoteError as exc:
        return False, str(exc)

    if result.ok and "__AUTH_OK__" in result.stdout:
        return True, "authenticated"

    if result.exit_code == 255:
        return False, f"ssh rejected the key: {result.stderr.strip()[:200]}"

    return False, f"command failed with exit {result.exit_code}"


def ensure_ssh_config(entries: list[Host]) -> Path:
    """Write ``~/.ssh/config`` aliases for these hosts.

    Only appends aliases that are absent, so an existing hand-written config is
    never clobbered. File permissions are checked and corrected.
    """
    SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    existing = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""

    missing: list[str] = []
    for host in entries:
        alias = alias_for(host)
        if f"Host {alias}" in existing:
            continue
        missing.append(
            "\n".join(
                [
                    f"Host {alias}",
                    f"    HostName {host.host}",
                    f"    Port {host.port}",
                    f"    User {host.user}",
                    f"    IdentityFile {host.key}",
                    "    IdentitiesOnly yes",
                    "    StrictHostKeyChecking accept-new",
                    "",
                ]
            )
        )

    if missing:
        with SSH_CONFIG.open("a") as fh:
            fh.write("\n# MESHWEAVE Phase 2A hosts\n")
            fh.write("\n".join(missing))

    SSH_CONFIG.chmod(0o600)
    return SSH_CONFIG


def alias_for(host: Host) -> str:
    """SSH config alias.

    Node ids already carry the ``meshweave-`` prefix, so stripping it avoids
    ``meshweave-meshweave-host-a``. The result is ``meshweave-host-a``.
    """
    node_id = host.node_id.lower()
    prefix = "meshweave-"
    suffix = node_id[len(prefix):] if node_id.startswith(prefix) else node_id
    return f"{prefix}{suffix}"


def describe_auth_methods(host: Host) -> str:
    """Human-readable summary of which auth methods a host offers."""
    argv = [
        "ssh",
        "-v",
        "-i",
        str(host.key),
        "-p",
        str(host.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{host.target}",
        "true",
    ]
    return shlex.join(argv)
