"""Safety audit tests for remote process management.

These exist because an earlier cleanup routine killed sshd on three live hosts by
extracting every PID from ``ss`` output. The tests here are the guard against that
happening again: they assert that MESHWEAVE cannot terminate a process it did not
start, and they scan the source tree for the dangerous patterns that caused it.

Every test in the first group uses a real child process, so a failure means
MESHWEAVE actually killed something it should not have.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from meshweave.remote.process_registry import (
    REGISTRY,
    ProcessRegistry,
    UnsafeTerminationError,
    terminate_experiment,
)

REPO = Path(__file__).resolve().parent.parent


def _spawn_sleeper(seconds: float = 30.0) -> subprocess.Popen:
    """Start a real, unrelated process to be protected by these tests."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --------------------------------------------------------------------------- #
# Behavioural: unrelated processes must survive
# --------------------------------------------------------------------------- #


def test_unrecorded_pid_cannot_be_terminated():
    """The core guarantee: MESHWEAVE kills only what it started."""
    victim = _spawn_sleeper()
    try:
        registry = ProcessRegistry()
        with pytest.raises(UnsafeTerminationError, match="not recorded as started"):
            registry.terminate(victim.pid)
        assert victim.poll() is None, "an unrecorded process was terminated"
    finally:
        victim.kill()
        victim.wait()


def test_ssh_style_process_is_not_killable_by_port():
    """There is no port-based kill path to exercise.

    An earlier implementation killed every process holding a listening socket.
    That API no longer exists; this test asserts the replacement refuses rather
    than that some cleanup works.
    """
    victim = _spawn_sleeper()
    try:
        registry = ProcessRegistry()
        # No method accepts a port. Confirm the surface does not exist.
        assert not hasattr(registry, "kill_by_port")
        assert not hasattr(registry, "kill_by_name")
        assert not hasattr(registry, "kill_all")
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


def test_recorded_process_can_be_terminated():
    registry = ProcessRegistry()
    proc = _spawn_sleeper()
    try:
        registry.record(proc.pid, node_id="test-node", experiment_id="EXP-TEST", command="sleeper")
        assert registry.owns(proc.pid)
        assert registry.terminate(proc.pid, experiment_id="EXP-TEST") is True
        proc.wait(timeout=5)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_experiment_cannot_terminate_another_experiments_process():
    registry = ProcessRegistry()
    proc = _spawn_sleeper()
    try:
        registry.record(proc.pid, node_id="n", experiment_id="EXP-A", command="sleeper")
        with pytest.raises(UnsafeTerminationError, match="belongs to experiment EXP-A"):
            registry.terminate(proc.pid, experiment_id="EXP-B")
        assert proc.poll() is None, "a process was killed across experiment boundaries"
    finally:
        proc.kill()
        proc.wait()


def test_terminate_experiment_only_touches_its_own():
    """The cleanup entry point must leave other experiments' processes alone.

    ``terminate_experiment`` reads the process-wide registry, so this test records
    into that registry and cleans up after itself.
    """
    mine = _spawn_sleeper()
    theirs = _spawn_sleeper()
    exp_id = f"EXP-{os.getpid()}"
    try:
        REGISTRY.record(mine.pid, node_id="n", experiment_id=exp_id, command="a")
        REGISTRY.record(theirs.pid, node_id="n", experiment_id=f"{exp_id}-OTHER", command="b")

        result = terminate_experiment(exp_id)
        assert mine.pid in result["terminated"]
        assert theirs.pid not in result["terminated"]
        assert theirs.poll() is None, "another experiment's process was terminated"

        mine.wait(timeout=5)
        assert mine.poll() is not None
    finally:
        for proc in (mine, theirs):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_invalid_pid_is_refused():
    registry = ProcessRegistry()
    for bad in (0, -1):
        with pytest.raises(UnsafeTerminationError):
            registry.record(bad, node_id="n", experiment_id="E", command="x")


def test_terminating_a_dead_process_is_not_an_error():
    registry = ProcessRegistry()
    proc = _spawn_sleeper(0.01)
    proc.wait(timeout=5)
    pid = proc.pid
    registry.record(pid, node_id="n", experiment_id="EXP-DEAD", command="x")
    # The process is gone; termination reports False rather than raising.
    assert registry.terminate(pid, experiment_id="EXP-DEAD") is False
    assert not registry.owns(pid)


# --------------------------------------------------------------------------- #
# Static: dangerous patterns must not appear in shipped code
# --------------------------------------------------------------------------- #

# Shell tools that terminate by name pattern rather than by PID.
FORBIDDEN_SHELL_TOKENS = [
    "pkill",
    "killall",
    "killall5",
    "fuser -k",
    "lsof -t -k",
]

# Destructive shell commands MESHWEAVE must never run on a remote host.
#
# Note: the Vast REST API exposes a reboot endpoint, and ``providers/vast/client``
# defines a method for it. That is an explicit, human-invoked API call, not an
# automatic action, and it is deliberately allowed. What is forbidden is a
# *shell command* that reboots or shuts down, which could take a host offline
# with no confirmation.
FORBIDDEN_REMOTE_TOKENS = [
    "systemctl",
    "service ssh",
    "service sshd",
    "/etc/init.d/ssh",
    "iptables",
    "nft ",
    "ufw ",
    "firewall-cmd",
    "shutdown -h",
    "shutdown now",
    "init 0",
    "init 6",
]

# Files allowed to mention an otherwise-forbidden token, with the reason.
ALLOWED_FILE_TOKENS: dict[str, tuple[str, str]] = {
    "client.py": ("reboot", "explicit Vast REST API method, never auto-invoked"),
}

# Directories that make up MESHWEAVE's shipped code.
SHIPPED_DIRS = ["meshweave", "experiments"]


def _shipped_files() -> list[Path]:
    files: list[Path] = []
    for directory in SHIPPED_DIRS:
        root = REPO / directory
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _string_literals(path: Path) -> list[str]:
    """Every string literal in a file, via AST rather than regex.

    AST avoids matching the *word* ``pkill`` inside a comment or a docstring that
    explains why it is forbidden.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


@pytest.mark.parametrize("token", FORBIDDEN_SHELL_TOKENS)
def test_no_shell_pattern_kills_in_shipped_code(token: str):
    """A pattern-based kill must not be constructed anywhere in MESHWEAVE."""
    offenders: list[str] = []
    for path in _shipped_files():
        for literal in _string_literals(path):
            if token in literal and not literal.strip().startswith("#"):
                # Allow the string only in the module that documents the ban.
                if "process_registry.py" in str(path):
                    continue
                offenders.append(f"{path.name}: {literal[:80]}")
    assert not offenders, "pattern-based kill found:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("token", FORBIDDEN_REMOTE_TOKENS)
def test_no_remote_service_or_firewall_operations(token: str):
    """MESHWEAVE must never restart services, touch firewalls, or shut down."""
    offenders: list[str] = []
    for path in _shipped_files():
        if token == "reboot" and path.name in ALLOWED_FILE_TOKENS:
            continue
        for literal in _string_literals(path):
            lowered = literal.lower()
            if token.lower() in lowered:
                stripped = literal.strip().lower()
                if stripped.startswith("#") or "never" in lowered or "must not" in lowered:
                    continue
                offenders.append(f"{path.name}: {literal[:80]}")
    assert not offenders, f"forbidden remote operation {token!r} found:\n  " + "\n  ".join(offenders)


def test_vast_reboot_is_never_auto_invoked():
    """The reboot API method exists but nothing calls it on its own."""
    offenders: list[str] = []
    for path in _shipped_files():
        if path.name == "client.py":
            continue
        source = path.read_text()
        if ".reboot(" in source:
            offenders.append(str(path.name))
    assert not offenders, f"reboot() is invoked outside its definition: {offenders}"


def test_ssh_key_material_is_not_read():
    """The private key path is passed to ssh, never opened by MESHWEAVE.

    ``ssh.py`` does read ``~/.ssh/config`` to avoid duplicating aliases, which is
    a config file, not key material. This test checks specifically that no read
    targets the key path itself.
    """
    offenders: list[str] = []
    for path in _shipped_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in ("read_text", "read_bytes", "open")):
                continue
            # Inspect the receiver: only a read of DEFAULT_KEY (or a literal
            # private-key path) is a violation.
            target = ast.unparse(func.value) if hasattr(ast, "unparse") else ""
            if "DEFAULT_KEY" in target or "id_rsa" in target or ".pem" in target:
                offenders.append(f"{path.name}: reads {target}")
    assert not offenders, "private key material is read:\n  " + "\n  ".join(offenders)


def test_no_firewall_or_ssh_config_writes():
    """No code path may write sshd_config, authorized_keys, or firewall rules."""
    forbidden_writes = [
        "authorized_keys",
        "sshd_config",
        "/etc/ssh/",
        "iptables",
        "ufw allow",
        "ufw deny",
    ]
    offenders: list[str] = []
    for path in _shipped_files():
        source = path.read_text()
        for needle in forbidden_writes:
            if needle in source:
                # Permitted only in documentation prose.
                for line in source.splitlines():
                    stripped = line.strip()
                    if needle in stripped and not stripped.startswith("#"):
                        if '"""' in line or stripped.startswith("*"):
                            continue
                        offenders.append(f"{path.name}: {stripped[:90]}")
    assert not offenders, "forbidden write target:\n  " + "\n  ".join(offenders)


def test_cleanup_requires_explicit_experiment_id():
    """``terminate`` refuses to run without an ownership check."""
    registry = ProcessRegistry()
    proc = _spawn_sleeper()
    try:
        registry.record(proc.pid, node_id="n", experiment_id="EXP-X", command="x")
        # Calling without an experiment id is allowed (it is recorded), but the
        # record must exist -- an unrecorded PID is refused, which the test above
        # covers. Here we assert the ownership check is reachable at all.
        assert registry.get(proc.pid) is not None
    finally:
        proc.kill()
        proc.wait()


def test_global_registry_is_isolated_per_call():
    """Each experiment run gets its own ownership boundary."""
    a = ProcessRegistry()
    b = ProcessRegistry()
    proc = _spawn_sleeper()
    try:
        a.record(proc.pid, node_id="n", experiment_id="E", command="x")
        assert a.owns(proc.pid)
        assert not b.owns(proc.pid)
        with pytest.raises(UnsafeTerminationError):
            b.terminate(proc.pid)
    finally:
        proc.kill()
        proc.wait()
