"""The installer is the only thing a new host runs, so it must stay honest.

Two failure modes are covered here, both of which have already happened once:

* ``deploy/install.sh`` carries its own copy of the agent in a heredoc, so the
  two can drift. A host would then run agent code nobody tests.
* The runtime fetch list contained a file that does not exist in the repo and
  every failed fetch was silenced. A node would report telemetry and then fail
  at stage start with a missing module.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "deploy" / "install.sh"
AGENT_PY = REPO / "deploy" / "agent.py"

HEREDOC = re.compile(r"<<'MESHWEAVE_AGENT_EOF'\n(.*?)\nMESHWEAVE_AGENT_EOF", re.S)


def _embedded_agent() -> str:
    match = HEREDOC.search(INSTALL_SH.read_text())
    assert match, "agent heredoc not found in install.sh"
    return match.group(1) + "\n"


def test_installer_embeds_the_agent_verbatim() -> None:
    """The shipped agent and the tested agent must be the same file."""
    assert _embedded_agent() == AGENT_PY.read_text()


def test_embedded_agent_compiles() -> None:
    compile(_embedded_agent(), "install.sh:MESHWEAVE_AGENT_EOF", "exec")


def test_runtime_fetch_list_matches_repository_files() -> None:
    """Every path the installer fetches for the runtime must exist locally.

    A 404 here is invisible at install time on a fresh node and only shows up
    when a stage tries to start.
    """
    text = INSTALL_SH.read_text()
    match = re.search(r"RUNTIME_MISSING=0\nfor f in \\\n(.*?); do", text, re.S)
    assert match, "runtime fetch list not found in install.sh"
    paths = [line.strip().rstrip("\\").strip() for line in match.group(1).splitlines()]
    paths = [p for p in paths if p]
    assert paths, "runtime fetch list is empty"
    for path in paths:
        assert (REPO / path).is_file(), f"installer fetches {path}, which does not exist"


def test_agent_reports_provider_declared_port_mappings() -> None:
    """mapped_ports is what makes a stage reachable from another provider."""
    source = AGENT_PY.read_text()
    assert "VAST_TCP_PORT_" in source
    assert '"mapped_ports"' in source


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, {}),
        ({"VAST_TCP_PORT_10100": "51400"}, {10100: 51400}),
        (
            {"VAST_TCP_PORT_10100": "51400", "VAST_TCP_PORT_10200": "46079"},
            {10100: 51400, 10200: 46079},
        ),
        ({"VAST_TCP_PORT_nonsense": "1", "VAST_TCP_PORT_22": "not-a-port"}, {}),
    ],
)
def test_published_port_parsing(env: dict[str, str], expected: dict[int, int]) -> None:
    """Parse the mapping with the real function, not a re-implementation."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("mw_agent", AGENT_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    saved = {k: v for k, v in os.environ.items() if k.startswith("VAST_TCP_PORT_")}
    for key in saved:
        del os.environ[key]
    try:
        os.environ.update(env)
        assert module._published_tcp_ports() == expected
    finally:
        for key in list(os.environ):
            if key.startswith("VAST_TCP_PORT_"):
                del os.environ[key]
        os.environ.update(saved)


# --------------------------------------------------------------------------- #
# node identity
# --------------------------------------------------------------------------- #


def _load_agent(identity_path: Path | None = None):
    """Import deploy/agent.py fresh, with identity confined to a temp file.

    Without the override these tests would write to the real ~/.meshweave or
    /var/lib path, which is exactly the kind of side effect a test must not have.
    """
    import importlib.util
    import sys

    sys.modules.pop("mw_agent_under_test", None)
    spec = importlib.util.spec_from_file_location("mw_agent_under_test", AGENT_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if identity_path is not None:
        module._identity_paths = lambda: [identity_path]
    return module


def test_generated_names_are_usable_hostnames() -> None:
    agent = _load_agent()
    pattern = re.compile(r"^[a-z]+-[a-z]+-[0-9a-f]{4}$")
    names = {agent.generate_node_id() for _ in range(200)}
    for name in names:
        assert pattern.match(name), name
        assert len(name) < 64
    # Random, not serial: 200 draws should not repeat.
    assert len(names) == 200


def test_identity_is_generated_once_and_then_reused(tmp_path: Path) -> None:
    """The load-bearing property: a restart must not create a second node."""
    path = tmp_path / "node_id"
    agent = _load_agent(path)

    first = agent.resolve_node_id(None, None)
    assert path.read_text().strip() == first
    assert agent.resolve_node_id(None, None) == first
    # A different process reading the same identity file must agree.
    assert _load_agent(path).resolve_node_id(None, None) == first


def test_explicit_names_win_over_remembered_ones(tmp_path: Path) -> None:
    path = tmp_path / "node_id"
    path.write_text("remembered-name\n")
    agent = _load_agent(path)

    assert agent.resolve_node_id("typed-name", None) == "typed-name"
    assert agent.resolve_node_id(None, "configured-name") == "configured-name"
    assert agent.resolve_node_id(None, None) == "remembered-name"


def test_identity_survives_an_unwritable_location(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked" / "node_id"
    (tmp_path / "blocked").write_text("not a directory")
    writable = tmp_path / "ok" / "node_id"
    agent = _load_agent()
    agent._identity_paths = lambda: [blocked, writable]

    name = agent.resolve_node_id(None, None)
    assert writable.read_text().strip() == name


def test_installer_generates_identity_when_not_given() -> None:
    text = INSTALL_SH.read_text()
    assert "--generate-node-id" in text
    assert "/etc/meshweave/node_id" in text or 'CONFIG_DIR/node_id' in text
    # The old hostname default must be gone: it produced names like C.61220645.
    assert 'NODE_ID="${NODE_ID:-$(hostname)}"' not in text


def test_installer_keeps_the_name_a_legacy_install_already_used() -> None:
    """Re-installing must not mint a second identity for one machine.

    Hosts enrolled before the identity file existed already registered under a
    name. The installer has to read it back out of agent.json, otherwise that
    node reappears as a healthy machine plus an unreachable ghost.
    """
    text = INSTALL_SH.read_text()
    assert 'CONFIG_DIR/agent.json' in text
    assert 'node_id") or ""' in text


def test_the_helpers_the_installer_needs_are_defined() -> None:
    """`log()` was deleted by an edit once, and the installer then failed on its
    first log line on every host. Presence is cheap to assert; the behavioural
    check is in the test below."""
    defined = set(re.findall(r"^(\w+)\(\)\s*\{", INSTALL_SH.read_text(), re.M))
    for required in ("log", "warn", "die", "_pid_cmdline", "_stop_existing_agent"):
        assert required in defined, f"{required}() is called but not defined"


def test_the_helpers_can_be_sourced_and_called() -> None:
    """Proves the definitions are valid shell, not just present as text."""
    text = INSTALL_SH.read_text()
    end = next(i for i, line in enumerate(text.splitlines()) if "this installer targets Linux" in line)
    body = "\n".join(text.splitlines()[:end])
    with tempfile.TemporaryDirectory() as tmp:
        defs = Path(tmp) / "defs.sh"
        defs.write_text(body)
        harness = Path(tmp) / "t.sh"
        harness.write_text(
            f'set -euo pipefail\nsource "{defs}"\nlog "ok"\nwarn "ok"\n'
            "echo \"pidfile=$PID_FILE\"\n"
        )
        result = subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, timeout=30
        )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert "/var/run/meshweave-agent.pid" in result.stdout


def test_the_detached_start_does_not_trust_a_setsid_wrapper_pid() -> None:
    """`setsid` forks when it is not a group leader, so `$!` was the wrapper's
    pid. The wrapper exits, the agent keeps running, and every later stop looked
    at a dead pid and left the old agent alive."""
    text = INSTALL_SH.read_text()
    body = re.search(r"^_start_detached\(\) \{(.*?)^\}", text, re.S | re.M)
    assert body, "_start_detached not found"
    # Ignore comments: the explanation of why setsid is gone mentions it.
    code = "\n".join(
        line for line in body.group(1).splitlines() if not line.lstrip().startswith("#")
    )
    assert "setsid" not in code
    assert "PID_FILE" in code


def test_a_leftover_agent_is_matched_by_exact_path() -> None:
    """The recovery for an install whose pidfile was wrong. The match is the
    install path, and _kill_pid is the only thing that signals anything."""
    text = INSTALL_SH.read_text()
    assert "_kill_pid() {" in text
    assert '"$INSTALL_DIR/agent.py"' in text


def test_the_leftover_agent_sweep_is_auditable() -> None:
    body = re.search(r"^_stop_existing_agent\(\) \{(.*?)^\}", INSTALL_SH.read_text(), re.S | re.M)
    assert body
    sweep = body.group(1)
    assert "/proc" in sweep
    assert "ps -eo pid=,command=" in sweep
    assert "_kill_pid" in sweep


def test_the_driver_version_is_read_without_a_device_handle() -> None:
    """nvmlSystemGetDriverVersion takes no handle. Passing one raised, the error
    was swallowed, and every node reported an unknown driver version."""
    source = AGENT_PY.read_text()
    call = "opt(pynvml.nvmlSystemGetDriverVersion)"
    assert call in source
    # It must not be inside the per-GPU loop's opt(handle) form.
    assert "opt(pynvml.nvmlSystemGetDriverVersion, h" not in source
