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
