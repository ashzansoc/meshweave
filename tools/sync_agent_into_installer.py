"""Copy deploy/agent.py into the installer's heredoc.

The installer carries the agent inline so a host needs one file, which means the
same code exists twice. This rewrites the embedded copy from the source of
truth; ``tests/test_installer.py`` fails if the two ever drift.

    python3 tools/sync_agent_into_installer.py [--check]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT = REPO / "deploy" / "agent.py"
INSTALLER = REPO / "deploy" / "install.sh"

PATTERN = re.compile(r"(<<'MESHWEAVE_AGENT_EOF'\n)(.*?)(\nMESHWEAVE_AGENT_EOF)", re.S)


def main(argv: list[str]) -> int:
    check = "--check" in argv
    installer = INSTALLER.read_text()
    match = PATTERN.search(installer)
    if not match:
        print("error: agent heredoc not found in install.sh", file=sys.stderr)
        return 1

    agent = AGENT.read_text()
    if match.group(2) + "\n" == agent:
        print("agent in install.sh is already current")
        return 0

    if check:
        print("error: install.sh embeds a stale agent; run without --check", file=sys.stderr)
        return 1

    updated = installer[: match.start(2)] + agent.rstrip("\n") + installer[match.end(2) :]
    INSTALLER.write_text(updated)
    print(f"rewrote the embedded agent from {AGENT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
