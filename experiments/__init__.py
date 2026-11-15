"""Experiment entry points for MESHWEAVE.

Each experiment is a module exposing ``main(argv)``. It either produces a
machine-readable report or raises ``ExperimentBlockedError`` naming the reason
and the command that would run it once the blocker is cleared.

Run one with:
    uv run python -m experiments.exp001_single_gpu --dry-run

A blocked experiment is a legitimate result. A fabricated one is not.
"""

from __future__ import annotations

__all__: list[str] = []
