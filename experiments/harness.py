"""Shared harness for experiment scripts.

Every experiment goes through this module so that provenance, cost accounting and
the honesty checks are identical across all of them. An experiment script decides
*what to measure*; this module decides *how it is recorded*.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshweave.bench.report import (
    EconomicReport,
    ExperimentReport,
    FailureReport,
    Measurement,
    collect_provenance,
    validate_report,
    write_report,
)
from meshweave.errors import MeshweaveError

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "reports" / "raw"


class ExperimentBlockedError(MeshweaveError):
    """The experiment cannot run in this environment.

    Raised instead of producing a report that pretends otherwise. The caller
    converts it into a ``status="blocked"`` report with a reason and the exact
    command that would run it once the blocker is cleared.
    """

    def __init__(self, reason: str, command: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.command = command


@dataclass
class Experiment:
    """One experiment's identity, hypothesis and evidence collection."""

    experiment_id: str
    title: str
    hypothesis: str
    command: str
    measurements: list[Measurement] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    economics: EconomicReport | None = None
    nodes: list[dict[str, Any]] = field(default_factory=list)
    strategy: str | None = None

    def measure(
        self, name: str, value: float | None, unit: str, samples: int = 1, note: str | None = None
    ) -> None:
        """Record a measurement. ``value=None`` is allowed only with a note."""
        self.measurements.append(
            Measurement(name=name, value=value, unit=unit, samples=samples, note=note)
        )

    def fail(self, kind: str, detail: str) -> None:
        """Record a failure. Never let a failure vanish silently."""
        self.failures.append({"kind": kind, "detail": detail, "at": datetime.now(UTC).isoformat()})

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail("precondition", message)
            raise ExperimentBlockedError(message)

    def finish(
        self,
        *,
        status: str = "completed",
        blocked_reason: str | None = None,
        ready_to_run_command: str | None = None,
        out: Path | None = None,
    ) -> ExperimentReport:
        economics = self.economics or EconomicReport(
            cost_known=False, unknown_components=["hourly_cost_usd (not supplied)"]
        )
        report = ExperimentReport(
            experiment_id=self.experiment_id,
            title=self.title,
            hypothesis=self.hypothesis,
            provenance=collect_provenance(self.command),
            status=status,
            nodes=self.nodes,
            strategy=self.strategy,
            measurements=self.measurements,
            economics=economics,
            failures=FailureReport(
                failures=self.failures,
                failures_investigated=True,
            ),
            findings=self.findings,
            evidence=self.evidence,
            blocked_reason=blocked_reason,
            ready_to_run_command=ready_to_run_command or self.command,
        )
        validate_report(report)

        if status == "completed" and not self.measurements:
            raise MeshweaveError("a completed experiment must record at least one measurement")

        target = out or (RAW_DIR / f"{self.experiment_id.lower()}.json")
        target = Path(target)
        if status == "completed":
            write_report(report, target)
            self.evidence.append(str(_relative(target)))
            print(f"\nwrote {target}", file=sys.stderr)
        else:
            # A blocked experiment still gets a record, so the gap is visible.
            write_report(report, target)
            print(f"\nBLOCKED: {blocked_reason}", file=sys.stderr)
            if ready_to_run_command:
                print(f"ready to run: {ready_to_run_command}", file=sys.stderr)

        return report


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check prerequisites and exit without renting anything",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the JSON report",
    )


def require_vast_key() -> None:
    """Raise ExperimentBlockedError if no Vast credentials are configured."""
    from meshweave.config import VastSettings

    try:
        VastSettings().require_key()
    except RuntimeError as exc:
        raise ExperimentBlockedError(
            str(exc),
            "export VAST_API_KEY=<your key>  # then re-run",
        ) from exc


def require_env(var: str, hint: str) -> str:
    import os

    value = os.environ.get(var)
    if not value:
        raise ExperimentBlockedError(
            f"{var} is not set",
            f"export {var}=<value>   # {hint}",
        )
    return value


def emit(report: ExperimentReport) -> None:
    """Print a compact human summary alongside the machine-readable report."""
    print(f"\n{'=' * 68}")
    print(f"{report.experiment_id}: {report.title}")
    print(f"{'=' * 68}")
    print(f"status      : {report.status}")
    if report.blocked_reason:
        print(f"blocked     : {report.blocked_reason}")
        print(f"run with    : {report.ready_to_run_command}")
        return

    print(f"hypothesis  : {report.hypothesis}")
    if report.measurements:
        print("\nmeasurements")
        for m in report.measurements:
            value = f"{m.value:.3f}" if m.value is not None else "n/a"
            note = f"  ({m.note})" if m.note else ""
            print(f"  {m.name:38s} {value:>12s} {m.unit}{note}")
    if report.findings:
        print("\nfindings")
        for f in report.findings:
            print(f"  - {f}")
    if report.failures.failures:
        print(f"\nfailures ({len(report.failures.failures)})")
        for failure in report.failures.failures:
            print(f"  - [{failure['kind']}] {failure['detail']}")
    eco = report.economics
    if eco.cost_known:
        print("\neconomics")
        if eco.total_cost_usd is not None:
            print(f"  total cost                    ${eco.total_cost_usd:.4f}")
        if eco.cost_per_m_output_tokens_usd is not None:
            print(f"  cost per M output tokens      ${eco.cost_per_m_output_tokens_usd:.2f}")
    else:
        print(f"\neconomics   : cost unknown ({', '.join(eco.unknown_components)})")


def write_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def _relative(path: Path) -> Path:
    """Path relative to the repo root when possible, else absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
