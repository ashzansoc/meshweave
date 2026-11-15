"""Experiment report schema and validator.

The validator exists to enforce MESHWEAVE's acceptance criteria mechanically.
A report that claims distributed execution without evidence, that omits cost, or
that hides failures is rejected here rather than published.

Required for any report that claims a benchmark:
  * provenance: git commit, timestamps, host, whether anything was synthetic
  * measured values, with units
  * a failures section (may be empty, but must be present and deliberate)
  * an economic section (cost per token, or an explicit "unknown")
  * the evidence that the numbers came from an actual run

A report with ``synthetic=true`` is structurally valid but is *never* counted as
evidence of distributed execution. That is the distinction the acceptance
criteria care about most.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from meshweave.errors import ReportValidationError


class Provenance(BaseModel):
    git_commit: str | None = None
    git_dirty: bool | None = None
    generated_at: datetime
    host: str
    python_version: str
    meshweave_version: str
    synthetic: bool = False
    command: str
    config: dict[str, Any] = Field(default_factory=dict)


class EconomicReport(BaseModel):
    """Cost accounting. ``cost_known=false`` is a valid, honest answer."""

    cost_known: bool
    hourly_cost_usd: float | None = None
    total_cost_usd: float | None = None
    cost_per_m_input_tokens_usd: float | None = None
    cost_per_m_output_tokens_usd: float | None = None
    cost_per_request_usd: float | None = None
    unknown_components: list[str] = Field(default_factory=list)

    @field_validator("cost_per_m_output_tokens_usd")
    @classmethod
    def _cost_requires_known(cls, v: float | None, info: Any) -> float | None:
        if v is not None and not info.data.get("cost_known", False):
            raise ValueError("cost per token cannot be reported when cost_known is false")
        return v


class FailureReport(BaseModel):
    """What broke. An empty list must be deliberate, not accidental."""

    failures: list[dict[str, Any]] = Field(default_factory=list)
    failures_investigated: bool = False
    notes: str | None = None


class Measurement(BaseModel):
    name: str
    value: float | None
    unit: str
    samples: int = 0
    note: str | None = None

    @model_validator(mode="after")
    def _must_have_value_or_note(self) -> Measurement:
        # A measurement with neither a value nor an explanation is a placeholder.
        if self.value is None and not self.note:
            raise ValueError("a measurement without a value must carry a note explaining why")
        return self


class ExperimentReport(BaseModel):
    schema_version: str = "1"
    experiment_id: str
    title: str
    hypothesis: str
    provenance: Provenance

    status: str  # completed | blocked | failed | partial

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    strategy: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    economics: EconomicReport
    failures: FailureReport

    # Explicit claims and the evidence backing them.
    findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    # Set when the experiment could not be run. A blocked experiment is a
    # legitimate result; a fabricated one is not.
    blocked_reason: str | None = None
    ready_to_run_command: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> ExperimentReport:
        """Cross-field rules.

        These must be model validators: a field validator on an optional field
        never fires when the field is omitted, which is exactly the case these
        rules exist to catch.
        """
        if self.status in ("blocked", "failed") and not self.blocked_reason:
            raise ValueError(f"status={self.status} requires blocked_reason")
        if self.status == "completed" and not self.evidence:
            raise ValueError("a completed experiment must cite its evidence")
        return self

    def claim_is_supported(self) -> bool:
        """Can this report support a claim of real distributed execution?"""
        if self.provenance.synthetic:
            return False
        if self.status != "completed":
            return False
        return bool(self.evidence)


def validate_report(report: ExperimentReport) -> ExperimentReport:
    """Run every check. Raises ReportValidationError on the first failure."""
    if report.status not in ("completed", "blocked", "failed", "partial"):
        raise ReportValidationError(f"unknown status {report.status!r}")

    if report.status == "completed":
        if not report.measurements:
            raise ReportValidationError(
                f"{report.experiment_id}: a completed experiment must record measurements"
            )
        if not report.economics.cost_known and not report.economics.unknown_components:
            raise ReportValidationError(
                f"{report.experiment_id}: cost is unknown but no unknown_components "
                "are listed. State what was missing."
            )
        if not report.failures.failures_investigated:
            raise ReportValidationError(
                f"{report.experiment_id}: failures_investigated is false. "
                "A completed run must state whether failures were examined."
            )

    if report.provenance.synthetic and report.findings:
        for finding in report.findings:
            if "real" in finding.lower() and "synthetic" not in finding.lower():
                raise ReportValidationError(
                    f"{report.experiment_id}: synthetic report may not claim a real result"
                )

    return report


def write_report(report: ExperimentReport, path: str | Path) -> Path:
    validated = validate_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as fh:
        json.dump(validated.model_dump(mode="json"), fh, indent=2)
        fh.write("\n")
    return target


def load_report(path: str | Path) -> ExperimentReport:
    with Path(path).open() as fh:
        return ExperimentReport.model_validate(json.load(fh))


def collect_provenance(command: str, config: dict[str, Any] | None = None) -> Provenance:
    """Capture what ran, from where, on what."""
    import platform
    import subprocess

    from meshweave import __version__

    commit = None
    dirty = None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            .stdout.strip()
            or None
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        dirty = bool(status)
    except (OSError, subprocess.SubprocessError):
        pass

    return Provenance(
        git_commit=commit,
        git_dirty=dirty,
        generated_at=datetime.now().astimezone(),
        host=platform.node(),
        python_version=platform.python_version(),
        meshweave_version=__version__,
        synthetic=False,
        command=command,
        config=config or {},
    )
