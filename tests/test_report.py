"""The report validator must reject dishonest reports.

Acceptance criteria 2, 3 and 4 are enforced here: reproducible provenance,
reported failures, and cost per token on every configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from meshweave.bench.report import (
    EconomicReport,
    ExperimentReport,
    FailureReport,
    Measurement,
    Provenance,
    validate_report,
)
from meshweave.errors import ReportValidationError


def provenance(**overrides: object) -> Provenance:
    kwargs = {
        "generated_at": datetime.now(UTC),
        "host": "test-host",
        "python_version": "3.13.0",
        "meshweave_version": "0.1.0",
        "command": "pytest",
    }
    kwargs.update(overrides)
    return Provenance(**kwargs)  # type: ignore[arg-type]


def test_completed_experiment_requires_measurements() -> None:
    report = ExperimentReport(
        experiment_id="EXP-001",
        title="baseline",
        hypothesis="h",
        provenance=provenance(),
        status="completed",
        measurements=[],
        economics=EconomicReport(cost_known=True, hourly_cost_usd=0.3),
        failures=FailureReport(failures_investigated=True),
        findings=["something"],
        evidence=["reports/raw/exp001.json"],
    )
    with pytest.raises(ReportValidationError, match="measurements"):
        validate_report(report)


def test_completed_experiment_requires_failure_investigation() -> None:
    report = ExperimentReport(
        experiment_id="EXP-002",
        title="t",
        hypothesis="h",
        provenance=provenance(),
        status="completed",
        measurements=[Measurement(name="tps", value=25.0, unit="tok/s", samples=3)],
        economics=EconomicReport(cost_known=True, hourly_cost_usd=0.3),
        failures=FailureReport(failures_investigated=False),
        findings=["f"],
        evidence=["e"],
    )
    with pytest.raises(ReportValidationError, match="failures_investigated"):
        validate_report(report)


def test_cost_per_token_cannot_be_claimed_when_cost_unknown() -> None:
    with pytest.raises(ValueError, match="cost per token"):
        EconomicReport(cost_known=False, cost_per_m_output_tokens_usd=1.5)


def test_measurement_without_value_needs_a_note() -> None:
    with pytest.raises(ValueError, match="note"):
        Measurement(name="tps", value=None, unit="tok/s")


def test_measurement_with_note_is_allowed() -> None:
    m = Measurement(name="tps", value=None, unit="tok/s", note="no GPU available")
    assert m.value is None


def test_blocked_status_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="blocked_reason"):
        ExperimentReport(
            experiment_id="EXP-003",
            title="t",
            hypothesis="h",
            provenance=provenance(),
            status="blocked",
            measurements=[],
            economics=EconomicReport(cost_known=False),
            failures=FailureReport(),
            findings=[],
            evidence=[],
        )

    # supplying the reason makes it valid
    valid = ExperimentReport(
        experiment_id="EXP-003",
        title="t",
        hypothesis="h",
        provenance=provenance(),
        status="blocked",
        measurements=[],
        economics=EconomicReport(cost_known=False, unknown_components=["hourly_cost_usd"]),
        failures=FailureReport(),
        findings=[],
        evidence=[],
        blocked_reason="no VAST_API_KEY configured",
    )
    assert validate_report(valid).blocked_reason == "no VAST_API_KEY configured"


def test_synthetic_report_is_never_evidence() -> None:
    report = ExperimentReport(
        experiment_id="EXP-004",
        title="t",
        hypothesis="h",
        provenance=provenance(synthetic=True),
        status="completed",
        measurements=[Measurement(name="tps", value=25.0, unit="tok/s", samples=3)],
        economics=EconomicReport(cost_known=True, hourly_cost_usd=0.3),
        failures=FailureReport(failures_investigated=True),
        findings=["real distributed execution achieved"],
        evidence=["reports/raw/x.json"],
    )
    assert report.claim_is_supported() is False

    with pytest.raises(ReportValidationError, match="may not claim a real result"):
        validate_report(report)


def test_real_completed_report_is_supported() -> None:
    report = ExperimentReport(
        experiment_id="EXP-005",
        title="t",
        hypothesis="h",
        provenance=provenance(),
        status="completed",
        measurements=[Measurement(name="tps", value=25.0, unit="tok/s", samples=3)],
        economics=EconomicReport(cost_known=True, hourly_cost_usd=0.3),
        failures=FailureReport(failures_investigated=True),
        findings=["measured 25 tok/s"],
        evidence=["reports/raw/exp005.json"],
    )
    assert report.claim_is_supported() is True


def test_unknown_cost_must_list_components() -> None:
    report = ExperimentReport(
        experiment_id="EXP-006",
        title="t",
        hypothesis="h",
        provenance=provenance(),
        status="completed",
        measurements=[Measurement(name="tps", value=25.0, unit="tok/s", samples=3)],
        economics=EconomicReport(cost_known=False, unknown_components=[]),
        failures=FailureReport(failures_investigated=True),
        findings=["f"],
        evidence=["e"],
    )
    with pytest.raises(ReportValidationError, match="unknown_components"):
        validate_report(report)
