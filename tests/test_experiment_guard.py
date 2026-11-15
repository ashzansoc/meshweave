"""The simulation guard.

MESHWEAVE's first acceptance criterion is that no simulated GPU aggregation is
ever presented as actual distributed inference. This test enforces that
structurally, by static inspection, rather than trusting developers to remember.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"

FORBIDDEN_IMPORTS = {
    "meshweave.agent.synthetic",
    "meshweave.agent.simulated",
}


def _experiment_files() -> list[Path]:
    if not EXPERIMENTS.exists():
        return []
    return sorted(EXPERIMENTS.rglob("*.py"))


def test_experiment_directory_exists() -> None:
    assert EXPERIMENTS.exists(), "experiments/ must exist before experiments are written"


@pytest.mark.skipif(not EXPERIMENTS.exists(), reason="no experiments yet")
def test_experiments_do_not_import_synthetic_telemetry() -> None:
    """An experiment must never import the synthetic GPU source."""
    offenders: list[str] = []
    for path in _experiment_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_IMPORTS:
                offenders.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTS:
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, (
        "experiments import synthetic telemetry, which would let simulated data be "
        "presented as a real measurement:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(not EXPERIMENTS.exists(), reason="no experiments yet")
def test_experiments_do_not_hardcode_results() -> None:
    """A benchmark result must come from a measurement, not a literal.

    Looks for suspicious patterns like assigning a throughput figure directly
    without any measurement call in the same file.
    """
    offenders: list[str] = []
    for path in _experiment_files():
        source = path.read_text()
        has_measurement = any(
            marker in source
            for marker in ("measure_", "run_benchmark", "execute", "probe_", "collect")
        )
        # a file that claims a tokens_per_second but never measures anything
        if "tokens_per_second" in source and not has_measurement:
            offenders.append(str(path.name))
    assert not offenders, f"these experiments report throughput without measuring: {offenders}"


def test_synthetic_module_is_self_labelling() -> None:
    """The synthetic module must stamp its output so it is traceable."""
    from meshweave.agent.synthetic import synthetic_inventory

    inventory = synthetic_inventory()
    assert inventory.agent_version.endswith("synthetic")
    assert inventory.gpus[0].uuid.startswith("synthetic-")
    assert inventory.gpus[0].driver_version.endswith("synthetic")


def test_real_telemetry_never_returns_synthetic_data() -> None:
    """The NVML path must not fall back to synthetic devices on failure."""
    from meshweave.agent import telemetry

    try:
        gpus = telemetry.discover_gpus()
    except telemetry.NvmlUnavailableError:
        return  # correct behaviour on a host with no NVIDIA driver
    for gpu in gpus:
        assert not gpu.uuid.startswith("synthetic-")
        assert "synthetic" not in (gpu.driver_version or "")
