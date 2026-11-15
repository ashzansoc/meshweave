"""Cost accounting and execution economics.

MESHWEAVE's acceptance criteria require every configuration to report cost per
token. This module is the single place that arithmetic lives, so the dashboard,
the planner and every experiment report agree.

Two things this module refuses to do:
  * Guess an hourly price. Unknown cost is ``None`` and propagates.
  * Report cost per token without a token count.

Cost model
----------
hourly_cost_usd  ->  cost per second
gpu_seconds      ->  integral of allocated GPUs over wall time
energy           ->  optional; only counted when a power measurement exists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshweave.core import ExecutionRecord, GpuDevice, NodeRecord, ParallelStrategy


@dataclass
class CostInputs:
    """Everything needed to price one execution."""

    hourly_cost_usd: float | None
    gpu_count: int = 1
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    power_watts: float | None = None
    energy_price_per_kwh: float = 0.15

    @property
    def cost_known(self) -> bool:
        return self.hourly_cost_usd is not None

    def compute(self) -> CostResult:
        return compute_cost(self)


@dataclass
class CostResult:
    cost_usd: float | None
    cost_per_m_input_tokens_usd: float | None
    cost_per_m_output_tokens_usd: float | None
    cost_per_request_usd: float | None
    gpu_seconds: float | None
    energy_kwh: float | None
    energy_cost_usd: float | None
    unknown_components: list[str] = field(default_factory=list)


def compute_cost(inputs: CostInputs) -> CostResult:
    unknown: list[str] = []

    if inputs.hourly_cost_usd is None:
        unknown.append("hourly_cost_usd")
        compute_cost_usd: float | None = None
    else:
        compute_cost_usd = inputs.hourly_cost_usd * (inputs.duration_s / 3600.0)

    energy_kwh: float | None = None
    energy_cost: float | None = None
    if inputs.power_watts is None or inputs.gpu_count <= 0:
        unknown.append("power_watts")
    else:
        energy_kwh = (inputs.power_watts * inputs.gpu_count * inputs.duration_s) / 3.6e6
        energy_cost = energy_kwh * inputs.energy_price_per_kwh

    if compute_cost_usd is None and energy_cost is None:
        total: float | None = None
    else:
        total = (compute_cost_usd or 0.0) + (energy_cost or 0.0)

    per_out = None
    per_in = None
    per_req = None
    if total is not None:
        if inputs.output_tokens > 0:
            per_out = total / inputs.output_tokens * 1e6
        if inputs.input_tokens > 0:
            per_in = total / inputs.input_tokens * 1e6
        per_req = total

    gpu_seconds = None
    if inputs.duration_s > 0:
        gpu_seconds = inputs.duration_s * inputs.gpu_count

    return CostResult(
        cost_usd=total,
        cost_per_m_input_tokens_usd=per_in,
        cost_per_m_output_tokens_usd=per_out,
        cost_per_request_usd=per_req,
        gpu_seconds=gpu_seconds,
        energy_kwh=energy_kwh,
        energy_cost_usd=energy_cost,
        unknown_components=unknown,
    )


def node_hourly_cost(node: NodeRecord) -> float | None:
    return node.hourly_cost_usd


def cluster_hourly_cost(nodes: list[NodeRecord]) -> float | None:
    costs = [n.hourly_cost_usd for n in nodes]
    if not costs or any(c is None for c in costs):
        return None
    return float(sum(c for c in costs if c is not None))


def estimate_node_power(gpus: list[GpuDevice]) -> float | None:
    """Sum of nameplate power limits. An upper bound, not a measurement."""
    values = [g.power_limit_w for g in gpus if g.power_limit_w is not None]
    if not values:
        return None
    return float(sum(values))


def attach_cost_to_record(record: ExecutionRecord, result: CostResult) -> ExecutionRecord:
    record.cost_usd = result.cost_usd
    record.cost_per_m_output_tokens_usd = result.cost_per_m_output_tokens_usd
    return record


# --------------------------------------------------------------------------- #
# Crossover analysis
# --------------------------------------------------------------------------- #


@dataclass
class ConfigurationCost:
    """One candidate deployment, fully costed."""

    label: str
    strategy: ParallelStrategy
    node_count: int
    gpu_count: int
    hourly_cost_usd: float | None
    measured_throughput_tps: float | None
    measured_ttft_ms: float | None = None

    def cost_per_m_output_tokens(self) -> float | None:
        if (
            self.hourly_cost_usd is None
            or self.measured_throughput_tps is None
            or self.measured_throughput_tps <= 0
        ):
            return None
        tokens_per_hour = self.measured_throughput_tps * 3600.0
        return self.hourly_cost_usd / tokens_per_hour * 1e6

    def throughput_per_dollar(self) -> float | None:
        if not self.hourly_cost_usd or self.measured_throughput_tps is None:
            return None
        return self.measured_throughput_tps / self.hourly_cost_usd


def compare_configurations(configs: list[ConfigurationCost]) -> dict[str, Any]:
    """Rank candidate configurations on $/M tokens and tok/s per dollar.

    This is the artefact that answers "is distributing worth it?". It reports
    the winner even when the winner is "do not distribute".
    """
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        rows.append(
            {
                "label": cfg.label,
                "strategy": cfg.strategy.value,
                "nodes": cfg.node_count,
                "gpus": cfg.gpu_count,
                "hourly_cost_usd": cfg.hourly_cost_usd,
                "throughput_tps": cfg.measured_throughput_tps,
                "cost_per_m_output_tokens_usd": cfg.cost_per_m_output_tokens(),
                "throughput_per_dollar": cfg.throughput_per_dollar(),
            }
        )

    priced = [r for r in rows if r["cost_per_m_output_tokens_usd"] is not None]
    ranked = sorted(
        priced, key=lambda r: float(r["cost_per_m_output_tokens_usd"] or 0.0)
    )
    cheapest = ranked[0] if ranked else None

    # Crossover: the point at which adding nodes stops lowering $/M tokens.
    crossover = _find_crossover(configs)

    return {
        "configurations": rows,
        "cheapest_by_cost_per_m_output_tokens": cheapest,
        "fastest_by_throughput_per_dollar": max(
            (r for r in rows if r["throughput_per_dollar"] is not None),
            key=lambda r: float(r["throughput_per_dollar"] or 0.0),
            default=None,
        ),
        "unpriced": [r["label"] for r in rows if r["cost_per_m_output_tokens_usd"] is None],
        "crossover": crossover,
    }


def _find_crossover(configs: list[ConfigurationCost]) -> dict[str, Any] | None:
    """Find where scaling stops improving cost per token.

    A crossover means: adding the next node did not reduce $/M tokens. That is
    the economically useful number -- it tells you when to stop renting.
    """
    ordered = sorted(configs, key=lambda c: c.node_count)
    points = []
    for cfg in ordered:
        c = cfg.cost_per_m_output_tokens()
        if c is not None:
            points.append((cfg.node_count, c, cfg.label))
    if len(points) < 2:
        return None

    for i in range(1, len(points)):
        if points[i][1] >= points[i - 1][1]:
            return {
                "beyond_node_count": points[i - 1][0],
                "at_node_count": points[i][0],
                "cost_before": points[i - 1][1],
                "cost_after": points[i][1],
                "explanation": (
                    "adding nodes no longer lowers cost per million output tokens; "
                    "the arrangement has stopped being economically useful"
                ),
            }
    return {
        "beyond_node_count": None,
        "at_node_count": None,
        "cost_before": None,
        "cost_after": None,
        "explanation": "cost per million output tokens still improving at the largest tested scale",
    }
