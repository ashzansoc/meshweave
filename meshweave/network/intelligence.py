"""Decide whether a measured link suits a given parallelism strategy.

A single RTT threshold is useless here: a 300 ms link is fatal for tensor
parallelism, irrelevant for replication, and merely slow for a pipeline. The
comparison has to be between the **communication a strategy requires** and the
**compute measured on this hardware**.

That means the inputs are measured, not assumed:

  * RTT, jitter and throughput come from a completed link measurement
  * bytes per token and hops come from the model's geometry and the stage count
  * compute time per token comes from a benchmark run, and if there is none the
    verdict is reported as indeterminate rather than filled with a guess

The rule is stated in one place, in ``VERDICT_RATIOS``, and every verdict carries
the numbers it was computed from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# communication time as a fraction of compute time. Below 0.5 the network is a
# minor cost; above 2.0 it dominates and the strategy is not worth running.
VERDICT_RATIOS = (0.5, 2.0)

STRATEGIES = ("PIPELINE", "TENSOR", "REPLICATE")


@dataclass(frozen=True)
class LinkView:
    source: str
    target: str
    rtt_p50_ms: float
    jitter_ms: float | None
    download_mbps: float | None
    ok: bool


def _ms_for_bytes(mbps: float | None, nbytes: float) -> float | None:
    if not mbps or mbps <= 0:
        return None
    return (nbytes * 8.0) / (mbps * 1e6) * 1000.0


def pipeline_communication_ms(
    link: LinkView, *, hops: int, bytes_per_token: float
) -> float | None:
    """One activation per hop per token, plus the return of the token id."""
    if not link.ok or link.rtt_p50_ms is None:
        return None
    transfer = _ms_for_bytes(link.download_mbps, bytes_per_token)
    per_hop = link.rtt_p50_ms + (transfer or 0.0)
    return round(per_hops(per_hop, hops), 2)


def per_hops(per_hop_ms: float, hops: int) -> float:
    return per_hop_ms * max(1, hops)


def tensor_communication_ms(
    link: LinkView, *, layers: int, params_per_layer: int, bytes_per_param: int = 2
) -> float | None:
    """One all-reduce per layer per forward pass.

    Each layer's parameters are gathered and reduced across ranks, so the volume
    is the whole model's parameters times the ring factor. This is why tensor
    parallelism is a datacenter technique.
    """
    if not link.ok:
        return None
    volume = layers * params_per_layer * bytes_per_param * 2
    return _ms_for_bytes(link.download_mbps, volume)


def _verdict(fraction: float | None) -> str:
    if fraction is None:
        return "indeterminate"
    low, high = VERDICT_RATIOS
    if fraction <= low:
        return "suitable"
    if fraction <= high:
        return "marginal"
    return "unsuitable"


def assess_strategy(
    link: LinkView,
    *,
    compute_ms_per_token: float | None,
    hops: int = 1,
    bytes_per_token: float = 0.0,
    layers: int = 0,
    params_per_layer: int = 0,
) -> dict[str, Any]:
    """One strategy, one link, one verdict with the arithmetic attached."""
    compute = compute_ms_per_token if compute_ms_per_token and compute_ms_per_token > 0 else None

    if not link.ok:
        return {
            "strategy": None,
            "verdict": "no_link",
            "reason": "the link did not carry a measurement",
            "communication_ms": None,
            "compute_ms_per_token": compute,
            "fraction_of_compute": None,
        }

    strategies: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        if strategy == "PIPELINE":
            communication = pipeline_communication_ms(
                link, hops=hops, bytes_per_token=bytes_per_token
            )
            basis = f"{hops} hop(s) x (RTT {link.rtt_p50_ms:.1f} ms + {bytes_per_token:.0f} B transfer)"
        elif strategy == "TENSOR":
            communication = tensor_communication_ms(
                link, layers=layers, params_per_layer=params_per_layer
            )
            basis = f"{layers} layer all-reduce(s)"
        else:
            communication = 0.0
            basis = "no inter-node traffic; each node holds a full copy"

        fraction = None
        if communication is not None and compute:
            fraction = round(communication / compute, 3)
        strategies.append(
            {
                "strategy": strategy,
                "verdict": _verdict(fraction),
                "basis": basis,
                "communication_ms": (
                    round(communication, 2) if communication is not None else None
                ),
                "compute_ms_per_token": (
                    round(compute, 2) if compute is not None else None
                ),
                "fraction_of_compute": fraction,
            }
        )

    return {
        "source": link.source,
        "target": link.target,
        "rtt_p50_ms": link.rtt_p50_ms,
        "jitter_ms": link.jitter_ms,
        "download_mbps": link.download_mbps,
        "verdicts": strategies,
        "verdict_scale": {
            "suitable": f"communication <= {VERDICT_RATIOS[0]}x compute",
            "marginal": f"communication between {VERDICT_RATIOS[0]}x and {VERDICT_RATIOS[1]}x compute",
            "unsuitable": f"communication > {VERDICT_RATIOS[1]}x compute",
            "indeterminate": "no measured compute time to compare against",
            "no_link": "the link was not measured",
        },
    }