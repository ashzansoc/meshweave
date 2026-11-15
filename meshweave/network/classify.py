"""Link classification from measured data.

Classes are derived from measured RTT and packet loss only. Geography, provider
labels and host names are never inputs -- a host in Frankfurt is not
automatically "remote", and a host next door is not automatically "fast".

Thresholds live here, in one place, so they can be argued with in DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LinkClass = Literal[
    "LOW_LATENCY",
    "MODERATE_LATENCY",
    "HIGH_LATENCY",
    "UNRELIABLE",
]

# RTT (p50) ceilings in milliseconds.
LOW_LATENCY_MAX_MS = 5.0
MODERATE_LATENCY_MAX_MS = 50.0
# Above this, a link is high latency regardless of loss.

# Packet loss at or above this makes a link unreliable regardless of latency.
UNRELIABLE_LOSS_PCT = 2.0


@dataclass(frozen=True)
class LinkThresholds:
    low_latency_max_ms: float = LOW_LATENCY_MAX_MS
    moderate_latency_max_ms: float = MODERATE_LATENCY_MAX_MS
    unreliable_loss_pct: float = UNRELIABLE_LOSS_PCT


DEFAULT_THRESHOLDS = LinkThresholds()


def classify(
    rtt_ms_p50: float | None,
    packet_loss_pct: float | None,
    *,
    thresholds: LinkThresholds = DEFAULT_THRESHOLDS,
) -> LinkClass:
    """Classify one measured link.

    Loss takes precedence: a 1 ms link that drops 5% of packets is unusable for
    activation transfer regardless of its latency. A link with no measurement is
    classified ``UNRELIABLE`` rather than guessed at, because the planner must not
    treat an unmeasured link as fast.
    """
    if rtt_ms_p50 is None:
        return "UNRELIABLE"

    if packet_loss_pct is not None and packet_loss_pct >= thresholds.unreliable_loss_pct:
        return "UNRELIABLE"

    if rtt_ms_p50 <= thresholds.low_latency_max_ms:
        return "LOW_LATENCY"
    if rtt_ms_p50 <= thresholds.moderate_latency_max_ms:
        return "MODERATE_LATENCY"
    return "HIGH_LATENCY"


def suitability(cls: LinkClass) -> dict[str, str]:
    """What each class means for distributed execution.

    These are engineering statements about the link, not promises about
    performance. Actual viability still has to be measured.
    """
    return {
        "LOW_LATENCY": {
            "tensor_parallel": "possible",
            "pipeline": "excellent",
            "note": "one round trip per token is cheap; TP may be viable",
        },
        "MODERATE_LATENCY": {
            "tensor_parallel": "unlikely",
            "pipeline": "usable",
            "note": "per-token latency is dominated by the network; pipeline only",
        },
        "HIGH_LATENCY": {
            "tensor_parallel": "no",
            "pipeline": "capacity only",
            "note": "pipeline buys memory capacity but per-token latency is poor",
        },
        "UNRELIABLE": {
            "tensor_parallel": "no",
            "pipeline": "not recommended",
            "note": "unmeasured or lossy; do not place a pipeline stage here",
        },
    }[cls]
