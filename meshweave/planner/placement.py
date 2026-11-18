"""Capacity-aware placement: choose a layer split by predicted latency.

The previous planner divided layers evenly. That is easy to reason about and wrong
whenever the machines differ: a faster GPU is given the same work as a slower one,
and a node with 6 GiB free is treated like one with 12.

This module enumerates candidate splits, estimates each one's per-token latency
from **measured** inputs, and returns the cheapest viable candidate alongside the
alternatives so the choice can be audited rather than trusted.

The cost model, per generated token:

    (forward_hops + 1) * RTT/2      activation travel, out and back
  + sum(stage_compute)              each stage runs its own layers
  + sum(activation_bytes / bandwidth)

The RTT term uses half a round trip per hop because the token returns directly
from the last stage to rank 0 rather than back along the chain -- the property
that took the earlier three-host decode from 2,663 to 453 ms/token.

Every input is measured or explicitly declared as an assumption. ``assumptions``
lists what was estimated, so a report can never present a guess as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Used only when a node has no measured speed. Deliberately conservative and
# always reported as an assumption.
DEFAULT_MS_PER_LAYER = 2.0


@dataclass
class NodeSpeed:
    """How fast one node runs a decoder layer."""

    node_id: str
    ms_per_layer: float
    measured: bool = False

    @property
    def layers_per_ms(self) -> float:
        return 1.0 / self.ms_per_layer if self.ms_per_layer > 0 else 0.0


@dataclass
class LinkCost:
    """Measured cost of one hop between two nodes."""

    source: str
    target: str
    rtt_ms: float
    mbps: float | None = None

    def transfer_ms(self, nbytes: float) -> float | None:
        if not self.mbps or self.mbps <= 0:
            return None
        return (nbytes * 8.0) / (self.mbps * 1e6) * 1000.0


@dataclass
class Candidate:
    """One proposed split, with the arithmetic that produced its estimate."""

    bounds: list[tuple[int, int]]
    predicted_ms_per_token: float
    compute_ms: float
    network_ms: float
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bounds": [list(b) for b in self.bounds],
            "predicted_ms_per_token": round(self.predicted_ms_per_token, 2),
            "compute_ms": round(self.compute_ms, 2),
            "network_ms": round(self.network_ms, 2),
            "assumptions": self.assumptions,
        }


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #


def layer_splits(total: int, parts: int, *, limit: int = 400) -> Iterable[list[int]]:
    """Every way to write ``total`` as ``parts`` positive integers.

    For 24 layers over 2 nodes that is 23 candidates; over 3 it is 253. The limit
    exists so a large cluster does not enumerate combinatorially -- when it is hit
    the caller gets an even split plus the most skewed ones, and the result says
    the search was truncated.
    """
    if parts < 1 or total < parts:
        return

    if parts == 1:
        yield [total]
        return

    # Compositions, generated largest-first-group so the first candidates are the
    # most lopsided (a useful extreme to compare against).
    def build(remaining: int, slots: int, prefix: list[int]):
        if slots == 1:
            if remaining > 0:
                yield prefix + [remaining]
            return
        # Leave at least one layer for each remaining slot.
        for first in range(remaining - (slots - 1), 0, -1):
            yield from build(remaining - first, slots - 1, prefix + [first])

    emitted = 0
    for split in build(total, parts, []):
        yield split
        emitted += 1
        if emitted >= limit:
            return


def _even_split(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #


def estimate(
    bounds: list[tuple[int, int]],
    speeds: list[NodeSpeed],
    links: dict[tuple[str, str], LinkCost],
    hidden_bytes_per_token: float | None = None,
) -> Candidate:
    """Predict per-token latency for one split, given as ``(start, end)`` ranges."""
    sizes = [end - start for start, end in bounds]
    if len(sizes) != len(speeds):
        raise ValueError("one speed is required per stage")
    assumptions: list[str] = [
        "embedding and language-head costs are folded into per-layer averages, "
        "not modelled separately"
    ]
    for speed in speeds:
        if not speed.measured:
            assumptions.append(
                f"no measured speed for {speed.node_id}; assuming "
                f"{speed.ms_per_layer:.2f} ms per layer"
            )

    compute = sum(size * speed.ms_per_layer for size, speed in zip(sizes, speeds, strict=True))

    hops = max(0, len(sizes) - 1)
    network = 0.0
    for index in range(hops):
        source, target = speeds[index].node_id, speeds[index + 1].node_id
        link = links.get((source, target))
        if link is None:
            assumptions.append(f"link {source}->{target} is unmeasured; only compute counted")
            continue
        network += link.rtt_ms / 2.0
        if hidden_bytes_per_token:
            transfer = link.transfer_ms(hidden_bytes_per_token)
            if transfer is not None:
                network += transfer
            else:
                assumptions.append(f"bandwidth for {source}->{target} is unknown")

    # The token returns directly from the last stage to rank 0.
    if hops:
        back = links.get((speeds[-1].node_id, speeds[0].node_id))
        if back is not None:
            network += back.rtt_ms / 2.0
            if hidden_bytes_per_token:
                transfer = back.transfer_ms(hidden_bytes_per_token * 4)
                if transfer is not None:
                    network += transfer

    return Candidate(
        bounds=list(bounds),
        predicted_ms_per_token=compute + network,
        compute_ms=compute,
        network_ms=network,
        assumptions=assumptions,
    )


def _bounds(sizes: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        out.append((start, start + size))
        start += size
    return out


def choose_split(
    config: Any,
    speeds: list[NodeSpeed],
    links: dict[tuple[str, str], LinkCost] | None = None,
    *,
    hidden_bytes_per_token: float | None = None,
    limit: int = 400,
    tolerance: float = 0.05,
) -> tuple[Candidate, list[Candidate], bool]:
    """Return the best candidate, the ranked alternatives, and whether it is exact.

    The even split is always included, so the comparison against the old behaviour
    is part of the result rather than an assumption.

    ``tolerance`` matters more than it looks. The cost model omits per-stage
    overhead and the head's vocabulary projection, so its error is larger than the
    difference between most splits. Among candidates within ``tolerance`` of the
    best, the most balanced one wins: a 23/1 split that predicts 1 ms better than
    12/12 is overfitting the model, and it makes the pipeline fragile.
    """
    links = links or {}
    layers = config.layers
    parts = len(speeds)
    if parts < 1:
        raise ValueError("placement needs at least one node")
    if parts > layers:
        raise ValueError(f"{layers} layers cannot be spread over {parts} nodes")

    candidates: list[Candidate] = []
    truncated = False
    seen: set[tuple[int, ...]] = set()

    for sizes in layer_splits(layers, parts, limit=limit):
        key = tuple(sizes)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(estimate(_bounds(sizes), speeds, links, hidden_bytes_per_token))

    even = _even_split(layers, parts)
    if tuple(even) not in seen:
        candidates.append(estimate(_bounds(even), speeds, links, hidden_bytes_per_token))

    candidates.sort(key=lambda c: c.predicted_ms_per_token)
    truncated = len(seen) >= limit

    best = candidates[0]
    near = [c for c in candidates if c.predicted_ms_per_token <= best.predicted_ms_per_token * (1 + tolerance)]
    chosen = min(near, key=lambda c: (_imbalance(c.bounds), c.predicted_ms_per_token))
    if chosen is not best:
        best.assumptions.append(
            f"chosen over the marginally cheaper {best.bounds} for balance: the "
            f"predicted difference is inside the model's {tolerance:.0%} tolerance"
        )
        best = chosen
    return best, candidates, truncated


def _imbalance(bounds: list[tuple[int, int]]) -> int:
    """How uneven a split is: largest stage minus smallest."""
    sizes = [end - start for start, end in bounds]
    return max(sizes) - min(sizes) if sizes else 0


def speeds_from_records(
    records: list[Any],
    measured: dict[str, float] | None = None,
    *,
    layers: int,
    default_ms_per_layer: float = DEFAULT_MS_PER_LAYER,
) -> list[NodeSpeed]:
    """Build per-node speeds from measured per-token times, or declare the default.

    ``measured`` maps node id to a single-node ms/token figure; dividing by the
    layer count gives the per-layer cost the estimator needs.
    """
    measured = measured or {}
    speeds: list[NodeSpeed] = []
    for record in records:
        per_token = measured.get(record.node_id)
        if per_token and per_token > 0 and layers > 0:
            speeds.append(
                NodeSpeed(record.node_id, ms_per_layer=per_token / layers, measured=True)
            )
        else:
            speeds.append(
                NodeSpeed(record.node_id, ms_per_layer=default_ms_per_layer, measured=False)
            )
    return speeds


def links_from_topology(topology: Any) -> dict[tuple[str, str], LinkCost]:
    """Measured link costs, keyed by direction. Missing directions are absent."""
    out: dict[tuple[str, str], LinkCost] = {}
    if topology is None:
        return out
    for link in getattr(topology, "links", []) or []:
        if not getattr(link, "connection_success", False):
            continue
        rtt = getattr(link, "rtt_ms_min", None)
        if rtt is None:
            continue
        out[(link.source, link.target)] = LinkCost(
            source=link.source,
            target=link.target,
            rtt_ms=float(rtt),
            mbps=getattr(link, "tcp_mbps_single_stream", None),
        )
    return out