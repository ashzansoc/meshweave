"""Co-location assessment from evidence.

A shared public IP is *suggestive*, not conclusive. On Vast it can mean:

  * two containers on the same physical machine,
  * two instances behind the same NAT gateway,
  * two instances sharing an external IP through port mapping.

This module turns measurements into a graded verdict and refuses to upgrade the
verdict beyond what the evidence supports. The verdict scale is deliberately
three-valued, not binary, because "we cannot tell" is a real answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["same-host confirmed", "same-host suspected", "undetermined", "different-host"]


@dataclass
class HardwareFingerprint:
    """Identifying facts about one host's physical machine.

    Fields are optional because many container runtimes hide them. A missing
    field is ``None``, never a guess.

    Weighting notes, established against real Vast hosts:

    * ``board_serial`` / ``board_name`` / ``sys_vendor`` identify the physical
      motherboard and are the strongest available signal.
    * ``boot_id`` is the host kernel's boot identifier. Containers share the host
      kernel, so two containers on one machine *must* report the same boot_id.
      A difference is therefore strong evidence of different machines.
    * ``machine_id`` is **not** evidence of co-location. Containers inherit
      ``/etc/machine-id`` from the image, so two containers on different physical
      hosts commonly share it. Real Vast hosts A and C share a machine-id while
      running on different motherboards.
    """

    system_product_name: str | None = None
    system_vendor: str | None = None
    system_uuid: str | None = None
    board_vendor: str | None = None
    board_name: str | None = None
    board_serial: str | None = None
    cpu_model: str | None = None
    cpu_cores: int | None = None
    kernel_version: str | None = None
    gpu_uuids: list[str] = field(default_factory=list)
    pci_bridge_ids: list[str] = field(default_factory=list)
    machine_id: str | None = None
    boot_id: str | None = None


@dataclass
class Evidence:
    """One observation bearing on co-location."""

    name: str
    supports_same_host: bool
    detail: str
    weight: int  # 1 = weak signal, 3 = strong signal


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def compare_fingerprints(a: HardwareFingerprint, b: HardwareFingerprint) -> list[Evidence]:
    """Compare two fingerprints and return the evidence found.

    Only fields present on *both* hosts produce evidence. Comparing a field that
    only one host reported would manufacture a difference that is really missing
    data.
    """
    evidence: list[Evidence] = []

    def _both(x: str | None, y: str | None) -> tuple[str, str] | None:
        nx, ny = _norm(x), _norm(y)
        if nx is None or ny is None:
            return None
        return nx, ny

    # Strong: a DMI UUID is intended to be unique per physical machine.
    pair = _both(a.system_uuid, b.system_uuid)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="system_uuid",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]}",
                weight=3,
            )
        )

    # machine_id is deliberately NOT evidence. Containers inherit it from the
    # image, so two containers on different physical hosts routinely share it.
    # Real Vast hosts A and C share a machine-id while running on different
    # motherboards; treating a match as "same host" produced a wrong verdict.
    pair = _both(a.machine_id, b.machine_id)
    if pair:
        evidence.append(
            Evidence(
                name="machine_id",
                supports_same_host=False,
                detail="inherited from the container image; uninformative",
                weight=0,
            )
        )

    # Containers share the host kernel, so boot_id must match for two containers
    # on one machine. A difference is strong evidence of different machines.
    pair = _both(a.boot_id, b.boot_id)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="boot_id",
                supports_same_host=same,
                detail="same kernel boot session" if same else "different kernel boot sessions",
                weight=3,
            )
        )

    pair = _both(a.board_serial, b.board_serial)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="board_serial",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]}",
                weight=3,
            )
        )

    pair = _both(a.board_name, b.board_name)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="board_name",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]}",
                weight=3,
            )
        )

    pair = _both(a.system_vendor, b.system_vendor)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="system_vendor",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]}",
                weight=2,
            )
        )

    pair = _both(a.system_product_name, b.system_product_name)
    if pair:
        same = pair[0] == pair[1]
        # Many hosts in a fleet share a product name, so this is weak either way.
        evidence.append(
            Evidence(
                name="system_product_name",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]} (shared across many machines)",
                weight=1,
            )
        )

    pair = _both(a.cpu_model, b.cpu_model)
    if pair:
        same = pair[0] == pair[1]
        evidence.append(
            Evidence(
                name="cpu_model",
                supports_same_host=same,
                detail=f"{pair[0]} vs {pair[1]}",
                weight=1,
            )
        )

    # GPU UUIDs are globally unique per device, so a shared UUID is impossible
    # and a different one is uninformative. Reported for completeness only.
    if a.gpu_uuids and b.gpu_uuids:
        overlap = set(a.gpu_uuids) & set(b.gpu_uuids)
        evidence.append(
            Evidence(
                name="gpu_uuids",
                supports_same_host=False,
                detail=(
                    f"overlap {sorted(overlap)}"
                    if overlap
                    else f"no shared UUIDs ({a.gpu_uuids[0][:12]}… vs {b.gpu_uuids[0][:12]}…)"
                ),
                weight=0,  # uninformative for co-location
            )
        )

    return evidence


def assess(
    evidence: list[Evidence],
    *,
    measured_rtt_ms: float | None = None,
    measured_throughput_mbps: float | None = None,
) -> tuple[Verdict, list[str]]:
    """Turn evidence into a verdict.

    The bar for "confirmed" is deliberately high: it requires a *contradictory*
    signal to be absent AND strong evidence in favour. Anything less stays at
    "suspected" or "undetermined".

    Returns ``(verdict, reasons)`` where reasons are plain-language justifications.
    """
    reasons: list[str] = []

    if not evidence:
        reasons.append("no comparable hardware evidence was collected")
        if measured_rtt_ms is None:
            return "undetermined", reasons
    else:
        for item in evidence:
            if item.weight == 0:
                continue
            direction = "supports same host" if item.supports_same_host else "suggests different hosts"
            reasons.append(f"{item.name}: {direction} ({item.detail})")

    strong_for = [e for e in evidence if e.supports_same_host and e.weight >= 3]
    strong_against = [e for e in evidence if not e.supports_same_host and e.weight >= 3]

    if strong_against:
        reasons.append(
            "at least one strong identifier differs, which is inconsistent with co-location"
        )
        return "different-host", reasons

    if measured_throughput_mbps is not None:
        # Bandwidth is recorded but never decisive: a shared host gives a fast
        # link, and so does any two machines in the same datacenter. It is
        # context for the human reading the report, not a verdict input.
        reasons.append(
            f"measured throughput {measured_throughput_mbps:.0f} Mbps "
            "(context only; fast links do not imply co-location)"
        )

    if measured_rtt_ms is not None:
        # Two containers on one host typically show well under 1 ms over the
        # docker bridge. Anything above a few ms indicates a real network hop.
        if measured_rtt_ms < 1.0:
            reasons.append(
                f"measured RTT {measured_rtt_ms:.3f} ms is consistent with a local bridge"
            )
        elif measured_rtt_ms < 5.0:
            reasons.append(
                f"measured RTT {measured_rtt_ms:.2f} ms is low but not conclusive"
            )
        else:
            reasons.append(
                f"measured RTT {measured_rtt_ms:.2f} ms indicates a real network path "
                "rather than a shared host"
            )
            return "different-host", reasons

    if strong_for:
        if measured_rtt_ms is not None and measured_rtt_ms < 5.0:
            return "same-host confirmed", reasons
        return "same-host suspected", reasons

    reasons.append("hardware identifiers are hidden or identical across the fleet")
    return "undetermined", reasons
