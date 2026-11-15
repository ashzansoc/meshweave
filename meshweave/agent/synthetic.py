"""Synthetic GPU telemetry for testing the *control plane* without hardware.

This module exists so that the coordinator, registry, planner, wire protocol and
report validator can be exercised end to end on a laptop. It is deliberately a
separate module that imports nothing from ``telemetry.py``.

Hard rules:
  * ``SyntheticGpuInventory`` stamps every inventory with ``synthetic: true``.
  * Experiment scripts import the *real* agent. They never import this module.
    ``tests/test_experiment_guard.py`` enforces that with an AST scan.
  * No value produced here may be written into a report without a synthetic flag
    set on the report itself, and ``report.py`` refuses to validate a synthetic
    report as evidence.
"""

from __future__ import annotations

from typing import Any

from meshweave.core import (
    DeviceKind,
    GpuDevice,
    GpuInventory,
    HostResources,
    NetworkInfo,
    utcnow,
)

_SYNTHETIC_MARKER = "synthetic"

_SYNTHETIC_GPUS: list[dict[str, Any]] = [
    {
        "name": "RTX 4090",
        "arch": "Ada Lovelace",
        "cc": (8, 9),
        "vram_gib": 24.0,
        "bw": 1008.0,
    },
    {
        "name": "RTX 3090",
        "arch": "Ampere",
        "cc": (8, 6),
        "vram_gib": 24.0,
        "bw": 936.0,
    },
    {
        "name": "RTX 4080",
        "arch": "Ada Lovelace",
        "cc": (8, 9),
        "vram_gib": 16.0,
        "bw": 717.0,
    },
    {
        "name": "A5000",
        "arch": "Ampere",
        "cc": (8, 6),
        "vram_gib": 24.0,
        "bw": 768.0,
    },
]


def synthetic_inventory(
    spec: str = "RTX 4090",
    vram_free_gib: float | None = None,
    agent_version: str = "0.0.0+synthetic",
) -> GpuInventory:
    """Build one synthetic GPU inventory.

    Args:
        spec: GPU name from the table above.
        vram_free_gib: override free VRAM, else derived from a plausible usage.
        agent_version: stamped so the synthetic origin is traceable in the DB.
    """
    match = next((g for g in _SYNTHETIC_GPUS if g["name"].lower() == spec.lower()), None)
    if match is None:
        names = ", ".join(str(g["name"]) for g in _SYNTHETIC_GPUS)
        raise ValueError(f"unknown synthetic GPU {spec!r}; known: {names}")

    total_bytes = int(float(match["vram_gib"]) * (1024**3))
    if vram_free_gib is None:
        vram_free_gib = float(match["vram_gib"]) * 0.9
    free_bytes = int(vram_free_gib * (1024**3))

    gpu = GpuDevice(
        index=0,
        uuid=f"{_SYNTHETIC_MARKER}-{str(match['name']).replace(' ', '-').lower()}",
        name=str(match["name"]),
        architecture=str(match["arch"]),
        compute_capability=(int(match["cc"][0]), int(match["cc"][1])),
        cuda_version="12.4",
        driver_version="0.0.0+synthetic",
        vram_total_bytes=total_bytes,
        vram_free_bytes=free_bytes,
        vram_used_bytes=total_bytes - free_bytes,
        temperature_c=54.0,
        power_draw_w=210.0,
        power_limit_w=450.0,
        utilization_gpu_pct=0.0,
        utilization_memory_pct=round((1 - vram_free_gib / float(match["vram_gib"])) * 100, 2),
        memory_bandwidth_gbps=float(match["bw"]),
        kind=DeviceKind.DISCRETE_GPU,
    )
    return GpuInventory(
        gpus=[gpu],
        host=HostResources(
            cpu_model="AMD EPYC 7763 (synthetic)",
            cpu_cores=16,
            ram_total_bytes=64 * (1024**3),
            ram_available_bytes=48 * (1024**3),
            disk_total_bytes=500 * (1024**3),
            disk_available_bytes=400 * (1024**3),
            os_name="Linux",
            kernel_version="synthetic",
        ),
        network=NetworkInfo(
            hostname=f"synthetic-{str(match['name']).replace(' ', '').lower()}",
            private_ip="10.0.0.1",
            external_ip="203.0.113.10",
            nat=True,
            mapped_ports={8000: 34567},
        ),
        agent_version=agent_version,
        collected_at=utcnow(),
    )
