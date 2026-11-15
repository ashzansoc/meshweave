"""Agent package.

``telemetry`` is the real NVML path. ``synthetic`` is test-only and is never
imported by the runtime or by experiment scripts.
"""

from __future__ import annotations

__all__ = ["telemetry"]
