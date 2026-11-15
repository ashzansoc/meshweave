"""GPU discovery and telemetry via NVML.

Rules enforced here:
  * No fabricated GPUs. If NVML is unavailable the function raises; it never
    returns a plausible-looking synthetic device.
  * No fabricated counters. A counter NVML does not support is ``None``.
  * Simulation is a separate module (meshweave.agent.simulated) and is never
    imported by this one.
"""

from __future__ import annotations

import contextlib
import platform
import shutil
import socket
from dataclasses import dataclass
from typing import Any

from meshweave.core import (
    DeviceKind,
    GpuDevice,
    GpuInventory,
    HostResources,
    NetworkInfo,
)

# CUDA compute capability -> marketing architecture name.
_ARCH_BY_CC = {
    (7, 0): "Volta",
    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (10, 1): "Blackwell",
    (12, 0): "Blackwell",
}

# Nameplate memory bandwidth (GB/s) used only as a planning hint. Overridden by
# measurement where a benchmark exists.
_BW_HINT_GBPS = {
    "RTX 4090": 1008.0,
    "RTX 4080": 717.0,
    "RTX 4080 SUPER": 736.0,
    "RTX 3090": 936.0,
    "RTX 3090 Ti": 1008.0,
    "RTX 3080": 760.0,
    "RTX 3080 Ti": 912.0,
    "RTX A5000": 768.0,
    "RTX A6000": 768.0,
    "A100": 1555.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE-40GB": 1555.0,
    "H100": 3350.0,
    "H100-SXM": 3350.0,
    "H200": 4800.0,
    "L40S": 864.0,
    "L4": 300.0,
    "T4": 320.0,
}


class NvmlUnavailableError(RuntimeError):
    """NVML (and therefore the NVIDIA driver) is not present on this host."""


@dataclass
class NvmlHandle:
    """Thin ownership wrapper so the library is shut down deterministically."""

    lib: object

    def close(self) -> None:
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown must never mask real errors
            pass


def _import_pynvml() -> Any:
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - depends on host
        raise NvmlUnavailableError(
            "pynvml is not installed. Install the 'nvidia-ml-py' package on a host "
            "with an NVIDIA driver."
        ) from exc
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - pynvml raises NVMLError subclasses
        raise NvmlUnavailableError(f"NVML init failed: {exc}") from exc
    return pynvml


def _cc(major: int, minor: int) -> tuple[int, int]:
    return (int(major), int(minor))


def _read_gpu(pynvml: Any, index: int) -> GpuDevice:
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()

    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        uuid = uuid.decode()
    uuid = uuid.replace("GPU-", "")

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    try:
        cc_raw = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        cc = _cc(cc_raw[0], cc_raw[1])
    except Exception:  # noqa: BLE001
        cc = None

    def _opt(fn: Any, default: Any = None) -> Any:
        try:
            return fn(handle)
        except Exception:  # noqa: BLE001 - unsupported counter on this device
            return default

    driver = _opt(pynvml.nvmlSystemGetDriverVersion)
    cuda = _opt(pynvml.nvmlSystemGetCudaDriverVersion)
    cuda_str = f"{cuda // 1000}.{(cuda % 1000) // 10}" if isinstance(cuda, int) else None

    temp = _opt(pynvml.nvmlDeviceGetTemperature, None)
    if temp is not None:
        temp = float(temp)

    power = _opt(pynvml.nvmlDeviceGetPowerUsage)
    if power is not None:
        power = float(power) / 1000.0  # mW -> W

    power_limit = _opt(pynvml.nvmlDeviceGetPowerManagementLimit)
    if power_limit is not None:
        power_limit = float(power_limit) / 1000.0

    util = _opt(pynvml.nvmlDeviceGetUtilizationRates)
    util_gpu = float(util.gpu) if util is not None else None
    util_mem = float(util.memory) if util is not None else None

    try:
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = getattr(pci, "busId", None)
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode()
    except Exception:  # noqa: BLE001
        pci_bus_id = None

    arch = _ARCH_BY_CC.get(cc) if cc else None

    return GpuDevice(
        index=index,
        uuid=str(uuid),
        name=str(name),
        architecture=arch,
        compute_capability=cc,
        cuda_version=cuda_str,
        driver_version=str(driver) if driver else None,
        vram_total_bytes=int(mem.total),
        vram_free_bytes=int(mem.free),
        vram_used_bytes=int(mem.used),
        temperature_c=temp,
        power_draw_w=power,
        power_limit_w=power_limit,
        utilization_gpu_pct=util_gpu,
        utilization_memory_pct=util_mem,
        memory_bandwidth_gbps=_BW_HINT_GBPS.get(str(name).upper()),
        kind=DeviceKind.DISCRETE_GPU,
        pci_bus_id=pci_bus_id,
    )


def discover_gpus() -> list[GpuDevice]:
    """Return every GPU on this host, or raise if NVML is unavailable."""
    pynvml = _import_pynvml()
    try:
        count = pynvml.nvmlDeviceGetCount()
        return [_read_gpu(pynvml, i) for i in range(count)]
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def discover_host() -> HostResources:
    import os

    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:  # noqa: PTH123
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    meminfo[key.strip()] = int(parts[0]) * 1024
    except OSError:
        pass

    if meminfo:
        ram_total = meminfo.get("MemTotal")
        ram_available = meminfo.get("MemAvailable")
    else:  # macOS / BSD fallback for local development
        ram_total = None
        ram_available = None

    disk_total = disk_available = None
    try:
        usage = shutil.disk_usage("/")
        disk_total, disk_available = usage.total, usage.free
    except OSError:
        pass

    cpu_cores = os.cpu_count()

    return HostResources(
        cpu_model=platform.processor() or platform.machine(),
        cpu_cores=cpu_cores,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_total_bytes=disk_total,
        disk_available_bytes=disk_available,
        os_name=platform.system(),
        kernel_version=platform.release(),
    )


def discover_network() -> NetworkInfo:
    """Self-reported reachability.

    Vast publishes the external port map inside the container as
    ``VAST_TCP_PORT_<internal>`` and ``VAST_UDP_PORT_<internal>``. We read those
    rather than guessing an API field that may not exist.
    """
    import os

    mapped: dict[int, int] = {}
    for key, value in os.environ.items():
        if not key.startswith("VAST_TCP_PORT_"):
            continue
        try:
            internal = int(key.removeprefix("VAST_TCP_PORT_"))
            external = int(value)
        except ValueError:
            continue
        mapped[internal] = external

    return NetworkInfo(
        hostname=socket.gethostname(),
        private_ip=_private_ip(),
        external_ip=os.environ.get("PUBLIC_IPADDR") or None,
        nat=None,
        mapped_ports=mapped,
        overlay_interface=os.environ.get("MESHWEAVE_OVERLAY_IFACE") or None,
    )


def _private_ip() -> str | None:
    """Best-effort primary interface address without sending packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1, unrouted
        address: str = sock.getsockname()[0]
        return address
    except OSError:
        return None
    finally:
        sock.close()


def collect_inventory(agent_version: str) -> GpuInventory:
    """Full inventory snapshot. Raises NvmlUnavailableError if no NVIDIA host."""
    gpus = discover_gpus()
    if not gpus:
        raise NvmlUnavailableError("NVML reported zero GPUs on this host.")
    return GpuInventory(
        gpus=gpus,
        host=discover_host(),
        network=discover_network(),
        agent_version=agent_version,
    )
