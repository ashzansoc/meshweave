"""GPU discovery in the shipped agent, against a fake NVML driver.

These tests exist because two bugs reached real hardware:

* the per-device helper was defined *inside* the device loop, so the system-wide
  driver read raised NameError and every node reported "no GPUs"
* an earlier closure captured the loop variable, so a multi-GPU host reported the
  last card's counters for every GPU

Neither is catchable on a machine without an NVIDIA driver, which is most
development machines. A fake driver makes both testable anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

AGENT_PY = Path(__file__).resolve().parent.parent / "deploy" / "agent.py"


class _Memory:
    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free
        self.used = total - free


class _Util:
    def __init__(self, gpu: int, memory: int) -> None:
        self.gpu = gpu
        self.memory = memory


def _fake_pynvml(devices: int = 2, with_driver: bool = True) -> types.ModuleType:
    module = types.ModuleType("pynvml")
    module.nvmlInit = lambda: None
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetCount = lambda: devices
    module.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"
    module.nvmlDeviceGetMemoryInfo = lambda h: _Memory(8 * 1024**3, 6 * 1024**3)
    module.nvmlDeviceGetName = lambda h: f"Fake GPU {h[-1]}"
    module.nvmlDeviceGetUUID = lambda h: f"GPU-1234-{h[-1]}"
    # Distinct per device: a closure bug would repeat one card's values.
    module.nvmlDeviceGetUtilizationRates = lambda h: _Util(int(h[-1]) * 10, 5)
    module.nvmlDeviceGetPowerUsage = lambda h: 100000 + int(h[-1])
    module.nvmlDeviceGetTemperature = lambda h: 40 + int(h[-1])
    module.nvmlDeviceGetCudaComputeCapability = lambda h: (7, 5)

    if with_driver:
        module.nvmlSystemGetDriverVersion = lambda: "580.178.04"
    else:
        def _missing():
            raise RuntimeError("NVML: driver version unavailable")

        module.nvmlSystemGetDriverVersion = _missing
    return module


def _load_agent():
    spec = importlib.util.spec_from_file_location("mw_agent_gpu", AGENT_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_fake_pynvml(fake, fn):
    saved = sys.modules.get("pynvml")
    sys.modules["pynvml"] = fake
    try:
        return fn()
    finally:
        if saved is None:
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = saved


def test_gpus_are_discovered_with_the_driver_version() -> None:
    agent = _load_agent()
    gpus = _with_fake_pynvml(_fake_pynvml(2), agent.gpu_inventory)

    assert len(gpus) == 2
    assert gpus[0]["name"] == "Fake GPU 0"
    assert gpus[1]["name"] == "Fake GPU 1"
    assert gpus[0]["driver_version"] == "580.178.04"
    assert gpus[0]["compute_capability"] == [7, 5]
    assert gpus[0]["vram_total_bytes"] == 8 * 1024**3


def test_each_gpu_reports_its_own_counters() -> None:
    """The closure bug made every entry report the last device."""
    agent = _load_agent()
    gpus = _with_fake_pynvml(_fake_pynvml(3), agent.gpu_inventory)

    utils = [g["utilization_gpu_pct"] for g in gpus]
    assert utils == [0.0, 10.0, 20.0]


def test_a_missing_driver_version_does_not_hide_the_gpus() -> None:
    """The system-wide read must not take the whole inventory down with it."""
    agent = _load_agent()
    gpus = _with_fake_pynvml(_fake_pynvml(1, with_driver=False), agent.gpu_inventory)

    assert len(gpus) == 1
    assert gpus[0]["driver_version"] is None
    assert gpus[0]["name"] == "Fake GPU 0"


def test_the_driver_read_happens_outside_the_device_loop() -> None:
    """Structural guard: no per-device helper may be defined inside the loop."""
    source = AGENT_PY.read_text()
    body = source[source.index("def gpu_inventory") :]
    assert "def opt(" not in body
    assert "_nvml_safe(pynvml.nvmlSystemGetDriverVersion)" in body

def test_the_recorded_log_path_is_the_one_the_process_writes() -> None:
    """Writing to one filename and reading another made every stage look stuck
    in 'loading': the agent read a file that did not exist."""
    source = AGENT_PY.read_text()
    assignments = [line for line in source.splitlines() if "log_path = LOG_DIR" in line]
    assert len(assignments) == 1, assignments
    assert "replica" in assignments[0]
