"""Experiment process ownership tracking.

This module exists because of an incident: a cleanup routine killed ``sshd`` on
three live hosts by extracting every PID from ``ss`` output. The root cause was
acting on a *pattern* rather than a *record*.

The rule this module enforces:

    MESHWEAVE may terminate only a process it started, identified by the exact
    PID it recorded at start time, and tagged with the experiment that started
    it.

If ownership cannot be established, termination is refused. There is no fallback
that kills by name, by port, or by process-tree walk.

Two things this module deliberately does NOT do:

  * It does not kill by port. A port is not an identity; anything may hold it,
    including sshd, a database, or another tenant's service.
  * It does not kill by name pattern. ``pkill -f <pattern>`` matches the command
    line of the shell running the cleanup itself, which is how the first outage
    in this project happened.
"""

from __future__ import annotations

import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field

from meshweave.errors import MeshweaveError


class UnsafeTerminationError(MeshweaveError):
    """Refusal to terminate a process whose ownership cannot be established."""


@dataclass(frozen=True)
class OwnedProcess:
    """A process this run started, recorded at the moment of creation."""

    pid: int
    node_id: str
    experiment_id: str
    command: str
    started_at: float
    is_local: bool = True  # local client process; the remote side is its child
    token: str = field(default_factory=lambda: uuid.uuid4().hex)


class ProcessRegistry:
    """Records which processes an experiment started.

    Only PIDs in this registry can be terminated by MESHWEAVE. A PID that is not
    here is treated as belonging to someone else, whatever it appears to be.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[int, OwnedProcess] = {}

    def record(self, pid: int, node_id: str, experiment_id: str, command: str) -> OwnedProcess:
        """Record a process this run just started."""
        if pid <= 0:
            raise UnsafeTerminationError(f"refusing to record invalid pid {pid}")
        entry = OwnedProcess(
            pid=pid,
            node_id=node_id,
            experiment_id=experiment_id,
            command=command[:200],
            started_at=time.monotonic(),
        )
        with self._lock:
            self._entries[pid] = entry
        return entry

    def owns(self, pid: int) -> bool:
        with self._lock:
            return pid in self._entries

    def get(self, pid: int) -> OwnedProcess | None:
        with self._lock:
            return self._entries.get(pid)

    def pids_for(self, experiment_id: str) -> list[int]:
        with self._lock:
            return [pid for pid, e in self._entries.items() if e.experiment_id == experiment_id]

    def terminate(
        self,
        pid: int,
        *,
        experiment_id: str | None = None,
        sig: int = signal.SIGTERM,
    ) -> bool:
        """Terminate one recorded process.

        Refuses when the PID was not recorded by this registry, when it was
        recorded by a different experiment, or when the process no longer exists.

        Returns True when a signal was delivered.
        """
        with self._lock:
            entry = self._entries.get(pid)

        if entry is None:
            raise UnsafeTerminationError(
                f"refusing to terminate pid {pid}: it is not recorded as started by "
                "MESHWEAVE. Kill only what you started; never kill by port or name."
            )

        if experiment_id is not None and entry.experiment_id != experiment_id:
            raise UnsafeTerminationError(
                f"refusing to terminate pid {pid}: it belongs to experiment "
                f"{entry.experiment_id}, not {experiment_id}"
            )

        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            # Already gone; nothing to terminate.
            with self._lock:
                self._entries.pop(pid, None)
            return False
        except PermissionError as exc:
            raise UnsafeTerminationError(
                f"not permitted to signal pid {pid}: {exc}"
            ) from exc

        return True

    def forget(self, pid: int) -> None:
        """Drop a record without signalling, for a process that exited on its own."""
        with self._lock:
            self._entries.pop(pid, None)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "pid": e.pid,
                    "node_id": e.node_id,
                    "experiment_id": e.experiment_id,
                    "command": e.command,
                    "is_local": e.is_local,
                }
                for e in self._entries.values()
            ]


# Process-wide registry. Experiments are separate processes, so this is scoped to
# one run, which is exactly the ownership boundary we want.
REGISTRY = ProcessRegistry()


def terminate_experiment(experiment_id: str) -> dict[str, object]:
    """Terminate every process this experiment started, and nothing else.

    Each PID is checked against the registry individually. If any termination
    fails, the rest still proceed and the failure is reported -- a stuck process
    must not block cleanup of the others.
    """
    terminated: list[int] = []
    failed: list[dict[str, object]] = []

    for pid in REGISTRY.pids_for(experiment_id):
        try:
            if REGISTRY.terminate(pid, experiment_id=experiment_id):
                terminated.append(pid)
        except UnsafeTerminationError as exc:
            failed.append({"pid": pid, "error": str(exc)})

    return {"terminated": terminated, "failed": failed}
