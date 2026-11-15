"""Remote host access and hardware discovery."""

from __future__ import annotations

from meshweave.remote.hosts import (
    CLAIMED_SPECS,
    HOST_A,
    HOST_B,
    HOST_C,
    HOSTS,
    NETWORK_DIR,
    RAW_DIR,
    by_id,
    shared_ip_groups,
)
from meshweave.remote.process_registry import (
    REGISTRY,
    OwnedProcess,
    ProcessRegistry,
    UnsafeTerminationError,
    terminate_experiment,
)
from meshweave.remote.ssh import (
    Host,
    RemoteError,
    Result,
    alias_for,
    check_auth,
    ensure_ssh_config,
    run,
    run_many,
)

__all__ = [
    "CLAIMED_SPECS",
    "HOSTS",
    "HOST_A",
    "HOST_B",
    "HOST_C",
    "OwnedProcess",
    "ProcessRegistry",
    "REGISTRY",
    "UnsafeTerminationError",
    "NETWORK_DIR",
    "RAW_DIR",
    "Host",
    "RemoteError",
    "Result",
    "alias_for",
    "by_id",
    "check_auth",
    "ensure_ssh_config",
    "run",
    "run_many",
    "shared_ip_groups",
    "terminate_experiment",
]
