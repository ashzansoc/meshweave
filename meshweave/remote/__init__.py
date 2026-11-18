"""Remote host access and hardware discovery."""

from __future__ import annotations

from pathlib import Path

from meshweave.remote.hosts import (
    CLAIMED_SPECS,
    HOSTS,
    HOSTS_ENV,
    by_id,
    hosts_from_env,
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

# Directory for experiment evidence. Kept here rather than in hosts.py, which is
# now only about resolving operator-supplied endpoints.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NETWORK_DIR = REPO_ROOT / "experiments" / "network"
RAW_DIR = NETWORK_DIR / "raw"

__all__ = [
    "CLAIMED_SPECS",
    "HOSTS",
    "HOSTS_ENV",
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
    "hosts_from_env",
    "terminate_experiment",
]
