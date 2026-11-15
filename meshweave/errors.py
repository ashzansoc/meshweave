"""Errors.

MESHWEAVE never swallows an execution error. Every failure path raises one of
these, and every experiment report records the failure rather than dropping it.
"""

from __future__ import annotations


class MeshweaveError(Exception):
    """Base class."""


class ConfigError(MeshweaveError):
    pass


class AuthError(MeshweaveError):
    """Registration, request or workload authentication failed."""


class NodeUnavailableError(MeshweaveError):
    """A node could not be reached or has been declared failed."""


class InsufficientCapacityError(MeshweaveError):
    """No viable placement exists for the requested workload."""


class PlanningError(MeshweaveError):
    """The planner could not produce a plan, or refused to."""


class ExecutionError(MeshweaveError):
    """A workload failed during execution."""


class TransferError(MeshweaveError):
    """An activation transfer between pipeline stages failed."""


class ModelLoadError(MeshweaveError):
    """Model weights could not be loaded on a node."""


class ReportValidationError(MeshweaveError):
    """An experiment report is incomplete or dishonest (missing failure data,
    missing cost data, missing raw evidence)."""
