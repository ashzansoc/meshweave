"""Deployment planning and dispatch.

A deployment is a model plus an ordered list of nodes, each of which will run one
stage holding a contiguous range of decoder layers. This module decides whether
that is physically possible before anything is started:

  * the architecture must be one the runtime implements,
  * each node must have a free GPU and enough VRAM for its layers,
  * each node must have disk for the whole checkpoint (the shards are not split
    finely enough to fetch a single layer range),
  * every node in a multi-node deployment must publish a TCP port, because the
    chain has to be reachable in both directions.

Any of those failing is a refusal with the reason attached, not a deployment
that half-starts.

Nothing here claims a model *fits* because the VRAM totals add up. The check is
per node, against that node's own free memory, and the actual per-GPU
consumption is measured by the runner and reported back.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meshweave.errors import ModelLoadError
from meshweave.runtime.model_config import (
    DenseConfig,
    dtype_bytes_for,
    partition_layers,
)

GIB = 1024**3

# The agent gained the work queue in this version. An older agent registers and
# reports telemetry but never polls, so a deployment sent to it is a no-op that
# looks like a hang.
MIN_DEPLOY_AGENT = (0, 4, 0)


def agent_can_deploy(version: str | None) -> bool:
    if not version:
        return False
    parts: list[int] = []
    for piece in version.split(".")[:3]:
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts) >= MIN_DEPLOY_AGENT


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# model identity
# --------------------------------------------------------------------------- #


def normalise_repo(value: str) -> str:
    """Accept a pasted Hugging Face URL or a bare ``org/name``."""
    text = value.strip()
    if not text:
        raise ValueError("model is required")

    if text.startswith("http://") or text.startswith("https://"):
        parts = [p for p in text.split("/") if p]
        # https://huggingface.co/<org>/<name>[/tree/main]
        try:
            host_index = next(i for i, p in enumerate(parts) if "." in p)
        except StopIteration:
            raise ValueError("could not read a host from the model URL") from None
        rest = parts[host_index + 1 :]
        if rest and rest[0] != "models":
            rest = rest
        if len(rest) < 2:
            raise ValueError(
                "expected a repository URL like https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
            )
        return f"{rest[0]}/{rest[1]}"

    pieces = text.split("/")
    if len(pieces) != 2 or not all(pieces):
        raise ValueError("expected org/name, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    return text


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


@dataclass
class StagePlan:
    node_id: str
    rank: int
    start: int
    end: int
    resident_bytes: int
    checkpoint_bytes: int
    address: str | None
    listen_port: int | None
    external_port: int | None

    @property
    def layers(self) -> str:
        return f"[{self.start},{self.end})"


@dataclass
class Deployment:
    deployment_id: str
    model: str
    created_at: str = field(default_factory=utcnow)
    status: str = "PENDING"
    message: str | None = None
    stages: list[StagePlan] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    stop_requested: bool = False
    node_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeploymentStore:
    """Deployments, kept in memory and written to disk so a restart survives."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._items: dict[str, Deployment] = {}
        if path is not None and path.exists():
            self._load()

    # ------------------------------------------------------------------ io

    def _load(self) -> None:
        assert self._path is not None
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for item in raw.get("deployments", []):
            stages = [StagePlan(**stage) for stage in item.pop("stages", [])]
            self._items[item["deployment_id"]] = Deployment(stages=stages, **item)

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {"saved_at": utcnow(), "deployments": [d.to_dict() for d in self._items.values()]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)

    # ------------------------------------------------------------- access

    def all(self) -> list[Deployment]:
        return list(self._items.values())

    def get(self, deployment_id: str) -> Deployment | None:
        return self._items.get(deployment_id)

    def add(self, deployment: Deployment) -> None:
        self._items[deployment.deployment_id] = deployment
        self._save()

    def remove(self, deployment_id: str) -> bool:
        existed = self._items.pop(deployment_id, None) is not None
        if existed:
            self._save()
        return existed

    def touch(self, deployment: Deployment) -> None:
        self._save()

    # ------------------------------------------------------------ commands

    def commands_for(self, node_id: str) -> list[dict[str, Any]]:
        """What this node should be running right now, given live deployments.

        Returns nothing for a node whose stage is already reported running; the
        agent repeats that check on its side, so a missed status report cannot
        start a second copy of a stage.
        """
        commands: list[dict[str, Any]] = []
        dirty = False
        for deployment in self._items.values():
            for stage in deployment.stages:
                if stage.node_id != node_id:
                    continue
                state = deployment.node_status.get(node_id, {})
                # A FAILED deployment is not asked to stop: the stage is already
                # gone, and the "stopped" report would overwrite the reason it
                # failed, which is the only useful thing left.
                if deployment.stop_requested or deployment.status == "STOPPED":
                    commands.append(
                        {
                            "kind": "stop_stage",
                            "deployment_id": deployment.deployment_id,
                            "reason": "deployment stopped",
                        }
                    )
                    continue
                if state.get("phase") not in ("running", "starting", "loading"):
                    commands.append(
                        {
                            "kind": "run_stage",
                            "deployment_id": deployment.deployment_id,
                            "rank": stage.rank,
                            "world": len(deployment.stages),
                            "repo": deployment.model,
                            "start": stage.start,
                            "end": stage.end,
                            "listen": (
                                f"0.0.0.0:{stage.listen_port}" if stage.listen_port else None
                            ),
                            "next": self._peer_address(deployment, stage.rank + 1),
                            "reply": (
                                self._peer_address(deployment, 0) if stage.rank else None
                            ),
                            "device": "cuda",
                            "dtype": deployment.config.get("dtype", "bfloat16"),
                            "max_seq": deployment.config.get("max_seq", 2048),
                            "stage_token": deployment.config.get("stage_token"),
                        }
                    )

                # Prompts are served by rank 0 only, through the local control
                # socket. Nothing new is exposed to the network for this.
                if stage.rank == 0:
                    for prompt in deployment.prompts:
                        if prompt.get("state") != "queued":
                            continue
                        prompt["state"] = "sent"
                        dirty = True
                        commands.append(
                            {
                                "kind": "run_prompt",
                                "deployment_id": deployment.deployment_id,
                                "prompt_id": prompt["id"],
                                "prompt": prompt["prompt"],
                                "max_new_tokens": prompt.get("max_new_tokens", 32),
                                "temperature": prompt.get("temperature", 0.0),
                                "top_p": prompt.get("top_p", 1.0),
                                "control": "/run/meshweave/control.sock",
                            }
                        )
        if dirty:
            self._save()
        return commands

    def _peer_address(self, deployment: Deployment, rank: int) -> str | None:
        for stage in deployment.stages:
            if stage.rank == rank:
                return stage.address
        return None

    def mark_unreachable(self, node_ids: list[str]) -> list[str]:
        """Fail the stages of nodes the registry has reaped.

        A stage on a node that stopped heartbeating is not running; pretending
        otherwise leaves a deployment that reports RUNNING while nothing computes.
        """
        affected: list[str] = []
        for deployment in self._items.values():
            touched = False
            for stage in deployment.stages:
                if stage.node_id not in node_ids:
                    continue
                entry = deployment.node_status.get(stage.node_id, {})
                if entry.get("phase") in ("failed", "stopped"):
                    continue
                deployment.node_status[stage.node_id] = {
                    "phase": "failed",
                    "detail": "node unreachable: no heartbeat",
                    "pid": entry.get("pid"),
                    "at": utcnow(),
                }
                touched = True
            if touched and deployment.status not in ("STOPPED", "FAILED"):
                deployment.status = "FAILED"
                deployment.message = "a node hosting this deployment stopped responding"
                affected.append(deployment.deployment_id)
        if affected:
            self._save()
        return affected

    def stage_for(self, deployment_id: str, node_id: str) -> StagePlan | None:
        deployment = self._items.get(deployment_id)
        if deployment is None:
            return None
        for stage in deployment.stages:
            if stage.node_id == node_id:
                return stage
        return None


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


class DeploymentRejected(Exception):
    """A deployment that would not physically work, with the reason."""


# Ports a stage would rather listen on. Provider images often publish a low
# port that something else in the container already owns (1111 and 8080 are
# common), so the lowest published port is a poor choice; 10100/10200 are what
# these deployments have used and are normally free.
PREFERRED_INTERNAL_PORTS = (10100, 10200, 10101, 10201)


def pick_port(node_inventory: Any) -> tuple[int, int] | None:
    """Choose a published TCP port: returns (internal, external) or None.

    The process listens on the internal port; peers dial the public address and
    the provider's mapped port. A node that publishes nothing cannot take part
    in a multi-node chain, because the next stage has nowhere to connect.
    """
    mapped = getattr(getattr(node_inventory, "network", None), "mapped_ports", {}) or {}
    candidates = [(int(k), int(v)) for k, v in mapped.items() if int(k) != 22]
    if not candidates:
        return None

    for preferred in PREFERRED_INTERNAL_PORTS:
        for pair in candidates:
            if pair[0] == preferred:
                return pair

    # Nothing preferred: take the highest numbered port. High ports are the least
    # likely to collide with a service the image already runs.
    return max(candidates, key=lambda pair: pair[0])


def plan_deployment(
    *,
    model: str,
    records: list[Any],
    config: DenseConfig,
    max_seq: int = 2048,
    vram_headroom: float = 0.85,
    disk_headroom: float = 0.90,
) -> list[StagePlan]:
    """Assign layers to nodes, or raise DeploymentRejected explaining why not."""
    if not records:
        raise DeploymentRejected("select at least one node")

    world = len(records)
    if world > config.layers:
        raise DeploymentRejected(
            f"{config.layers} layers cannot be spread over {world} nodes without an "
            "empty stage"
        )

    dtype_bytes = dtype_bytes_for(config.dtype)
    bounds = partition_layers(config, world)
    checkpoint = config.checkpoint_bytes(dtype_bytes)

    stages: list[StagePlan] = []
    for rank, (record, (start, end)) in enumerate(zip(records, bounds, strict=True)):
        inventory = record.inventory
        if inventory is None or not inventory.gpus:
            raise DeploymentRejected(
                f"{record.node_id} reports no GPU, so it cannot hold a stage"
            )

        version = inventory.agent_version
        if not agent_can_deploy(version):
            raise DeploymentRejected(
                f"{record.node_id} runs agent {version or 'unknown'}; deployments need "
                f"{'.'.join(str(p) for p in MIN_DEPLOY_AGENT)} or newer, because that is the "
                "agent that polls for work. Re-run the installer on that node."
            )

        gpus = inventory.gpus
        free = max(g.vram_free_bytes for g in gpus)
        resident = config.stage_bytes(
            start,
            end,
            owns_embed=(rank == 0),
            owns_head=(rank == world - 1),
            dtype_bytes=dtype_bytes,
        )
        kv_reserve = config.kv_bytes_per_stage_token(start, end, dtype_bytes) * max_seq
        needed = resident + kv_reserve
        if needed > free * vram_headroom:
            raise DeploymentRejected(
                f"{record.node_id} has {free / GIB:.2f} GiB free VRAM but stage "
                f"{rank} needs {needed / GIB:.2f} GiB "
                f"({resident / GIB:.2f} GiB weights + {kv_reserve / GIB:.2f} GiB KV). "
                "Add a node, or choose a smaller model."
            )

        disk_free = inventory.host.disk_available_bytes or 0
        if checkpoint > disk_free * disk_headroom:
            raise DeploymentRejected(
                f"{record.node_id} has {disk_free / GIB:.1f} GiB disk free but the "
                f"checkpoint is {checkpoint / GIB:.1f} GiB. Each node downloads the "
                "whole checkpoint even though it keeps only its own layers."
            )

        address = None
        internal = external = None
        if world > 1:
            ports = pick_port(inventory)
            if ports is None:
                raise DeploymentRejected(
                    f"{record.node_id} publishes no TCP port other than SSH, so no "
                    "other stage can connect to it. Publish a TCP port on the "
                    "instance and re-run the installer so the agent reports it."
                )
            internal, external = ports
            public_ip = inventory.network.external_ip or inventory.network.private_ip
            if not public_ip:
                raise DeploymentRejected(
                    f"{record.node_id} reports no reachable address"
                )
            address = f"{public_ip}:{external}"

        stages.append(
            StagePlan(
                node_id=record.node_id,
                rank=rank,
                start=start,
                end=end,
                resident_bytes=resident,
                checkpoint_bytes=checkpoint,
                address=address,
                listen_port=internal,
                external_port=external,
            )
        )

    return stages


def effective_dtype(config: DenseConfig, records: list[Any]) -> str:
    """Pick a dtype the chosen cards can actually execute.

    bfloat16 needs Ampere (compute 8.0). On a Turing card the planner switches to
    float16 rather than starting a stage that fails at the first kernel. Both are
    two bytes, so the memory arithmetic is unchanged -- only the label differs.
    """
    if config.dtype != "bfloat16":
        return config.dtype
    for record in records:
        inventory = record.inventory
        if inventory is None:
            continue
        for gpu in inventory.gpus:
            capability = gpu.compute_capability
            if capability and tuple(int(part) for part in capability) < (8, 0):
                return "float16"
    return config.dtype


def new_deployment(
    model: str,
    stages: list[StagePlan],
    config: DenseConfig,
    max_seq: int,
    dtype: str | None = None,
) -> Deployment:
    return Deployment(
        deployment_id=f"dep-{secrets.token_hex(6)}",
        model=model,
        status="DISPATCHED",
        stages=stages,
        config={
            "dtype": dtype or config.dtype,
            # Every stage of this deployment must present this secret in its
            # handshake, so a published port cannot be fed by a stranger.
            "stage_token": secrets.token_hex(16),
            "max_seq": max_seq,
            "architecture": config.architecture,
            "layers": config.layers,
            "hidden": config.hidden,
            "strategy": "PIPELINE",
        },
    )


def load_remote_config(text: str) -> DenseConfig:
    try:
        return DenseConfig.from_json_text(text)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        raise ModelLoadError(f"invalid model config: {exc}") from exc


class ModelConfigUnavailable(Exception):
    """config.json could not be read, with the HTTP status to report."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def fetch_model_config(repo: str) -> str:
    """Read config.json from the hub, so planning uses the model's real geometry.

    A model is never sized from its parameter count or its name.
    """
    import httpx

    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            response = await client.get(url)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        raise ModelConfigUnavailable(
            502, f"could not reach huggingface.co: {exc}"
        ) from exc

    if response.status_code == 404:
        raise ModelConfigUnavailable(
            404, f"{repo} has no config.json; check the repository name"
        )
    if response.status_code != 200:
        raise ModelConfigUnavailable(
            502, f"huggingface.co returned {response.status_code} for {repo}"
        )
    return response.text
