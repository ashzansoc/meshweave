"""Node-to-node network measurement, orchestrated through the work queue.

A measurement runs in stages, because nothing dials into a node:

  1. the coordinator queues ``run_echo`` for each node, telling it which published
     port to listen on
  2. each agent binds that port and reports ``echo_ready``
  3. once every node is listening, the coordinator queues ``probe_peer`` on each
     node for every other node, so both directions are measured
  4. results are collected, written as a topology matrix, and stored as history

The state machine is event-driven: each report from an agent advances it. There
is no polling loop and no timer, so a slow node simply takes longer.

The address a peer must dial is computed **here** from the node's own reported
inventory, not taken from the agent's reply, so a stale or mistaken agent cannot
redirect a probe.

What comes back is a TCP measurement between two specific machines. It is not an
ICMP ping and it is not packet loss; ``failures`` counts round trips that timed
out, which is the honest thing TCP can tell you.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from meshweave.core import LinkMeasurement, LinkKind, TopologyMatrix

# Ports a probe prefers to listen on. Stages prefer 10100/10200, so the probe
# takes the next pair to avoid contending with a running deployment.
# A probe listens while a deployment may also be starting, and the stage runner
# prefers 10100. Taking the other half of the pair avoids the race without
# needing to know whether a stage is up.
PREFERRED_PROBE_PORTS = (10200, 10201, 10101)
# Ports a deployment uses. A probe may take them when no stage is running; they
# are tried before the arbitrary low ports, which container images tend to occupy.
STAGE_PORTS = (10100, 10200)


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class NetworkMeasurement:
    measurement_id: str
    node_ids: list[str]
    phase: str = "starting_echo"
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    finished_at: str | None = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    echo: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def expected_pairs(self) -> list[tuple[str, str]]:
        """Both directions between every pair.

        A link is not symmetric, and the two directions are measured separately
        rather than assumed to match.
        """
        return [(a, b) for a in self.node_ids for b in self.node_ids if a != b]

    @property
    def complete(self) -> bool:
        return all(f"{a}->{b}" in self.results for a, b in self.expected_pairs())


def candidate_probe_ports(inventory: Any) -> list[int]:
    """Published ports a probe may listen on, best first.

    Only real TCP ports count: some providers report entries whose internal value
    is not a port at all (Vast published 72299), and binding that raises
    OverflowError on the node. The order matters because a container often has
    its own services on the low ports, so the agent tries each in turn and reports
    which one it got.
    """
    network = getattr(inventory, "network", None)
    if network is None:
        return []

    mapped = {}
    for key, value in (network.mapped_ports or {}).items():
        try:
            internal, external = int(key), int(value)
        except (TypeError, ValueError):
            continue
        if internal == 22 or not (1 <= internal <= 65535 and 1 <= external <= 65535):
            continue
        mapped[internal] = external

    if not mapped:
        return []

    ordered: list[int] = []
    for preferred in (*PREFERRED_PROBE_PORTS, *STAGE_PORTS):
        if preferred in mapped and preferred not in ordered:
            ordered.append(preferred)
    ordered.extend(sorted(port for port in mapped if port not in ordered))
    return ordered


class NetworkProbeManager:
    """Runs measurements and holds their command queues."""

    def __init__(self, resolve_node: Callable[[str], Any], limit: int = 20) -> None:
        self._resolve = resolve_node
        self._items: dict[str, NetworkMeasurement] = {}
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._order: list[str] = []
        self._limit = limit

    # ------------------------------------------------------------------ read

    def all(self) -> list[NetworkMeasurement]:
        return [self._items[i] for i in reversed(self._order) if i in self._items]

    def get(self, measurement_id: str) -> NetworkMeasurement | None:
        return self._items.get(measurement_id)

    # ----------------------------------------------------------------- start

    def start(self, node_ids: list[str], params: dict[str, Any] | None = None) -> NetworkMeasurement:
        unique = list(dict.fromkeys(node_ids))
        if len(unique) < 2:
            raise ValueError("a link measurement needs at least two distinct nodes")

        params = params or {}
        measurement = NetworkMeasurement(
            measurement_id=f"net-{secrets.token_hex(5)}",
            node_ids=unique,
            params={
                "rtt_count": int(params.get("rtt_count", 20)),
                "bulk_bytes": int(params.get("bulk_bytes", 4 * 1024 * 1024)),
                "ttl_s": int(params.get("ttl_s", 180)),
            },
        )

        for node_id in unique:
            record = self._resolve(node_id)
            if record is None or record.inventory is None:
                return self._fail(measurement, f"{node_id} is not registered")
            ports = candidate_probe_ports(record.inventory)
            if not ports:
                return self._fail(
                    measurement,
                    f"{node_id} publishes no TCP port other than SSH, so another node "
                    "has no address to dial",
                )
            measurement.echo[node_id] = {
                "ports": ports,
                "port": ports[0],
                "address": self._address_for(record, ports[0]),
                "ready": False,
            }
            self._pending.setdefault(node_id, []).append(
                {
                    "kind": "run_echo",
                    "measurement_id": measurement.measurement_id,
                    # "port" is the first candidate, for agents that predate the
                    # multi-candidate form; "ports" lets a newer agent fall back.
                    "port": ports[0],
                    "ports": ports,
                    "ttl_s": measurement.params["ttl_s"],
                }
            )

        self._remember(measurement)
        return measurement

    def _fail(self, measurement: NetworkMeasurement, reason: str) -> NetworkMeasurement:
        measurement.phase = "failed"
        measurement.error = reason
        measurement.finished_at = utcnow().isoformat()
        self._remember(measurement)
        return measurement

    def _remember(self, measurement: NetworkMeasurement) -> None:
        self._items[measurement.measurement_id] = measurement
        self._order.append(measurement.measurement_id)
        while len(self._order) > self._limit:
            dropped = self._order.pop(0)
            self._items.pop(dropped, None)

    # -------------------------------------------------------------- dispatch

    def commands_for(self, node_id: str) -> list[dict[str, Any]]:
        return self._pending.pop(node_id, [])

    # ---------------------------------------------------------------- report

    def record(
        self, node_id: str, measurement_id: str, phase: str, payload: dict[str, Any]
    ) -> NetworkMeasurement | None:
        measurement = self._items.get(measurement_id)
        if measurement is None:
            return None

        if phase == "echo_ready":
            entry = measurement.echo.get(node_id)
            if entry is not None:
                entry["ready"] = True
                # The agent reports the port it actually bound and the address a
                # peer must dial. The coordinator cannot know which candidate won.
                if payload.get("port"):
                    entry["port"] = int(payload["port"])
                    if not entry.get("address"):
                        # An older agent reports only the port. Resolve the address
                        # from the inventory it heartbeated.
                        record = self._resolve(node_id)
                        if record is not None:
                            entry["address"] = self._address_for(record, int(payload["port"]))
                if payload.get("address"):
                    entry["address"] = payload["address"]
            if all(info.get("ready") for info in measurement.echo.values()):
                self._start_probes(measurement)
            return measurement

        if phase == "probe_result":
            target = payload.get("target")
            if target:
                result = dict(payload.get("result") or {})
                result["source"] = node_id
                result["target"] = target
                measurement.results[f"{node_id}->{target}"] = result
            if measurement.complete and measurement.phase != "done":
                measurement.phase = "done"
                measurement.finished_at = utcnow().isoformat()
            return measurement

        if phase == "error":
            measurement.phase = "failed"
            measurement.error = f"{node_id}: {payload.get('error')}"
            measurement.finished_at = utcnow().isoformat()
            return measurement

        return measurement

    @staticmethod
    def _address_for(record: Any, internal_port: int) -> str | None:
        """Public address for a port this node listens on, from its inventory."""
        network = getattr(record.inventory, "network", None)
        if network is None:
            return None
        host = network.external_ip or network.private_ip
        external = (network.mapped_ports or {}).get(internal_port)
        if not host or external is None:
            return None
        return f"{host}:{external}"

    def _start_probes(self, measurement: NetworkMeasurement) -> None:
        measurement.phase = "probing"
        for source, target in measurement.expected_pairs():
            address = measurement.echo.get(target, {}).get("address")
            if not address:
                # The peer never reported a reachable address; report it as a
                # missing direction rather than probing nothing silently.
                measurement.results[f"{source}->{target}"] = {
                    "ok": False,
                    "error": "peer reported no reachable address",
                    "source": source,
                    "target": target,
                }
                continue
            host, _, port = str(address).rpartition(":")
            self._pending.setdefault(source, []).append(
                {
                    "kind": "probe_peer",
                    "measurement_id": measurement.measurement_id,
                    "target": target,
                    "host": host,
                    "port": int(port) if port.isdigit() else 0,
                    "rtt_count": measurement.params["rtt_count"],
                    "bulk_bytes": measurement.params["bulk_bytes"],
                    "timeout_s": 25.0,
                }
            )

    # ------------------------------------------------------------- topology

    def topology(self, measurement: NetworkMeasurement) -> TopologyMatrix | None:
        """Turn a finished measurement into the matrix the planner consumes.

        Built only from measurements that returned samples. A direction that
        failed appears with ``connection_success=False`` and its error, not with
        a zero that would read as a fast link.
        """
        if measurement.phase != "done":
            return None

        links: list[LinkMeasurement] = []
        for source, target in measurement.expected_pairs():
            result = measurement.results.get(f"{source}->{target}")
            if result is None:
                continue
            samples = result.get("rtt_count") or 0
            failures = result.get("failures") or 0
            links.append(
                LinkMeasurement(
                    source=source,
                    target=target,
                    rtt_ms_min=result.get("rtt_min_ms"),
                    one_way_ms_estimate=(
                        result["rtt_p50_ms"] / 2.0
                        if isinstance(result.get("rtt_p50_ms"), (int, float))
                        else None
                    ),
                    jitter_ms=result.get("jitter_ms"),
                    tcp_mbps_single_stream=result.get("download_mbps"),
                    sustained_mbps=result.get("download_mbps"),
                    transfer_time_s=None,
                    connection_success=bool(result.get("ok")),
                    connection_stability_pct=(
                        round(100.0 * samples / max(1, samples + failures), 2)
                        if samples
                        else None
                    ),
                    udp_supported=None,
                    link_kind=LinkKind.INTERNET,
                    error=result.get("error"),
                )
            )

        if not links:
            return None
        return TopologyMatrix(nodes=list(measurement.node_ids), links=links)
