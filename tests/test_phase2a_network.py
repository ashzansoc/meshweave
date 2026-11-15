"""Tests for Phase 2A network primitives.

These cover the parts that must be correct before touching a real host: address
allocation, WireGuard config generation, link classification, and co-location
verdicts. The co-location tests matter most -- they are the guard against
concluding "same machine" from a shared IP.
"""

from __future__ import annotations

import pytest
from meshweave.errors import MeshweaveError
from meshweave.network.classify import (
    LinkThresholds,
    classify,
    suitability,
)
from meshweave.network.colocation import (
    HardwareFingerprint,
    assess,
    compare_fingerprints,
)
from meshweave.network.overlay.addresses import (
    Interface,
    allocate,
    detect_conflicts,
    parse_interfaces,
)
from meshweave.network.overlay.wgconfig import (
    PeerSpec,
    build_configs,
    render_config,
    validate_peer_endpoints,
)

# --------------------------------------------------------------------------- #
# Address allocation
# --------------------------------------------------------------------------- #


def test_allocation_is_one_address_per_node():
    allocs = allocate(["a", "b", "c"])
    assert [a.address for a in allocs] == ["10.10.0.1", "10.10.0.2", "10.10.0.3"]
    assert all(a.prefix == 24 for a in allocs)


def test_conflict_with_existing_interface_is_rejected():
    existing = {"node-a": [Interface(name="eth0", address="10.10.0.5", prefix=24)]}
    with pytest.raises(MeshweaveError, match="overlaps existing interface"):
        allocate(["node-a"], existing_by_node=existing)


def test_loopback_does_not_conflict():
    existing = {"node-a": [Interface(name="lo", address="127.0.0.1", prefix=8, is_loopback=True)]}
    allocs = allocate(["node-a"], existing_by_node=existing)
    assert allocs[0].address == "10.10.0.1"


def test_detect_conflicts_names_the_interface():
    ifaces = [
        Interface(name="eth0", address="10.10.0.5", prefix=24),
        Interface(name="eth1", address="192.168.1.10", prefix=24),
    ]
    assert detect_conflicts(ifaces, "10.10.0.0/24") == ["eth0"]
    assert detect_conflicts(ifaces, "172.16.0.0/24") == []


def test_parse_interfaces_reads_ip_addr_output():
    sample = (
        "1: lo    inet 127.0.0.1/8 scope host lo\n"
        "2: eth0    inet 172.17.0.5/16 brd 172.17.255.255 scope global eth0\n"
    )
    parsed = parse_interfaces(sample)
    assert [i.name for i in parsed] == ["lo", "eth0"]
    assert parsed[1].address == "172.17.0.5"
    assert parsed[1].prefix == 16


# --------------------------------------------------------------------------- #
# WireGuard config generation
# --------------------------------------------------------------------------- #


def test_render_config_has_interface_and_one_peer_per_node():
    peers = [
        PeerSpec("host-b", "PUB_B", "83.63.1.187:51820", "10.10.0.0/24"),
        PeerSpec("host-c", "PUB_C", None, "10.10.0.0/24"),
    ]
    config = render_config(
        __import__("meshweave.network.overlay.wgconfig", fromlist=["InterfaceSpec"]).InterfaceSpec(
            node_id="host-a", private_key="PRIV_A", address="10.10.0.1"
        ),
        peers,
    )
    assert "[Interface]" in config
    assert config.count("[Peer]") == 2
    assert "AllowedIPs = 10.10.0.0/24" in config
    # A peer with no reachable endpoint is passive: no Endpoint line.
    assert "Endpoint = 83.63.1.187:51820" in config
    assert "PersistentKeepalive = 25" in config
    assert "MTU = 1280" in config


def test_build_configs_excludes_self():
    allocs = allocate(["a", "b", "c"])
    keys = {n: f"PUB_{n}" for n in "abc"}
    privs = {n: f"PRIV_{n}" for n in "abc"}
    endpoints = {"a": "1.1.1.1:1", "b": None, "c": "3.3.3.3:3"}

    configs = build_configs(allocs, keys, privs, endpoints)

    assert set(configs) == {"a", "b", "c"}
    # Each config has exactly the other two as peers.
    for node_id, config in configs.items():
        assert config.count("[Peer]") == 2
        assert f"PrivateKey = PRIV_{node_id}" in config


def test_build_configs_requires_keys():
    """A missing key must be named, not surface as a KeyError mid-build."""
    allocs = allocate(["a", "b"])
    with pytest.raises(ValueError, match="missing public key for b"):
        build_configs(allocs, {"a": "PUB_A"}, {"a": "PRIV_A"}, {"a": None, "b": None})


def test_validate_peer_endpoints_rejects_out_of_mesh_allowed_ips():
    peers = [PeerSpec("x", "PUB", None, "192.168.0.0/24")]
    problems = validate_peer_endpoints(peers, "10.10.0.0/24")
    assert len(problems) == 1
    assert "outside mesh" in problems[0]


# --------------------------------------------------------------------------- #
# Link classification
# --------------------------------------------------------------------------- #


def test_classification_uses_measured_rtt():
    assert classify(1.2, 0.0) == "LOW_LATENCY"
    assert classify(11.0, 0.0) == "MODERATE_LATENCY"
    assert classify(88.0, 0.0) == "HIGH_LATENCY"


def test_packet_loss_overrides_low_latency():
    """A 1 ms link dropping 5% of packets is unusable, not fast."""
    assert classify(1.0, 5.0) == "UNRELIABLE"


def test_unmeasured_link_is_not_assumed_fast():
    assert classify(None, None) == "UNRELIABLE"


def test_thresholds_are_configurable():
    strict = LinkThresholds(low_latency_max_ms=1.0, moderate_latency_max_ms=10.0)
    assert classify(0.5, 0.0, thresholds=strict) == "LOW_LATENCY"
    assert classify(5.0, 0.0, thresholds=strict) == "MODERATE_LATENCY"


def test_suitability_covers_every_class():
    for cls in ("LOW_LATENCY", "MODERATE_LATENCY", "HIGH_LATENCY", "UNRELIABLE"):
        info = suitability(cls)  # type: ignore[arg-type]
        assert "note" in info


# --------------------------------------------------------------------------- #
# Co-location assessment -- the guard against over-claiming
# --------------------------------------------------------------------------- #


def test_no_evidence_is_undetermined():
    verdict, reasons = assess([])
    assert verdict == "undetermined"
    assert reasons


def test_matching_machine_id_alone_proves_nothing():
    """Containers inherit /etc/machine-id from the image, so a match is
    uninformative. Real Vast hosts A and C share one while running on different
    motherboards; treating it as proof produced a wrong verdict."""
    a = HardwareFingerprint(machine_id="abc123")
    b = HardwareFingerprint(machine_id="abc123")
    evidence = compare_fingerprints(a, b)
    assert all(e.weight == 0 for e in evidence if e.name == "machine_id")
    verdict, _ = assess(evidence, measured_rtt_ms=0.4)
    assert verdict == "undetermined"


def test_differing_boot_id_proves_different_machines():
    """Containers on one host share the kernel, so boot_id must match. A
    difference means two physical machines."""
    a = HardwareFingerprint(boot_id="boot-1")
    b = HardwareFingerprint(boot_id="boot-2")
    evidence = compare_fingerprints(a, b)
    verdict, reasons = assess(evidence)
    assert verdict == "different-host"
    assert any("boot_id" in r for r in reasons)


def test_differing_board_serial_proves_different_machines():
    a = HardwareFingerprint(board_serial="MB-1", board_name="X99-D8 MAX")
    b = HardwareFingerprint(board_serial="ZM-2", board_name="X10SRA")
    evidence = compare_fingerprints(a, b)
    verdict, _ = assess(evidence)
    assert verdict == "different-host"


def test_matching_board_and_boot_id_confirms_same_host():
    a = HardwareFingerprint(board_serial="MB-1", board_name="X99", boot_id="b1")
    b = HardwareFingerprint(board_serial="MB-1", board_name="X99", boot_id="b1")
    evidence = compare_fingerprints(a, b)
    verdict, _ = assess(evidence, measured_rtt_ms=0.3)
    assert verdict == "same-host confirmed"


def test_differing_system_uuid_is_different_host():
    a = HardwareFingerprint(system_uuid="uuid-1")
    b = HardwareFingerprint(system_uuid="uuid-2")
    evidence = compare_fingerprints(a, b)
    verdict, reasons = assess(evidence)
    assert verdict == "different-host"


def test_same_uuid_with_high_rtt_is_not_confirmed():
    """Identical identifiers but a real network path is contradictory evidence.

    The conservative verdict is correct: a shared DMI UUID with 40 ms RTT should
    not be reported as co-located.
    """
    a = HardwareFingerprint(system_uuid="same")
    b = HardwareFingerprint(system_uuid="same")
    evidence = compare_fingerprints(a, b)
    verdict, _ = assess(evidence, measured_rtt_ms=40.0)
    assert verdict == "different-host"


def test_partial_fields_do_not_manufacture_evidence():
    """A field only one host reported must not be compared."""
    a = HardwareFingerprint(system_uuid="x", cpu_model="EPYC")
    b = HardwareFingerprint(system_uuid="x")
    evidence = compare_fingerprints(a, b)
    names = {e.name for e in evidence}
    assert "cpu_model" not in names


def test_differing_machine_id_alone_is_weak_not_different_host():
    """Machine-id differs even on one physical host, so it cannot prove separation."""
    a = HardwareFingerprint(machine_id="one")
    b = HardwareFingerprint(machine_id="two")
    evidence = compare_fingerprints(a, b)
    verdict, _ = assess(evidence, measured_rtt_ms=0.3)
    assert verdict == "undetermined"


def test_shared_ip_is_not_evidence_by_itself():
    """The central guard: nothing about a shared IP enters the verdict."""
    verdict, reasons = assess([], measured_rtt_ms=0.5)
    assert verdict == "undetermined"
    assert not any("ip" in r.lower() for r in reasons)
