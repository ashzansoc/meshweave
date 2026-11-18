"""Deployment planning: what MESHWEAVE accepts, and what it refuses.

The refusals matter more than the acceptances. A deployment that cannot work
must be rejected with the node and the shortfall named, because the alternative
is a half-started pipeline that looks like a bug in the model.
"""

from __future__ import annotations

import pytest

from meshweave.coordinator.deployments import (
    DeploymentRejected,
    DeploymentStore,
    normalise_repo,
    pick_port,
    plan_deployment,
)
from meshweave.core import GpuDevice, GpuInventory, HostResources, NetworkInfo, NodeRecord
from meshweave.runtime.model_config import DenseConfig

GIB = 1024**3

QWEN_05B = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "attention_bias": True,
    "torch_dtype": "bfloat16",
}


def node(
    node_id: str,
    *,
    vram_gib: float = 8.0,
    disk_gib: float = 40.0,
    mapped: dict[int, int] | None = None,
    external_ip: str | None = "203.0.113.9",
    gpus: int = 1,
    agent_version: str = "0.4.0",
) -> NodeRecord:
    inventory = GpuInventory(
        gpus=[
            GpuDevice(
                index=0,
                uuid=f"GPU-{node_id}-{i}",
                name="Test GPU",
                vram_total_bytes=int(vram_gib * GIB),
                vram_free_bytes=int(vram_gib * GIB),
                vram_used_bytes=0,
            )
            for i in range(gpus)
        ],
        host=HostResources(disk_available_bytes=int(disk_gib * GIB)),
        network=NetworkInfo(
            private_ip="10.0.0.2",
            external_ip=external_ip,
            mapped_ports=mapped or {},
        ),
        agent_version=agent_version,
    )
    from meshweave.core import NodeRegistration

    return NodeRecord(
        node_id=node_id,
        registration=NodeRegistration(
            node_id=node_id, hostname=node_id, agent_version=agent_version, provider="manual"
        ),
        inventory=inventory,
    )


# --------------------------------------------------------------------------- #
# model identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/", "Qwen/Qwen2.5-0.5B-Instruct"),
        (
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
    ],
)
def test_repo_urls_normalise(value: str, expected: str) -> None:
    assert normalise_repo(value) == expected


@pytest.mark.parametrize("value", ["", "justoneword", "a/b/c"])
def test_bad_repo_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        normalise_repo(value)


# --------------------------------------------------------------------------- #
# port selection
# --------------------------------------------------------------------------- #


def test_ssh_port_is_never_chosen_for_peer_traffic() -> None:
    record = node("a", mapped={22: 40000, 10100: 41000})
    inventory = record.inventory
    assert inventory is not None
    assert pick_port(inventory) == (10100, 41000)


def test_a_node_with_only_ssh_publishes_nothing_usable() -> None:
    record = node("a", mapped={22: 40000})
    inventory = record.inventory
    assert inventory is not None
    assert pick_port(inventory) is None


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_two_nodes_split_the_layers_exactly_once() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    stages = plan_deployment(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        records=[node("a", mapped={10100: 41000}), node("b", mapped={10100: 42000})],
        config=config,
    )
    assert [(s.start, s.end) for s in stages] == [(0, 12), (12, 24)]
    covered = [layer for s in stages for layer in range(s.start, s.end)]
    assert covered == list(range(config.layers))
    assert stages[0].address == "203.0.113.9:41000"
    assert stages[1].address == "203.0.113.9:42000"


def test_a_node_with_no_published_port_is_refused() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    with pytest.raises(DeploymentRejected) as excinfo:
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            records=[node("a", mapped={10100: 41000}), node("b")],
            config=config,
        )
    message = str(excinfo.value)
    assert "b" in message and "TCP port" in message


def test_a_node_without_enough_vram_is_refused_with_the_numbers() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    with pytest.raises(DeploymentRejected) as excinfo:
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            records=[node("a", vram_gib=0.2, mapped={10100: 41000})],
            config=config,
        )
    message = str(excinfo.value)
    assert "0.20 GiB free VRAM" in message
    assert "needs" in message


def test_a_node_without_disk_for_the_checkpoint_is_refused() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    with pytest.raises(DeploymentRejected) as excinfo:
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            records=[node("a", disk_gib=0.5, mapped={10100: 41000})],
            config=config,
        )
    assert "whole checkpoint" in str(excinfo.value)


def test_a_node_with_no_gpu_is_refused() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    with pytest.raises(DeploymentRejected) as excinfo:
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            records=[node("a", gpus=0, mapped={10100: 41000})],
            config=config,
        )
    assert "no GPU" in str(excinfo.value)


def test_single_node_needs_no_published_port() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    stages = plan_deployment(
        model="Qwen/Qwen2.5-0.5B-Instruct", records=[node("a")], config=config
    )
    assert len(stages) == 1
    assert stages[0].address is None
    assert (stages[0].start, stages[0].end) == (0, 24)


def test_more_nodes_than_layers_is_refused() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    records = [node(f"n{i}", mapped={10100: 41000 + i}) for i in range(config.layers + 1)]
    with pytest.raises(DeploymentRejected):
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct", records=records, config=config
        )


# --------------------------------------------------------------------------- #
# command dispatch
# --------------------------------------------------------------------------- #


def test_stage_command_carries_the_right_peer_addresses() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    stages = plan_deployment(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        records=[node("a", mapped={10100: 41000}), node("b", mapped={10100: 42000})],
        config=config,
    )
    from meshweave.coordinator.deployments import new_deployment

    deployment = new_deployment("Qwen/Qwen2.5-0.5B-Instruct", stages, config, 2048)
    store = DeploymentStore()
    store.add(deployment)

    first = [c for c in store.commands_for("a") if c["kind"] == "run_stage"][0]
    second = [c for c in store.commands_for("b") if c["kind"] == "run_stage"][0]

    assert first["rank"] == 0 and first["next"] == "203.0.113.9:42000"
    assert first["reply"] is None
    assert second["rank"] == 1 and second["reply"] == "203.0.113.9:41000"


def test_running_stage_is_not_dispatched_twice() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    stages = plan_deployment(
        model="Qwen/Qwen2.5-0.5B-Instruct", records=[node("a")], config=config
    )
    from meshweave.coordinator.deployments import new_deployment

    deployment = new_deployment("Qwen/Qwen2.5-0.5B-Instruct", stages, config, 2048)
    store = DeploymentStore()
    store.add(deployment)
    deployment.node_status[f"{deployment.deployment_id}#a#0"] = {"phase": "running"}

    assert [c for c in store.commands_for("a") if c["kind"] == "run_stage"] == []


def test_stop_is_dispatched_when_the_deployment_is_stopped() -> None:
    config = DenseConfig.from_json(QWEN_05B)
    stages = plan_deployment(
        model="Qwen/Qwen2.5-0.5B-Instruct", records=[node("a")], config=config
    )
    from meshweave.coordinator.deployments import new_deployment

    deployment = new_deployment("Qwen/Qwen2.5-0.5B-Instruct", stages, config, 2048)
    store = DeploymentStore()
    store.add(deployment)
    deployment.stop_requested = True

    kinds = [c["kind"] for c in store.commands_for("a")]
    assert kinds == ["stop_stage"]


def test_an_agent_that_cannot_poll_is_refused_with_the_version() -> None:
    """0.2.0 registers and heartbeats but never polls for work.

    Sending it a stage produces silence, which is worse than a refusal: the
    deployment looks hung rather than unsupported.
    """
    config = DenseConfig.from_json(QWEN_05B)
    with pytest.raises(DeploymentRejected) as excinfo:
        plan_deployment(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            records=[node("old", mapped={10100: 41000}, agent_version="0.3.0")],
            config=config,
        )
    message = str(excinfo.value)
    assert "0.3.0" in message and "0.4.0" in message and "installer" in message


def test_the_version_comparison_is_numeric_not_lexical() -> None:
    from meshweave.coordinator.deployments import agent_can_deploy

    assert agent_can_deploy("0.4.0")
    assert agent_can_deploy("0.10.1")
    assert agent_can_deploy("1.0.0")
    assert not agent_can_deploy("0.3.9")
    assert not agent_can_deploy(None)


def test_turing_cards_get_float16_instead_of_bfloat16() -> None:
    """bf16 needs compute 8.0; a 2070 SUPER is 7.5 and would fail at the first op."""
    from meshweave.coordinator.deployments import effective_dtype

    config = DenseConfig.from_json(QWEN_05B)
    turing = node("turing", mapped={10100: 41000})
    turing.inventory.gpus[0].compute_capability = (7, 5)
    ampere = node("ampere", mapped={10100: 42000})
    ampere.inventory.gpus[0].compute_capability = (8, 6)

    assert effective_dtype(config, [turing]) == "float16"
    assert effective_dtype(config, [ampere]) == "bfloat16"
    assert effective_dtype(config, [ampere, turing]) == "float16"


def test_a_known_free_port_is_preferred_over_a_low_one() -> None:
    """Both live nodes publish 1111, which something in the image may own."""
    record = node("a", mapped={1111: 11517, 8080: 11568, 10100: 11524, 10200: 11493})
    assert pick_port(record.inventory) == (10100, 11524)


def test_without_a_preferred_port_the_highest_is_used() -> None:
    record = node("a", mapped={1111: 11517, 6006: 11471, 8080: 11568})
    assert pick_port(record.inventory) == (8080, 11568)


def test_a_port_that_cannot_be_bound_is_ignored() -> None:
    """Vast publishes an entry whose internal value (72299) is not a TCP port.
    Choosing it made the stage fail with OverflowError at bind()."""
    record = node("a", mapped={"72299": 46012, "22": 46063, "10100": 46058})
    internal, _ = pick_port(record.inventory)
    assert internal == 10100

    only_bad = node("b", mapped={"72299": 46012})
    assert pick_port(only_bad.inventory) is None


def test_a_qwen2_config_without_attention_bias_still_loads_the_biases() -> None:
    """Qwen/Qwen2.5-0.5B-Instruct's config.json has no attention_bias key, but
    its checkpoint has q_proj.bias, k_proj.bias and v_proj.bias. Defaulting to
    False computed attention without them."""
    from meshweave.runtime.dense import layer_keys
    from meshweave.runtime.model_config import DenseConfig

    # The real Qwen/Qwen2.5-0.5B-Instruct config.json has no attention_bias key.
    without = {k: v for k, v in QWEN_05B.items() if k != "attention_bias"}
    config = DenseConfig.from_json(without)
    assert config.attn_bias is True
    assert "model.layers.0.self_attn.q_proj.bias" in layer_keys(config, 0)
