"""The stage runner's own wiring, without transformers or the network.

The bug this exists for: ``build_stage`` read ``Path(args.model_dir)``, which is
``None`` whenever the model came from the hub rather than a local directory. The
earlier pipeline test only ever passed ``--model-dir``, so the repo path failed
at the first real deployment with a TypeError after the download had already
completed.
"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent

TINY = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "vocab_size": 16,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "attention_bias": True,
    "torch_dtype": "float32",
}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "mw_stage_runner", REPO / "deploy" / "stage_runner.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(directory: Path, config: dict) -> None:
    """A Qwen2 checkpoint laid out exactly as the hub ships one."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    hidden, heads, kv, dim = (
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    ffn, vocab = config["intermediate_size"], config["vocab_size"]

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape) / 10.0

    tensors = {
        "model.embed_tokens.weight": rand(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": rand(vocab, hidden),
    }
    for index in range(config["num_hidden_layers"]):
        p = f"model.layers.{index}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.self_attn.q_proj.weight"] = rand(heads * dim, hidden)
        tensors[f"{p}.self_attn.k_proj.weight"] = rand(kv * dim, hidden)
        tensors[f"{p}.self_attn.v_proj.weight"] = rand(kv * dim, hidden)
        tensors[f"{p}.self_attn.o_proj.weight"] = rand(hidden, heads * dim)
        tensors[f"{p}.self_attn.q_proj.bias"] = rand(heads * dim)
        tensors[f"{p}.self_attn.k_proj.bias"] = rand(kv * dim)
        tensors[f"{p}.self_attn.v_proj.bias"] = rand(kv * dim)
        tensors[f"{p}.mlp.gate_proj.weight"] = rand(ffn, hidden)
        tensors[f"{p}.mlp.up_proj.weight"] = rand(ffn, hidden)
        tensors[f"{p}.mlp.down_proj.weight"] = rand(hidden, ffn)
    save_file(tensors, str(directory / "model.safetensors"))


def _args(model_dir: str | None, **overrides) -> Namespace:
    base = dict(
        model_dir=model_dir,
        rank=0,
        world=1,
        start=0,
        end=2,
        device="cpu",
        dtype="float32",
        max_seq=16,
    )
    base.update(overrides)
    return Namespace(**base)


def test_build_stage_accepts_a_hub_downloaded_model(tmp_path: Path) -> None:
    """args.model_dir is None when the model came from a repository."""
    from meshweave.runtime.model_config import DenseConfig

    _write_checkpoint(tmp_path, TINY)
    runner = _load_runner()
    config = DenseConfig.from_file(tmp_path / "config.json")

    stage, resident = runner.build_stage(_args(None), config, tmp_path)
    assert resident > 0
    assert sorted(stage.weights.layers) == [0, 1]
    assert stage.weights.owns_embed and stage.weights.owns_head


def test_a_partial_stage_does_not_load_the_other_layers(tmp_path: Path) -> None:
    """The memory claim rests on this: a stage must not read the whole model."""
    from meshweave.runtime.model_config import DenseConfig

    _write_checkpoint(tmp_path, TINY)
    runner = _load_runner()
    config = DenseConfig.from_file(tmp_path / "config.json")

    first, first_bytes = runner.build_stage(_args(None, rank=0, world=2, start=0, end=1), config, tmp_path)
    second, second_bytes = runner.build_stage(
        _args(None, rank=1, world=2, start=1, end=2), config, tmp_path
    )

    assert list(first.weights.layers) == [0]
    assert list(second.weights.layers) == [1]
    assert first.weights.embedding is not None
    assert second.weights.embedding is None
    assert second.weights.lm_head is not None
    assert first_bytes + second_bytes < 2 * max(first_bytes, second_bytes) + 1


def test_a_missing_checkpoint_file_is_reported(tmp_path: Path) -> None:
    from meshweave.errors import ModelLoadError
    from meshweave.runtime.model_config import DenseConfig

    _write_checkpoint(tmp_path, TINY)
    (tmp_path / "model.safetensors").unlink()
    runner = _load_runner()
    config = DenseConfig.from_file(tmp_path / "config.json")

    with pytest.raises(ModelLoadError):
        runner.build_stage(_args(None), config, tmp_path)
