"""Build a tiny Qwen2 checkpoint and compare the sharded stage maths to the
reference implementation.

    .venv/bin/python tools/check_dense_parity.py

The point is not that a toy model is interesting. It is that splitting the model
across stages must not change a single logit: the stage boundary is a transport
detail, not a different computation. Run in float32 so a real mismatch is not
hidden under bf16 rounding.

transformers is a *test oracle only*. It is not installed on the compute nodes
and is not imported by anything under meshweave/.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from meshweave.runtime.dense import DenseStage, load_stage
from meshweave.runtime.model_config import DenseConfig, partition_layers

TINY = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": False,
    "attention_bias": True,
    "max_position_embeddings": 64,
}


def build_reference(tmp: Path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).float().eval()
    model.save_pretrained(tmp, safe_serialization=True)
    return model


def run_stages(stages: list[DenseStage], first_hidden: torch.Tensor, offset: int):
    """Carry hidden states through every stage; return the logits and the tail."""
    hidden = first_hidden
    logits = None
    for stage in stages:
        result = stage.forward(hidden, position_offset=offset)
        hidden = result.hidden
        if result.logits is not None:
            logits = result.logits
    return logits, hidden


def main() -> int:
    device = torch.device("cpu")
    dtype = torch.float32
    tolerance = 1e-5

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        reference = build_reference(tmp)
        config = DenseConfig.from_file(tmp / "config.json")
        print(f"config: theta={config.rope_theta:g} dtype={config.dtype} "
              f"head_dim={config.head_dim} layers={config.layers}")

        tokens = torch.tensor([[1, 5, 9, 13, 21]], dtype=torch.long)
        with torch.no_grad():
            expected = reference(input_ids=tokens).logits

        # ---- one stage holds everything -------------------------------
        whole = load_stage(
            tmp, config, 0, config.layers,
            owns_embed=True, owns_head=True, device=device, dtype=dtype,
        )
        single = DenseStage(whole, device, capacity=32, dtype=dtype)
        logits_one, _ = run_stages([single], single.embed(tokens), 0)
        delta_one = (logits_one - expected).abs().max().item()
        print(f"single stage       max |logit delta| = {delta_one:.3e}"
              f"   ({whole.bytes / 1024:.0f} KiB resident)")

        # ---- three stages, hidden states carried between them --------
        bounds = partition_layers(config, 3)
        stages: list[DenseStage] = []
        sizes: list[int] = []
        for index, (start, end) in enumerate(bounds):
            weights = load_stage(
                tmp, config, start, end,
                owns_embed=(index == 0),
                owns_head=(index == len(bounds) - 1),
                device=device, dtype=dtype,
            )
            stages.append(DenseStage(weights, device, capacity=32, dtype=dtype))
            sizes.append(weights.bytes)

        logits_split, _ = run_stages(stages, stages[0].embed(tokens), 0)
        delta_split = (logits_split - expected).abs().max().item()
        print(f"three-stage split  max |logit delta| = {delta_split:.3e}")
        for index, (start, end) in enumerate(bounds):
            print(f"  stage {index}: layers [{start},{end})  {sizes[index] / 1024:.0f} KiB")

        # ---- a decode step must match as well ------------------------
        with torch.no_grad():
            prefill = reference(input_ids=tokens, use_cache=True)

        for stage in stages:
            stage.reset()
        run_stages(stages, stages[0].embed(tokens), 0)

        next_token = torch.tensor([[7]], dtype=torch.long)
        decode_logits, _ = run_stages(stages, stages[0].embed(next_token), tokens.shape[1])

        with torch.no_grad():
            ref_decode = reference(
                input_ids=next_token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            ).logits
        delta_decode = (decode_logits - ref_decode).abs().max().item()
        print(f"decode step        max |logit delta| = {delta_decode:.3e}")

        # ---- a shard without its layers must refuse ------------------
        empty = load_stage(
            tmp, config, 2, 2, owns_embed=False, owns_head=False,
            device=device, dtype=dtype,
        )
        refused = empty.bytes == 0 and not empty.layers
        print(f"empty shard holds no weights: {refused}")

    worst = max(delta_one, delta_split, delta_decode)
    ok = worst < tolerance and refused
    print()
    print(f"PARITY {'OK' if ok else 'FAILED'} (worst {worst:.3e}, tolerance {tolerance:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
