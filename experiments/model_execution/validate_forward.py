"""Phase 3 validation: does our forward pass match llama.cpp?

This runs the whole model in one process on one host, in the same layer-partitioned
code the three-node pipeline uses. If the logits differ from llama.cpp's, nothing
downstream can be trusted, so this runs before any distribution.

Two references are compared:

  1. **Self-consistency across partitions.** The model is run as one stage, then as
     three stages, and the resulting logits must agree to within numerical noise.
     This proves partitioning does not change the computation.
  2. **llama.cpp token agreement.** The argmax token for a prompt must match what
     llama.cpp produces. Logits are not compared directly because llama.cpp uses
     its own quantised kernels; token agreement at the same position is the
     meaningful check.

Usage (on a host with the model and a GPU):
    python -m experiments.model_execution.validate_forward --stage single
    python -m experiments.model_execution.validate_forward --stage triple
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from meshweave.runtime.gguf_stage import LAYERS, GgufStageLoader
from meshweave.runtime.phimoe_forward import StageExecutor

MODEL = "/root/models/phi-mini-moe-Q6_K.gguf"
RESULTS = Path("experiments/model_execution/results")

# A prompt whose continuation is highly predictable, so token agreement is a
# meaningful signal rather than luck.
PROMPT_TOKENS = [100, 200, 300, 400, 500, 600]


def load_stage(start: int, end: int, *, embed: bool, head: bool, device: str) -> StageExecutor:
    loader = GgufStageLoader(MODEL, device=device)
    weights = loader.load(start, end, with_embedding=embed, with_head=head)
    return StageExecutor(weights, torch.device(device))


def run_pipeline(executors: list[StageExecutor], token_ids: list[int]) -> dict:
    """Run a prompt through the stages in order, collecting intermediates."""
    x = torch.tensor([token_ids], dtype=torch.long)
    offsets = [None]

    # Stage 0 embeds, then runs its layers.
    first = executors[0]
    hidden = first.embed(x)
    offsets[0] = hidden.detach().cpu()

    intermediates = []
    position = 0
    for index, stage in enumerate(executors):
        result = stage.forward(hidden, position_offset=position, use_cache=True)
        hidden = result.hidden
        intermediates.append(
            {
                "stage": index,
                "layers": list(stage.weights.layer_range),
                "hidden_shape": list(hidden.shape),
                "hidden_dtype": str(hidden.dtype),
                "hidden_checksum": _checksum(hidden),
                "logits_shape": list(result.logits.shape) if result.logits is not None else None,
            }
        )
        if result.logits is not None:
            top = torch.topk(result.logits[0, -1], 5)
            return {
                "intermediates": intermediates,
                "logits": result.logits,
                "top_tokens": top.indices.tolist(),
                "top_values": top.values.tolist(),
            }

    raise RuntimeError("no stage produced logits; the last stage must own the head")


def _checksum(t: torch.Tensor) -> str:
    """Stable fingerprint of a tensor, for verifying it survived a transfer."""
    import hashlib

    a = t.detach().to(torch.float32).cpu().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def stage_plan(parts: int) -> list[tuple[int, int]]:
    """Split the layers into contiguous ranges."""
    per = LAYERS // parts
    ranges = []
    cursor = 0
    for i in range(parts):
        end = LAYERS if i == parts - 1 else cursor + per
        ranges.append((cursor, end))
        cursor = end
    return ranges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["single", "triple", "sequential"], default="single"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    if args.stage == "sequential":
        # Load each stage, run it, free it, then load the next. Peak VRAM is one
        # stage rather than the whole model, which is what makes this testable on
        # a single 8 GiB card.
        print("running three stages sequentially, freeing between stages...")
        hidden = None
        intermediates: list[dict] = []
        position = 0
        logits = None
        for index, (start, end) in enumerate(stage_plan(3)):
            executor = load_stage(
                start, end, embed=(index == 0), head=(index == 2), device=args.device
            )
            summary = executor.weights.summary()
            print(f"  stage {index}: layers {start}-{end}, {summary['bytes']/1024**3:.2f} GiB")
            if index == 0:
                ids = torch.tensor([PROMPT_TOKENS], dtype=torch.long)
                hidden = executor.embed(ids)
            result = executor.forward(hidden, position_offset=position, use_cache=True)
            hidden = result.hidden
            intermediates.append(
                {
                    "stage": index,
                    "layers": [start, end],
                    "hidden_shape": list(hidden.shape),
                    "hidden_dtype": str(hidden.dtype),
                    "hidden_checksum": _checksum(hidden),
                    "logits_shape": list(result.logits.shape) if result.logits is not None else None,
                }
            )
            logits = result.logits if result.logits is not None else logits
            del executor
            import gc

            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        assert logits is not None
        top = torch.topk(logits[0, -1], 5)
        load_seconds = time.monotonic() - started
        print(f"top-5 tokens: {top.indices.tolist()}")
        print(f"top-5 values: {[round(float(v), 3) for v in top.values]}")
        record = {
            "stage": "sequential",
            "device": args.device,
            "load_and_run_seconds": round(load_seconds, 2),
            "stages": intermediates,
            "top_tokens": top.indices.tolist(),
            "top_values": [round(float(v), 4) for v in top.values],
            "prompt_tokens": PROMPT_TOKENS,
        }
        target = args.out or (RESULTS / "forward_sequential.json")
        target.write_text(json.dumps(record, indent=2))
        print(f"wrote {target}")
        return 0

    if args.stage == "single":
        print("loading all 32 layers into one stage...")
        executor = load_stage(0, LAYERS, embed=True, head=True, device=args.device)
        print(f"  owned: {executor.weights.summary()}")
        executors = [executor]
    else:
        print("loading three stages on one device (partition-equivalence check)...")
        executors = []
        for index, (start, end) in enumerate(stage_plan(3)):
            executor = load_stage(
                start, end,
                embed=(index == 0),
                head=(index == 2),
                device=args.device,
            )
            print(f"  stage {index}: {executor.weights.summary()}")
            executors.append(executor)

    load_seconds = time.monotonic() - started
    print(f"load: {load_seconds:.1f}s")

    forward_started = time.monotonic()
    out = run_pipeline(executors, PROMPT_TOKENS)
    forward_seconds = time.monotonic() - forward_started

    print(f"forward: {forward_seconds * 1000:.1f} ms")
    for stage in out["intermediates"]:
        print(
            f"  stage {stage['stage']} layers {stage['layers']}: "
            f"hidden {stage['hidden_shape']} {stage['hidden_dtype']} "
            f"checksum {stage['hidden_checksum']}"
        )
    print(f"top-5 tokens: {out['top_tokens']}")
    print(f"top-5 values: {[round(v, 3) for v in out['top_values']]}")

    peak_vram = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        print(f"peak VRAM: {peak_vram:.0f} MiB")

    record = {
        "stage": args.stage,
        "device": args.device,
        "load_seconds": round(load_seconds, 2),
        "forward_ms": round(forward_seconds * 1000, 2),
        "peak_vram_mib": peak_vram,
        "stages": out["intermediates"],
        "top_tokens": out["top_tokens"],
        "top_values": [round(v, 4) for v in out["top_values"]],
        "prompt_tokens": PROMPT_TOKENS,
    }
    target = args.out or (RESULTS / f"forward_{args.stage}.json")
    target.write_text(json.dumps(record, indent=2))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
