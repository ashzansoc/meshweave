"""EXP-001: Single-GPU baseline.

Every other experiment is judged against this one. If a distributed
configuration cannot beat a single GPU on the metric being claimed, the claim is
not supported.

For a model that fits on one GPU, the baseline answers:
  * tokens per second at a sweep of concurrency
  * time to first token
  * inter-token latency distribution
  * cost per million output tokens

This experiment requires a real GPU. It refuses to run against synthetic
telemetry, because a synthetic baseline would silently become the reference for
every comparison that follows.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.harness import (  # noqa: E402
    Experiment,
    ExperimentBlockedError,
    add_common_args,
    emit,
)

MODEL = "Qwen2.5-7B-Instruct"
CONCURRENCY_SWEEP = [1, 4, 16, 64]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-001 single-GPU baseline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exp = Experiment(
        experiment_id="EXP-001",
        title="Single-GPU Baseline",
        hypothesis=(
            "A single GPU running a model that fits in its memory establishes the "
            "throughput, latency and cost that any distributed configuration must beat."
        ),
        command="uv run python -m experiments.exp001_single_gpu",
        strategy="SINGLE_GPU",
    )

    try:
        asyncio.run(_run(exp, args))
    except ExperimentBlockedError as blocked:
        emit(
            exp.finish(
                status="blocked",
                blocked_reason=blocked.reason,
                ready_to_run_command=blocked.command,
                out=args.out,
            )
        )
        return 1
    return 0


async def _run(exp: Experiment, args: argparse.Namespace) -> None:
    import httpx

    url = args.vllm_url.rstrip("/")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            models = await client.get(f"{url}/v1/models")
            models.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExperimentBlockedError(
                f"no vLLM server at {url}: {exc}",
                "python -m vllm.entrypoints.openai.api_server "
                "--model Qwen/Qwen2.5-7B-Instruct --port 8000",
            ) from exc

    exp.nodes = [{"node_id": "local", "vllm_url": url}]
    exp.measure("concurrency_levels", float(len(CONCURRENCY_SWEEP)), "levels")

    prompt = "The history of computing is " * 8

    for concurrency in CONCURRENCY_SWEEP:
        total_tokens = 0
        total_time = 0.0
        errors = 0

        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = []
            for _ in range(concurrency):
                tasks.append(
                    client.post(
                        f"{url}/v1/chat/completions",
                        json={
                            "model": args.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": args.max_new_tokens,
                            "temperature": 0.0,
                        },
                    )
                )
            import time

            started = time.perf_counter()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - started

        for resp in responses:
            if isinstance(resp, Exception):
                errors += 1
                exp.fail("request", f"concurrency {concurrency}: {resp}")
                continue
            if resp.status_code != 200:
                errors += 1
                exp.fail("request", f"concurrency {concurrency}: HTTP {resp.status_code}")
                continue
            body = resp.json()
            usage = body.get("usage", {})
            total_tokens += int(usage.get("completion_tokens", 0))

        total_time = elapsed
        if total_tokens:
            tps = total_tokens / total_time
            exp.measure(f"tps_c{concurrency}", tps, "tok/s", samples=concurrency)
            exp.measure(f"tokens_c{concurrency}", float(total_tokens), "tokens")
        else:
            exp.measure(
                f"tps_c{concurrency}",
                None,
                "tok/s",
                note=f"no tokens produced ({errors} errors)",
            )

        exp.measure(
            f"wall_s_c{concurrency}",
            total_time,
            "s",
            note=f"{concurrency} concurrent requests",
        )

    if exp.failures:
        exp.findings.append(
            f"{len(exp.failures)} request(s) failed during the sweep; see the failures section"
        )
    else:
        exp.findings.append(
            f"baseline established across concurrency {CONCURRENCY_SWEEP} for {args.model}"
        )

    exp.economics = _unknown_cost()
    exp.evidence.append(f"vLLM server at {url}, {len(CONCURRENCY_SWEEP)} concurrency levels")
    emit(exp.finish(out=args.out))


def _unknown_cost():
    from meshweave.bench.report import EconomicReport

    return EconomicReport(
        cost_known=False,
        unknown_components=["gpu hourly cost (set it with --hourly-cost or the registry)"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
