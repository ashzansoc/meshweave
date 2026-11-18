"""Link suitability must follow the arithmetic, not a headline threshold.

The rule compares the communication a strategy requires against compute measured
on this hardware. That is why a 300 ms link is "unsuitable" for a pipeline, why
tensor parallelism is hopeless on every link here, and why replication is always
fine -- and why no verdict appears at all when no compute time was measured.
"""

from __future__ import annotations

from meshweave.network.intelligence import (
    VERDICT_RATIOS,
    LinkView,
    assess_strategy,
    pipeline_communication_ms,
    tensor_communication_ms,
)

COMPUTE_MS = 30.0
HIDDEN_BYTES = 1792.0  # 1 x 896 x fp16: one Qwen2.5-0.5B decode step


def _slow() -> LinkView:
    return LinkView("a", "b", rtt_p50_ms=310.0, jitter_ms=3.0, download_mbps=0.8, ok=True)


def _fast() -> LinkView:
    return LinkView("a", "b", rtt_p50_ms=0.5, jitter_ms=0.1, download_mbps=200.0, ok=True)


def _verdicts(link: LinkView, **kwargs) -> dict:
    return {
        v["strategy"]: v
        for v in assess_strategy(
            link,
            compute_ms_per_token=kwargs.pop("compute_ms_per_token", COMPUTE_MS),
            hops=kwargs.pop("hops", 2),
            bytes_per_token=kwargs.pop("bytes_per_token", HIDDEN_BYTES),
            layers=kwargs.pop("layers", 24),
            params_per_layer=kwargs.pop("params_per_layer", 60_000_000),
            **kwargs,
        )["verdicts"]
    }


def test_a_pipeline_over_a_300ms_link_is_unsuitable() -> None:
    verdicts = _verdicts(_slow())
    pipeline = verdicts["PIPELINE"]
    # The measured 453 ms/token is roughly 15x a 30 ms compute time.
    assert pipeline["communication_ms"] > 600
    assert pipeline["fraction_of_compute"] > VERDICT_RATIOS[1]
    assert pipeline["verdict"] == "unsuitable"


def test_the_same_pipeline_on_a_fast_link_is_suitable() -> None:
    pipeline = _verdicts(_fast())["PIPELINE"]
    assert pipeline["communication_ms"] < 2
    assert pipeline["fraction_of_compute"] <= VERDICT_RATIOS[0]
    assert pipeline["verdict"] == "suitable"


def test_tensor_parallelism_is_hopeless_on_every_link_here() -> None:
    for link in (_slow(), _fast()):
        tensor = _verdicts(link)["TENSOR"]
        assert tensor["verdict"] == "unsuitable"
        assert tensor["communication_ms"] > 1e5  # minutes per token at best


def test_replication_never_pays_a_communication_cost() -> None:
    for link in (_slow(), _fast()):
        replica = _verdicts(link)["REPLICATE"]
        assert replica["communication_ms"] == 0.0
        assert replica["verdict"] == "suitable"
        assert "full copy" in replica["basis"]


def test_no_compute_time_means_no_verdict() -> None:
    result = assess_strategy(
        _fast(), compute_ms_per_token=None, hops=2, bytes_per_token=HIDDEN_BYTES
    )
    assert all(v["verdict"] == "indeterminate" for v in result["verdicts"])
    assert "no measured compute time" in result["verdict_scale"]["indeterminate"]


def test_a_broken_link_is_reported_as_missing_not_slow() -> None:
    broken = LinkView("a", "b", rtt_p50_ms=0.0, jitter_ms=None, download_mbps=None, ok=False)
    result = assess_strategy(broken, compute_ms_per_token=COMPUTE_MS, hops=2, bytes_per_token=1)
    # A missing link is reported once, not as three strategies that all failed.
    assert result["verdict"] == "no_link"
    assert "did not carry a measurement" in result["reason"]
    assert result["communication_ms"] is None


def test_hops_change_the_verdict() -> None:
    two = pipeline_communication_ms(_fast(), hops=2, bytes_per_token=HIDDEN_BYTES)
    four = pipeline_communication_ms(_fast(), hops=4, bytes_per_token=HIDDEN_BYTES)
    assert four is not None and two is not None
    assert four > two


def test_tensor_cost_scales_with_layer_count() -> None:
    small = tensor_communication_ms(_fast(), layers=4, params_per_layer=1_000_000)
    large = tensor_communication_ms(_fast(), layers=24, params_per_layer=60_000_000)
    assert large is not None and small is not None
    assert large > small * 100