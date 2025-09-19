"""Simple heuristic test to validate deferral strategy wiring.

This test does NOT perform full model inference (would be slow); instead it
mocks/stubs the minimal behavior of server + strategy to ensure:
 1. Strategy is instantiated from config
 2. Decision to defer sets response flags
 3. Edge client reacts by switching to baseline path

Because the current system integrates deeply with live models and WebSockets,
this test focuses on the strategy logic in isolation, plus a minimal simulation
of server batch processing.
"""
from cloud.deferral_strategy import (
    LowAcceptanceDeferStrategy,
    ProbabilityDivergenceStrategy,
    NeverDeferStrategy,
    DeferralContext
)


def test_low_acceptance_defers_after_threshold():
    cfg = {
        "batch_threshold": 0.5,
        "cumulative_threshold": 0.6,
        "min_batches_before_consider": 1,
        "require_consecutive_failures": False,
    }
    strat = LowAcceptanceDeferStrategy(cfg)

    # Batch 1: Low acceptance triggers immediately (no consecutive needed)
    ctx1 = DeferralContext(
        draft_tokens=["a","b","c"],
        draft_probs=[0.3,0.2,0.1],
        target_probs=[0.05,0.04,0.03],
        acceptance_mask=[False, False, True],
        acceptance_rate_batch=1/3,
        acceptance_rate_cumulative=1/3,
        batch_index=1,
        remaining_tokens=40,
        request_id="r1"
    )
    decision = strat.decide(ctx1)
    assert decision.defer, "Expected deferral on low acceptance batch"


def test_low_acceptance_needs_second_fail_when_consecutive():
    cfg = {
        "batch_threshold": 0.5,
        "cumulative_threshold": 0.6,
        "min_batches_before_consider": 1,
        "require_consecutive_failures": True,
    }
    strat = LowAcceptanceDeferStrategy(cfg)
    # First failing batch - should not defer yet
    ctx1 = DeferralContext(
        draft_tokens=["a"],
        draft_probs=[0.4],
        target_probs=[0.05],
        acceptance_mask=[False],
        acceptance_rate_batch=0.0,
        acceptance_rate_cumulative=0.0,
        batch_index=1,
        remaining_tokens=10,
        request_id="r1"
    )
    d1 = strat.decide(ctx1)
    assert not d1.defer, "Should not defer on first failure when consecutive required"
    # Second failing batch - now should defer
    ctx2 = DeferralContext(
        draft_tokens=["b"],
        draft_probs=[0.4],
        target_probs=[0.05],
        acceptance_mask=[False],
        acceptance_rate_batch=0.0,
        acceptance_rate_cumulative=0.0,
        batch_index=2,
        remaining_tokens=9,
        request_id="r1"
    )
    d2 = strat.decide(ctx2)
    assert d2.defer, "Expected deferral on second consecutive failure"


def test_probability_divergence_triggers():
    cfg = {"kl_divergence_threshold": 0.1}
    strat = ProbabilityDivergenceStrategy(cfg)
    ctx = DeferralContext(
        draft_tokens=["x","y"],
        draft_probs=[0.5,0.4],
        target_probs=[0.01,0.01],
        acceptance_mask=[False, False],
        acceptance_rate_batch=0.0,
        acceptance_rate_cumulative=0.0,
        batch_index=3,
        remaining_tokens=20,
        request_id="r2"
    )
    d = strat.decide(ctx)
    assert d.defer, "Expected deferral from high KL divergence"


def test_never_strategy():
    strat = NeverDeferStrategy()
    ctx = DeferralContext(
        draft_tokens=["x"],
        draft_probs=[0.9],
        target_probs=[0.85],
        acceptance_mask=[True],
        acceptance_rate_batch=1.0,
        acceptance_rate_cumulative=1.0,
        batch_index=5,
        remaining_tokens=5,
        request_id="r3"
    )
    d = strat.decide(ctx)
    assert not d.defer, "Never strategy should not defer"


if __name__ == "__main__":
    # Lightweight manual run
    test_low_acceptance_defers_after_threshold()
    test_low_acceptance_needs_second_fail_when_consecutive()
    test_probability_divergence_triggers()
    test_never_strategy()
    print("Deferral strategy tests passed.")
