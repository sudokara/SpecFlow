"""
Deferral (speculative cascade) strategy interfaces and basic implementations.

The strategy decides whether the cloud should instruct the edge to stop
sending further draft tokens and switch to a cloud-only (baseline) path.

Initial implementation provides a No-Op (NeverDeferStrategy). Future strategies
can use acceptance rates, probability divergence, or latency heuristics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
import math

@dataclass
class DeferralContext:
    """Contextual signals for making a deferral decision.

    All fields are optional for the simplest strategy, but populated when
    probabilistic verification is available.
    """
    draft_tokens: List[str]
    draft_probs: List[float]
    target_probs: List[float]
    acceptance_mask: List[bool]
    acceptance_rate_batch: float
    acceptance_rate_cumulative: float
    batch_index: int
    remaining_tokens: int
    request_id: str = ""

@dataclass
class DeferralDecision:
    defer: bool
    reason: str = ""
    scope: str = "request"  # Reserved for future granular scopes

class DeferralStrategy(ABC):
    """Abstract base class for deferral strategies."""

    @abstractmethod
    def decide(self, context: DeferralContext) -> DeferralDecision:
        """Return a DeferralDecision for the given context."""
        raise NotImplementedError

class NeverDeferStrategy(DeferralStrategy):
    """Default strategy that never defers speculative drafting."""
    def decide(self, context: DeferralContext) -> DeferralDecision:
        return DeferralDecision(defer=False, reason="never", scope="request")

class LowAcceptanceDeferStrategy(DeferralStrategy):
    """Defer when batch & cumulative acceptance drop below configured thresholds.

    Config keys used (with defaults if missing):
      batch_threshold
      cumulative_threshold
      min_batches_before_consider
      require_consecutive_failures
    """
    def __init__(self, cfg: dict):
        self.batch_threshold = float(cfg.get("batch_threshold", 0.0))
        self.cumulative_threshold = float(cfg.get("cumulative_threshold", 0.0))
        self.min_batches = int(cfg.get("min_batches_before_consider", 1))
        self.require_consecutive = bool(cfg.get("require_consecutive_failures", True))
        # Internal state for consecutive failures
        self._prev_failed = False

    def decide(self, context: DeferralContext) -> DeferralDecision:
        if context.batch_index < self.min_batches:
            self._prev_failed = False
            return DeferralDecision(defer=False)
        batch_fail = context.acceptance_rate_batch < self.batch_threshold if self.batch_threshold > 0 else False
        cumulative_fail = context.acceptance_rate_cumulative < self.cumulative_threshold if self.cumulative_threshold > 0 else False
        decision = False
        if batch_fail and cumulative_fail:
            if self.require_consecutive:
                decision = self._prev_failed  # Only defer on second consecutive failure
            else:
                decision = True
        self._prev_failed = batch_fail and cumulative_fail
        if decision:
            reason = (
                f"low_acceptance batch={context.acceptance_rate_batch:.3f} < {self.batch_threshold} "
                f"and cumulative={context.acceptance_rate_cumulative:.3f} < {self.cumulative_threshold}"
            )
            return DeferralDecision(defer=True, reason=reason)
        return DeferralDecision(defer=False)

class ProbabilityDivergenceStrategy(DeferralStrategy):
    """Heuristic: Defer if the KL divergence between target and draft implied probs exceeds a threshold,
    indicating draft model distribution is unhelpful.

    Approximate draft distribution with provided sample probs; compute per-token ratio.
    Config key: kl_divergence_threshold (default 5.0)
    """
    def __init__(self, cfg: dict):
        self.kl_threshold = float(cfg.get("kl_divergence_threshold", 5.0))

    def decide(self, context: DeferralContext) -> DeferralDecision:
        # Need aligned probabilities
        if not context.draft_probs or not context.target_probs:
            return DeferralDecision(defer=False)
        kl = 0.0
        count = 0
        for p_draft, p_target in zip(context.draft_probs, context.target_probs):
            if p_draft > 0 and p_target > 0:
                kl += p_target * math.log(p_target / p_draft)
                count += 1
        avg_kl = kl / count if count else 0.0
        if avg_kl >= self.kl_threshold:
            return DeferralDecision(defer=True, reason=f"prob_divergence KL={avg_kl:.2f} >= {self.kl_threshold}")
        return DeferralDecision(defer=False)

# Factory for strategy instantiation (extensible later)

def build_deferral_strategy(name: str, config: dict) -> DeferralStrategy:
    name_lc = (name or "").lower()
    if name_lc in ("", "never", "none", "off"):
        return NeverDeferStrategy()
    if name_lc in ("low_acceptance", "low-acceptance", "acceptance"):
        return LowAcceptanceDeferStrategy(config)
    if name_lc in ("prob_divergence", "prob-divergence", "divergence", "kl"):
        return ProbabilityDivergenceStrategy(config)
    # Placeholder for future strategies; fallback to never
    return NeverDeferStrategy()
