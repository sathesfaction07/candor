import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sme_bridge.models import ConfidenceBreakdown

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    should_answer: bool
    reason: str
    # A calibrated gate reports the probability it actually used. ThresholdGate leaves these
    # alone, so a caller can always tell whether the threshold it was compared against
    # carries a probabilistic meaning or is merely an evidence score.
    probability: float | None = None
    calibrated: bool = False


class Gate(ABC):
    @abstractmethod
    def decide(self, confidence: ConfidenceBreakdown) -> GateDecision: ...


class ThresholdGate(Gate):
    # Compares the raw evidence composite against a fixed threshold. Note that an
    # UNCALIBRATED composite is not a probability, so a 0.90 threshold here does not mean
    # "90% likely correct" -- it means "this much evidence". CalibratedGate, once a
    # calibrator has been fitted, is the version whose threshold carries that meaning.
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def decide(self, confidence: ConfidenceBreakdown) -> GateDecision:
        if confidence.composite >= self.threshold:
            return GateDecision(should_answer=True, reason="Confidence meets threshold")

        # A None agreement means it was never measured, so it must not be eligible to be
        # named the weakest component. Availability is derived from the data rather than
        # assumed, so a breakdown built by hand in a test behaves the same as a real one.
        scores = {
            "retrieval_support": confidence.retrieval_support,
            "citation_coverage": confidence.citation_coverage,
        }
        if confidence.ensemble_agreement is not None:
            scores["ensemble_agreement"] = confidence.ensemble_agreement
        weakest, weakest_score = min(scores.items(), key=lambda kv: kv[1])

        reason = (
            f"Confidence below threshold ({confidence.composite:.2f} < {self.threshold:.2f}); "
            f"weakest measured component: {weakest} ({weakest_score:.2f})"
        )
        return GateDecision(should_answer=False, reason=reason)
