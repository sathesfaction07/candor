from sme_bridge.core.gate import ThresholdGate
from sme_bridge.models import ConfidenceBreakdown


def test_just_above_threshold():
    gate = ThresholdGate(threshold=0.90)
    confidence = ConfidenceBreakdown(
        retrieval_support=0.95,
        ensemble_agreement=0.92,
        citation_coverage=0.91,
        composite=0.92,
        components={},
    )
    decision = gate.decide(confidence)
    assert decision.should_answer is True


def test_just_below_threshold():
    gate = ThresholdGate(threshold=0.90)
    confidence = ConfidenceBreakdown(
        retrieval_support=0.85,
        ensemble_agreement=0.88,
        citation_coverage=0.87,
        composite=0.86,
        components={},
    )
    decision = gate.decide(confidence)
    assert decision.should_answer is False


def test_reason_names_weakest_component():
    gate = ThresholdGate(threshold=0.90)
    confidence = ConfidenceBreakdown(
        retrieval_support=0.95,
        ensemble_agreement=0.60,
        citation_coverage=0.92,
        composite=0.82,
        components={},
    )
    decision = gate.decide(confidence)
    assert decision.should_answer is False
    assert "ensemble_agreement" in decision.reason
    assert "0.60" in decision.reason
