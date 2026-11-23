import pytest

from sme_bridge.core.confidence import compute_confidence


def test_renormalization_when_agreement_unavailable():
    confidence = compute_confidence(
        retrieval_support=0.8,
        citation_coverage=0.7,
        ensemble_agreement=None,
    )
    assert confidence.composite == pytest.approx(0.75)
    assert confidence.ensemble_agreement is None
    available = confidence.components["components_available"]
    assert "retrieval_support" in available
    assert "citation_coverage" in available
    assert "ensemble_agreement" not in available
    weights = confidence.components["weights_used"]
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert weights["retrieval_support"] == weights["citation_coverage"] == 0.5


def test_unmeasured_agreement_scores_higher_than_measured_disagreement():
    unmeasured = compute_confidence(0.8, 0.7, None)
    disagreed = compute_confidence(0.8, 0.7, 0.0)
    assert unmeasured.composite > disagreed.composite


def test_composite_is_bounded_by_weakest_and_strongest_components():
    confidence = compute_confidence(
        retrieval_support=0.5,
        citation_coverage=0.9,
        ensemble_agreement=0.8,
    )
    assert min(0.5, 0.9, 0.8) <= confidence.composite <= max(0.5, 0.9, 0.8)
    assert confidence.composite == pytest.approx(0.73)


def test_all_zero_inputs():
    confidence = compute_confidence(
        retrieval_support=0.0,
        citation_coverage=0.0,
        ensemble_agreement=0.0,
    )
    assert confidence.composite == 0.0


def test_all_perfect_inputs():
    confidence = compute_confidence(
        retrieval_support=1.0,
        citation_coverage=1.0,
        ensemble_agreement=1.0,
    )
    assert confidence.composite == 1.0


def test_composite_is_never_marked_calibrated():
    confidence = compute_confidence(0.9, 0.9, 0.9)
    assert confidence.calibrated is False


def test_two_components_only():
    confidence = compute_confidence(
        retrieval_support=0.6,
        citation_coverage=0.8,
        ensemble_agreement=None,
    )
    assert confidence.composite == pytest.approx(0.7)
    assert len(confidence.components["components_available"]) == 2
