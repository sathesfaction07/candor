import tempfile
from pathlib import Path

import pytest

from sme_bridge.learning.classifier import AbstentionClassifier, CalibratedGate
from sme_bridge.models import ConfidenceBreakdown


def test_classifier_fit_on_separable_data():
    classifier = AbstentionClassifier()

    training = []
    for _i in range(50):
        bd = ConfidenceBreakdown(
            retrieval_support=0.9, ensemble_agreement=0.95, citation_coverage=0.9, composite=0.9
        )
        training.append((bd, True))

    for _i in range(50):
        bd = ConfidenceBreakdown(
            retrieval_support=0.2, ensemble_agreement=0.3, citation_coverage=0.25, composite=0.25
        )
        training.append((bd, False))

    classifier.fit(training)

    high_bd = ConfidenceBreakdown(
        retrieval_support=0.85, ensemble_agreement=0.9, citation_coverage=0.88, composite=0.88
    )
    low_bd = ConfidenceBreakdown(
        retrieval_support=0.15, ensemble_agreement=0.2, citation_coverage=0.18, composite=0.18
    )

    high_prob = classifier.predict_proba(high_bd)
    low_prob = classifier.predict_proba(low_bd)

    assert high_prob > low_prob
    assert 0.0 <= high_prob <= 1.0
    assert 0.0 <= low_prob <= 1.0


def test_classifier_agreement_measured_feature():
    classifier = AbstentionClassifier()

    training = []
    for _i in range(30):
        bd = ConfidenceBreakdown(
            retrieval_support=0.8, ensemble_agreement=0.85, citation_coverage=0.8, composite=0.8
        )
        training.append((bd, True))

    for _i in range(30):
        bd = ConfidenceBreakdown(
            retrieval_support=0.8, ensemble_agreement=None, citation_coverage=0.8, composite=0.8
        )
        training.append((bd, False))

    classifier.fit(training)

    with_agreement = ConfidenceBreakdown(
        retrieval_support=0.8, ensemble_agreement=0.85, citation_coverage=0.8, composite=0.8
    )
    without_agreement = ConfidenceBreakdown(
        retrieval_support=0.8, ensemble_agreement=None, citation_coverage=0.8, composite=0.8
    )

    prob_with = classifier.predict_proba(with_agreement)
    prob_without = classifier.predict_proba(without_agreement)

    assert abs(prob_with - prob_without) > 0.01


def test_classifier_save_load_roundtrip():
    classifier = AbstentionClassifier()

    training = []
    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.9, ensemble_agreement=0.9, citation_coverage=0.9, composite=0.9
        )
        training.append((bd, True))

    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.3, ensemble_agreement=0.3, citation_coverage=0.3, composite=0.3
        )
        training.append((bd, False))

    classifier.fit(training)

    test_bd = ConfidenceBreakdown(
        retrieval_support=0.85, ensemble_agreement=0.88, citation_coverage=0.87, composite=0.87
    )
    original_prob = classifier.predict_proba(test_bd)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "classifier.pkl"
        classifier.save(str(path))

        loaded = AbstentionClassifier.load(str(path))
        loaded_prob = loaded.predict_proba(test_bd)

        assert abs(original_prob - loaded_prob) < 1e-6


def test_classifier_unfitted_predict_raises():
    classifier = AbstentionClassifier()
    bd = ConfidenceBreakdown(
        retrieval_support=0.8, ensemble_agreement=0.8, citation_coverage=0.8, composite=0.8
    )

    with pytest.raises(RuntimeError, match="not been fitted"):
        classifier.predict_proba(bd)


def test_classifier_unfitted_save_raises():
    classifier = AbstentionClassifier()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "classifier.pkl"
        with pytest.raises(RuntimeError, match="unfitted"):
            classifier.save(str(path))


def test_classifier_empty_examples_raises():
    classifier = AbstentionClassifier()

    with pytest.raises(ValueError, match="zero examples"):
        classifier.fit([])


def test_calibrated_gate_sets_calibrated_flag():
    classifier = AbstentionClassifier()

    training = []
    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.9, ensemble_agreement=0.9, citation_coverage=0.9, composite=0.9
        )
        training.append((bd, True))

    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.3, ensemble_agreement=0.3, citation_coverage=0.3, composite=0.3
        )
        training.append((bd, False))

    classifier.fit(training)

    gate = CalibratedGate(classifier, threshold=0.5)

    test_bd = ConfidenceBreakdown(
        retrieval_support=0.85, ensemble_agreement=0.88, citation_coverage=0.87, composite=0.87
    )

    decision = gate.decide(test_bd)
    assert isinstance(decision.should_answer, bool)


def test_calibrated_gate_without_fitted_classifier_raises():
    gate = CalibratedGate(None, threshold=0.9)

    bd = ConfidenceBreakdown(
        retrieval_support=0.8, ensemble_agreement=0.8, citation_coverage=0.8, composite=0.8
    )

    with pytest.raises(RuntimeError, match="fitted classifier"):
        gate.decide(bd)


def test_calibrated_gate_reason_cites_probability():
    classifier = AbstentionClassifier()

    training = []
    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.9, ensemble_agreement=0.9, citation_coverage=0.9, composite=0.9
        )
        training.append((bd, True))

    for _i in range(20):
        bd = ConfidenceBreakdown(
            retrieval_support=0.3, ensemble_agreement=0.3, citation_coverage=0.3, composite=0.3
        )
        training.append((bd, False))

    classifier.fit(training)

    gate = CalibratedGate(classifier, threshold=0.5)

    test_bd = ConfidenceBreakdown(
        retrieval_support=0.85, ensemble_agreement=0.88, citation_coverage=0.87, composite=0.87
    )

    decision = gate.decide(test_bd)
    assert "probability" in decision.reason.lower() or "calibrated" in decision.reason.lower()
    assert any(char.isdigit() for char in decision.reason)
