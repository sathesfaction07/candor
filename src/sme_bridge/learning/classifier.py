import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from sme_bridge.core.gate import Gate, GateDecision
from sme_bridge.models import ConfidenceBreakdown

logger = logging.getLogger(__name__)


class AbstentionClassifier:
    def __init__(self):
        self._model: LogisticRegression | None = None
        self._fitted = False

    def _extract_features(self, breakdown: ConfidenceBreakdown) -> np.ndarray:
        retrieval = breakdown.retrieval_support
        citation = breakdown.citation_coverage
        agreement = (
            breakdown.ensemble_agreement
            if breakdown.ensemble_agreement is not None
            else 0.0
        )
        agreement_measured = 1.0 if breakdown.ensemble_agreement is not None else 0.0
        return np.array([[retrieval, citation, agreement, agreement_measured]])

    def fit(self, examples: list[tuple[ConfidenceBreakdown, bool]]) -> None:
        if not examples:
            raise ValueError("Cannot fit classifier with zero examples")

        feature_matrix = np.vstack([self._extract_features(bd) for bd, _ in examples])
        y = np.array([1 if correct else 0 for _, correct in examples])

        # Logistic regression cannot fit a single class, and the ledger alone produces only
        # negatives: every SME-answered gap is by definition a case the system was unsure
        # about. Positives have to come from answers a user ACCEPTED. Fail with that
        # explanation rather than letting the solver raise something opaque.
        if len(set(y.tolist())) < 2:
            only = "correct" if y[0] == 1 else "incorrect"
            raise ValueError(
                f"Need both classes to fit, but all {len(y)} examples are labelled {only}. "
                "Gap-derived examples are all negatives by construction, so accepted-answer "
                "feedback is required before the abstention classifier can be trained."
            )

        self._model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._model.fit(feature_matrix, y)

        self._fitted = True
        logger.info(f"Fitted abstention classifier on {len(examples)} examples")

    def predict_proba(self, breakdown: ConfidenceBreakdown) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError(
                "Classifier has not been fitted. Call fit() with training examples first."
            )
        feature_matrix = self._extract_features(breakdown)
        proba = self._model.predict_proba(feature_matrix)[0, 1]
        return float(proba)

    def save(self, path: str) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Cannot save an unfitted classifier")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "fitted": self._fitted}, p)
        logger.info(f"Saved classifier to {path}")

    @classmethod
    def load(cls, path: str) -> "AbstentionClassifier":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Classifier file not found: {path}")
        data = joblib.load(p)
        classifier = cls()
        classifier._model = data["model"]
        classifier._fitted = data["fitted"]
        logger.info(f"Loaded classifier from {path}")
        return classifier


class CalibratedGate(Gate):
    def __init__(self, classifier: AbstentionClassifier | None, threshold: float = 0.90):
        self.classifier = classifier
        self.threshold = threshold

    def decide(self, confidence: ConfidenceBreakdown) -> GateDecision:
        if self.classifier is None or not self.classifier._fitted:
            raise RuntimeError(
                "CalibratedGate requires a fitted classifier. "
                "Provide a trained AbstentionClassifier or use ThresholdGate instead."
            )

        proba = self.classifier.predict_proba(confidence)

        if proba >= self.threshold:
            return GateDecision(
                should_answer=True,
                reason=(
                    f"Calibrated probability {proba:.2f} meets threshold {self.threshold:.2f}"
                ),
                probability=proba,
                calibrated=True,
            )

        return GateDecision(
            should_answer=False,
            reason=(
                f"Calibrated probability {proba:.2f} below threshold {self.threshold:.2f}"
            ),
            probability=proba,
            calibrated=True,
        )
