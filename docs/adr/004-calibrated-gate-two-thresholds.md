# ADR 004: Raw Composite Is Not a Probability; Gate Must Be Calibrated

## Status

Accepted

## Context

The evidence composite ranges from 0.0 to 1.0, but it is not a probability. Retrieval support is a TF-IDF cosine similarity that sits around 0.5-0.7 even for strong matches. Citation coverage can be near 1.0 for an extractive answer but much lower for a generative one. Ensemble agreement, when measured, is a cosine between answer texts.

Setting a threshold of 0.90 on this raw composite would abstain on essentially everything, because the scale does not match the intuitive meaning of "90% confident this is correct." A 0.90 threshold only means something if the score is a calibrated probability.

## Decision

Two thresholds with distinct meanings:

1. **`evidence_gate`** (default 0.55): threshold for the raw composite, used by `ThresholdGate`. This is not a probability. It means "this much evidence was gathered." The default is set to allow answers when retrieval and citations are reasonable, even if not overwhelming.

2. **`confidence_gate`** (default 0.90): threshold for the calibrated probability, used by `CalibratedGate`. This can honestly claim "roughly 90% of the answers I allow are correct" after calibration.

Calibration is performed by `AbstentionClassifier` in `learning/classifier.py`, which fits logistic regression over the evidence features (retrieval support, citation coverage, ensemble agreement if available, plus a boolean flag for whether agreement was measured). The fitted model maps evidence to probability.

Calibration quality is measured via three metrics in `learning/calibration.py`:
- **ECE (Expected Calibration Error)**: are predictions well-calibrated across confidence bins?
- **Brier score**: squared difference between predicted probability and actual outcome.
- **Implied accuracy at the gate**: of the answers the gate would have allowed, what fraction were actually correct. This is the number that makes or breaks a 0.90 claim.

The system ships with `ThresholdGate` active. `CalibratedGate` becomes available once training data exists and `calibrate` has been run.

## Consequences

- The system can operate immediately with an uncalibrated gate, using the evidence composite as a heuristic.
- The 0.90 confidence claim is only honest after calibration. Until then, the system uses `evidence_gate` and makes no probabilistic claims.
- Calibration requires labelled data: examples where the answer was correct or incorrect. The knowledge gap ledger alone is insufficient (see ADR 004's companion issue: all gaps are negatives).
- Once calibrated, `ConfidenceBreakdown.calibrated` is set to `True` and `components["calibrated_probability"]` holds the fitted probability. Downstream systems can check this flag to determine whether the score has probabilistic meaning.

## What Would Make Us Revisit This

- The uncalibrated evidence composite proves systematically miscalibrated in one direction (example: 0.55 threshold allows answers that are wrong 80% of the time, meaning the heuristic scale is inverted or compressed).
- A simpler calibration method becomes available that does not require logistic regression (example: isotonic regression, Platt scaling, or histogram binning with fewer data requirements).
- Calibration becomes feasible without labelled positives (example: a semi-supervised method that learns from gap records plus unlabelled queries assumed to be mostly correct).
