# ADR 006: "Not Measured" Is Modelled as `None`, Not 0.0

## Status

Accepted

## Context

`ConfidenceBreakdown.ensemble_agreement` represents consensus across multiple model answers. When only one model is configured or fewer than two models successfully answer, agreement cannot be measured.

Storing `0.0` in this case is a lie: zero means the models disagreed completely (cosine similarity of zero). Storing `None` is honest: the measurement was not taken.

The original implementation used `0.0` as the default. This masked two latent crashes in question generation, which had been written to assume agreement was always a float and called `.format()` on it. When the field became `None`, both crashes surfaced immediately and were fixed.

## Decision

`ensemble_agreement` is `None` when fewer than two models answered. It is never `0.0` unless the models actually disagreed.

The confidence computation in `confidence.py` checks for `None` and withholds the ensemble weight in that case, renormalising the retrieval and citation weights to sum to 1.0. No fabricated score is injected.

In `gate.py`, `ThresholdGate` explicitly excludes `None` values from the weakest-component calculation. A signal that was not measured cannot be reported as the weakest.

Tests assert that unmeasured agreement (modelled as `None`) scores strictly higher than measured disagreement (modelled as a low float), because the absence of a negative signal is better than a confirmed negative signal.

## Consequences

- The type signature `ensemble_agreement: float | None` is honest. Code that accesses it must handle the `None` case.
- Two latent defects were surfaced and fixed: question generation had been formatting the agreement score into a string without checking whether it was `None`. Both are now guarded.
- Adding a new optional signal to the composite requires the same discipline: model the absence as `None`, withhold the weight, do not fabricate a default.
- The test suite enforces this: `test_confidence_none_agreement_beats_measured_disagreement` asserts that `None` is scored as more favorable than `0.05`, which would not hold if `None` were treated as `0.0`.

## What Would Make Us Revisit This

- Python typing evolves to provide a clearer way to express "measured value or absent" (example: a `Measured[T]` generic that forces explicit unwrapping, similar to Rust's `Option`).
- The composite computation changes such that withholding a weight is no longer the right treatment for an unmeasured signal (example: if we switch to a probabilistic graphical model where missing variables must be marginalised rather than ignored).
- Evidence emerges that treating unmeasured agreement as `None` causes systematic bias (example: single-model runs consistently score higher than multi-model runs even when the single model is less accurate).
