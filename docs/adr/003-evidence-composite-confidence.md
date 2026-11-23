# ADR 003: Confidence as Measurable Evidence, Not LLM Self-Report

## Status

Accepted

## Context

The defining feature of SME Bridge is knowing when to abstain. A confidence mechanism that permits the system to answer incorrectly with high confidence undermines the entire design. LLM self-reported confidence is therefore not an option: models have no access to ground truth and tend to report confidence based on fluency rather than correctness.

A prior internal implementation computed confidence by starting at a hardcoded 0.85 and adding small bonuses for response length and for the presence of bullet points and bold formatting characters. This measured whether a response looked well-formatted rather than whether it was correct. Under this scheme, a confidently wrong answer with good formatting would pass the gate while a terse correct answer might not.

## Decision

Confidence is a composite of three measured evidence signals, each contributing a weight that is renormalised over whichever signals are actually available:

1. **Retrieval support** (weight 0.35): TF-IDF cosine similarity between the query and the best-matching retrieved chunk. Measures whether relevant documents were found.
2. **Citation coverage** (weight 0.35): the fraction of answer sentences that are supported by retrieved text spans. Measures whether the answer is grounded in the documents.
3. **Ensemble agreement** (weight 0.30): TF-IDF cosine similarity between answers from multiple models. Measures consensus. This signal is `None` when fewer than two models answered, and the weight is withheld in that case.

The composite is computed as a weighted mean over only the signals that were measured. If ensemble agreement is unavailable, the retrieval and citation weights are renormalised to sum to 1.0.

LLM self-reported confidence is explicitly rejected. The model is never asked "how confident are you?" or given a scale to rate itself.

## Consequences

- Confidence scores are auditable: `ConfidenceBreakdown.components` exposes which signals contributed and with what weights.
- The system can refuse to answer even when the LLM produces fluent text, because fluency and evidence are separate.
- Adding a new evidence signal requires updating the weight constants and the composite computation, but the formula is centralised in `confidence.py`.
- The composite is not calibrated to ground truth until training data exists, which is addressed in ADR 004.

## What Would Make Us Revisit This

- Evidence emerges that LLM self-report becomes reliable (example: a model trained explicitly to predict its own correctness probability, with adversarial validation).
- A fundamentally different evidence signal becomes available (example: external fact-checking API, human feedback loop with confidence labels).
- The weight distribution proves systematically wrong after deployment (example: retrieval support dominates to the point that citation coverage adds no information).
