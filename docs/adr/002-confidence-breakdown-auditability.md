# ADR 002: Confidence Breakdown Auditability

## Status

Accepted

## Context

The defining feature of Candor is knowing what it doesn't know. A single confidence float would be opaque: users couldn't tell whether low confidence came from poor retrieval, model disagreement, or weak citations.

## Decision

`ConfidenceBreakdown` is a structured model with:
- `retrieval_support`: how well retrieved chunks match the query
- `ensemble_agreement`: consensus across multiple model answers
- `citation_coverage`: whether the answer is grounded in retrieved text
- `composite`: the final combined score
- `components`: dict for intermediate values and debugging

Every score is traceable. No fabricated numbers.

## Consequences

- Confidence refusal can be explained to users: "The documents I found don't strongly support this answer."
- Knowledge gap records include the breakdown, so SMEs see why the system refused
- Future work: calibration of the composite formula against ground-truth gaps
- Cost: more complex than a single float, but opacity is worse than complexity
