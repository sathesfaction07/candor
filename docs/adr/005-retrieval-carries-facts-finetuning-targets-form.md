# ADR 005: Retrieval Carries Factual Recall; Fine-Tuning Targets Form and Phrasing

## Status

Accepted

## Context

Two mechanisms exist for incorporating domain knowledge: retrieval-augmented generation (RAG) and fine-tuning the language model. Both have costs and constraints.

RAG retrieves relevant text at query time and passes it to the model as context. The model answers based on the provided text. When a document changes, re-indexing makes the new text available immediately.

Fine-tuning trains the model's weights on domain-specific examples. The model internalises patterns and phrasing. When a document changes, the model must be retrained, which is slower and more expensive than re-indexing.

## Decision

Retrieval carries factual recall. Fine-tuning targets form and phrasing.

Specifically:
- **Facts that change** (procedures, policies, regulatory text) live in indexed documents and are retrieved at query time. When a document updates, it is re-indexed and becomes available the same day.
- **Form and phrasing** (how to structure an answer, domain-specific terminology, tone) are targets for fine-tuning. These change more slowly and benefit from the model learning patterns across many examples.

The decisive argument is correctness latency: a fine-tuned model cannot be corrected the same day a critical document changes. Retrieval can. For a system whose defining feature is knowing when it doesn't know, giving stale weights precedence over current documents is backwards.

## Consequences

- Document updates do not require model retraining. The operational path is: update document, run ingestion, re-index, done.
- Fine-tuning is optional. The system works with a base model plus retrieval. Fine-tuning improves quality (phrasing, structure) but is not a prerequisite.
- The fine-tuning path in `learning/sagemaker/finetune.py` exists to improve answer quality when resources permit. It exports training data derived from SME-answered gaps (the expert's answer becomes the gold target) and submits a SageMaker job.
- Model versioning and rollback are built in: the previous version is retained when a new fine-tuned model is deployed, so a regression can be reverted without re-running the entire training job.

## What Would Make Us Revisit This

- A fine-tuning method becomes available that is fast enough to run on every document update (example: parameter-efficient fine-tuning that completes in minutes, not hours).
- Evidence emerges that fine-tuned models systematically outperform retrieval even on day-of-change queries (contradicting the stale-weights argument).
- The domain shifts to one where facts do not change (example: historical analysis, where the corpus is static).
