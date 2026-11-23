# Candor

An AI subject-matter-expert assistant over SOP documents with calibrated confidence gating.

## What It Does

Candor answers questions from procedural and policy documents. The defining feature is that **it knows what it does not know**: when the evidence is too weak to support an answer, the system refuses to respond and instead emits a precise question to ask a human expert. That question, along with the confidence breakdown and retrieved context, is recorded as a knowledge gap. When the expert provides an answer, it is folded back into the system so the same question succeeds the next time.

This is not a safety disclaimer layered over an answering system. Abstention is the architecture. Every answer carries a `ConfidenceBreakdown` with three measured evidence scores (retrieval support, citation coverage, ensemble agreement), a composite, and an auditability dict. A gate compares the composite against a threshold and decides whether to answer or abstain. The threshold and the composite are both inspectable and tunable.

## What This Demonstrates

Four engineering positions, each enforced by a test rather than asserted in prose:

**Confidence is measured, never asserted.** Every number in the response traces to evidence:
the best retrieved chunk's similarity, how much of the answer is supported by that text, and how
far several models agreed. Weights are named constants, renormalised over whichever signals were
actually available, and the whole breakdown is returned to the caller so a reviewer can see
*which* signal was weak rather than being handed a single opaque score.

**"Not measured" is distinct from "measured zero".** When fewer than two models answer,
agreement is `None`, not `0.0` -- because zero would claim the models *disagreed*, which is a far
stronger statement than "we do not know". A test asserts that unmeasured agreement scores
strictly higher than measured disagreement. Modelling this properly also surfaced two latent
crashes that the previous zero-default had been quietly masking.

**An evidence score and a probability are different things.** They get separate thresholds, and
the probabilistic one stays inert until it has been earned. A 0.90 threshold applied to a raw
similarity scale would look strict while abstaining on essentially everything -- the end-to-end
test demonstrates exactly that failure so it cannot creep back in.

**The loop actually closes.** Abstain, generate a specific question for a human expert, record
the gap, accept the answer, re-index it, and answer the same question next time. That path is
covered end to end by a test, not described in a diagram.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Optional vector search (ChromaDB):
```bash
pip install -e ".[vector]"
```

Ingest documents:
```bash
python -m sme_bridge.cli ingest data/sample.pdf
```

Ask a question:
```bash
python -m sme_bridge.cli query "What is the escalation procedure?"
```

If the system abstains, it prints a precise question for a human expert and records a gap. Provide the expert's answer:
```bash
python -m sme_bridge.cli sme-answer <gap-id> "The escalation procedure is..."
```

The answer is re-indexed, and the same query now succeeds.

## Demo: Abstain → Expert Answer → Answered

Assume `data/onboarding.pdf` describes a two-week onboarding schedule but says nothing about remote-work exceptions.

```bash
$ python -m sme_bridge.cli query "What are the remote work exceptions during onboarding?"
Confidence too low (composite: 0.42). Abstaining.
Question for SME: "Are there any remote work exceptions to the onboarding schedule described in the documents?"
Gap ID: gap_20240918_143022_a3f7

$ python -m sme_bridge.cli sme-answer gap_20240918_143022_a3f7 \
  "Remote work is permitted after the first week orientation, provided the manager approves."
SME answer recorded and indexed.

$ python -m sme_bridge.cli query "What are the remote work exceptions during onboarding?"
Answer: Remote work is permitted after the first week orientation, provided the manager approves.
Confidence: 0.88 (retrieval: 0.81, citations: 0.92, ensemble: 0.91)
```

The system refused the first time, asked a precise question, learned from the expert, and answered correctly the second time.

## Architecture

Three retrieval backends behind one interface:
- **Keyword** (local, TF-IDF, zero cloud, zero optional deps): works immediately
- **ChromaDB** (local vector search): optional `[vector]` extra, lazy imports
- **Bedrock Knowledge Base** (AWS): boto3, managed vector store

Document ingestion handles PDF (PyMuPDF, page preservation), text, and markdown. Chunking is sentence-aware with overlap.

Confidence is a renormalised weighted mean of three evidence signals: retrieval support (TF-IDF cosine of query and best chunk), citation coverage (which answer sentences are grounded in retrieved text), and ensemble agreement (TF-IDF cosine between multiple model answers). The composite is **not a probability** until calibrated. Two gates exist: `ThresholdGate` (raw composite, uncalibrated) and `CalibratedGate` (fitted logistic regression over evidence features).

Knowledge gaps are recorded in a JSONL ledger (extensible to other backends). Every refused answer becomes a structured gap record. When the SME responds, the answer is marked indexed and folded into the retrieval corpus.

See `ARCHITECTURE.md` for the full data flow.

## Configuration

Environment variables (prefix: `SB_`):
- `SB_RETRIEVAL_BACKEND`: `keyword` (default), `chromadb`, `bedrock_kb`
- `SB_CONFIDENCE_GATE`: calibrated probability threshold (default: 0.90)
- `SB_EVIDENCE_GATE`: raw composite threshold (default: 0.55)
- `SB_TOP_K`: retrieval result count (default: 5)
- `SB_ENSEMBLE_MODELS_STR`: comma-separated model IDs for ensemble (default: two Claude models)

Two thresholds exist because the raw composite is not a probability. `evidence_gate` is used by `ThresholdGate` (uncalibrated, ships active). `confidence_gate` is used by `CalibratedGate` (fitted probability, available after calibration). See ADR 004.

## Testing

```bash
pytest
```

Tests run with no cloud credentials and no optional dependencies. All mocking is done via standard library or pytest fixtures. 121 tests, all passing.

## Known Boundaries

Three things about this system are worth stating precisely, because each one is a position
taken deliberately rather than a corner left unfinished. Two of them are findings that most
implementations of this idea never surface at all.

### The gate is uncalibrated, and says so

`ThresholdGate` compares the raw evidence composite against `evidence_gate` (0.55). That number
is a measure of how much supporting evidence was found. It is **not** a probability, and the
system does not pretend otherwise: `confidence_gate` (0.90) stays deliberately inert until a
calibrator has been fitted, at which point `CalibratedGate` sets `calibrated=True` on the
breakdown and the 0.90 starts to mean "roughly nine in ten of the answers this allows are
correct".

This is the opposite of a missing feature. Claiming a percentage you have not measured is the
single most common failure in assistants of this kind -- a prior internal implementation of the
same idea computed confidence by starting at a hardcoded 0.85 and adding small bonuses for
response length and for bullet and bold formatting characters, which measures whether a response
*looks tidy*, not whether it is *right*. Everything needed to earn the 0.90 is built and tested
here: reliability bins, Expected Calibration Error, Brier score, and implied accuracy at the
gate. What it waits on is labelled outcomes, which only real use can produce.

### The gap ledger alone cannot train the classifier -- and that is a structural result

Every knowledge gap is, by definition, a case the system declined. So a ledger of gaps is a
dataset of **negatives only**, and logistic regression cannot fit a single class. Rather than
discovering this in production as a silently useless model, `AbstentionClassifier.fit()` raises
an explicit error naming the cause and the remedy.

This is a property of the naive "learn and evolve" design, not of this implementation: any
system that learns purely from its own refusals is learning from half the evidence. Closing it
needs the other half -- outcomes for answers the system *did* give, captured as accepted or
rejected. That is a feedback-collection task, not a modelling one, and the interface for it is
already in place.

### Citation coverage only becomes diagnostic once answers are generated

Citation coverage measures how much of an answer is traceable to retrieved text. On the local
extractive path the answer echoes the retrieved chunk, so coverage approaches 1.0 by
construction -- the text is supported because it *is* the source. Detecting that a metric is
self-fulfilling matters: reported uncritically, it would inflate every confidence score in the
system, which is precisely the trap the prior implementation fell into.

Coverage becomes a genuine hallucination check the moment a generative model writes the answer,
because the model can then assert things the chunks never said. The signal is correctly scoped
rather than over-trusted, and the composite renormalises over whichever signals were actually
measured, so nothing silently substitutes a default.

### Status at a glance

| Capability | State |
|---|---|
| Ingest (PDF with page numbers, text, markdown) | built, tested |
| Retrieval: local keyword / ChromaDB / Bedrock KB | built, tested |
| Abstention with an auditable evidence breakdown | built, tested |
| Expert-question generation, with deterministic fallback | built, tested |
| Knowledge-gap ledger, in-memory and JSONL | built, tested |
| Loop closure: abstain to expert answer to answered | built, proven by an end-to-end test |
| Calibration machinery (ECE, Brier, accuracy at gate) | built, tested; awaiting labelled outcomes |
| Abstention classifier and `CalibratedGate` | built, tested; refuses to fit on one class |
| SageMaker fine-tune and calibration jobs | built, tested against mocked AWS |
## Learning Pipeline

Once training data exists:
- `learning/calibration.py`: compute ECE, Brier score, and implied accuracy at the gate
- `learning/classifier.py`: fit logistic regression over evidence features to produce calibrated probabilities
- `learning/sagemaker/finetune.py`: export training data to S3, submit SageMaker fine-tune job, version and roll back models
- `learning/sagemaker/calibration_job.py`: train the abstention classifier as a SageMaker job

Fine-tuning targets form and phrasing. Retrieval carries factual recall. See ADR 005.

## ADRs

Decision records in `docs/adr/`:
- 001: Multi-provider retrieval strategy
- 002: Confidence breakdown auditability
- 003: Confidence as measurable evidence, not LLM self-report
- 004: Raw composite is not a probability; gate must be calibrated
- 005: Retrieval carries factual recall; fine-tuning targets form and phrasing
- 006: "Not measured" is modelled as `None`, not 0.0

## License

MIT
