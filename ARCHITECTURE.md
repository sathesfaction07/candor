# Architecture

SME Bridge is a question-answering system that knows when to refuse. The defining flow is: ask a question, get either a grounded answer or a precise question to ask a human expert, fold the expert's answer back in, and answer that question successfully the next time.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INGESTION                                                           │
│  documents → chunking → index                                       │
│  (PDF/text/markdown with page preservation, sentence-aware overlap) │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ QUERY                                                               │
│  user question → retrieval (keyword/chromadb/bedrock) → top_k      │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ENSEMBLE (if configured)                                            │
│  multiple models answer → agreement measured via TF-IDF cosine      │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CITATION EXTRACTION                                                 │
│  answer text → sentence splitting → span matching → coverage score │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE SCORING                                                  │
│  retrieval_support (TF-IDF) + citation_coverage + ensemble_agreement│
│  → weighted composite (not a probability)                           │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GATE DECISION                                                       │
│  ThresholdGate: raw composite ≥ evidence_gate (0.55) → answer      │
│  CalibratedGate: calibrated probability ≥ confidence_gate (0.90)   │
│  → answer | abstain                                                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────┐           ┌─────────────┐
    │ ANSWER  │           │ ABSTAIN     │
    │ return  │           │ (low conf)  │
    └─────────┘           └──────┬──────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │ QUESTION GENERATION        │
                    │ "What is X?" → precise Q   │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │ KNOWLEDGE GAP LEDGER       │
                    │ record gap (query, Q, etc) │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │ SME ANSWERS (external)     │
                    │ human provides answer      │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │ INGEST SME ANSWER          │
                    │ engine.ingest_sme_answer() │
                    │ → mark indexed, reindex    │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │ RETRAIN (if data available)│
                    │ calibrate gate classifier  │
                    └────────────────────────────┘
```

## Module Ownership

### Ingestion (`ingest/`)
- `chunking.py`: splits documents into overlapping sentence-aware chunks
- `pdf.py`: extracts text from PDF with page number preservation (PyMuPDF)
- `text.py`: loads plain text and markdown

**Shape**: Documents become `Chunk` objects with `chunk_id`, `doc_id`, `text`, optional `page`, and metadata.

### Retrieval (`retrieval/`)
Three backends behind a single `RetrievalBackend` ABC:
- `keyword.py`: local TF-IDF with scikit-learn, zero dependencies, zero cloud
- `chromadb_backend.py`: local vector search, lazy imports, optional `[vector]` extra
- `bedrock_kb.py`: AWS-managed vector store via boto3

**Shape**: Query string → list of `RetrievedChunk(chunk, score)` ranked by relevance.

### Core Reasoning (`core/`)
- `engine.py`: top-level `Engine.answer()` orchestrates the flow
- `ensemble.py`: queries multiple models, measures agreement via TF-IDF cosine between their answers
- `citations.py`: finds which sentences in the answer are supported by retrieved text, returns `Citation` list and coverage score
- `confidence.py`: computes `ConfidenceBreakdown.composite` as a renormalised weighted mean (0.35/0.35/0.30) over measured signals
- `gate.py`: `ThresholdGate` (raw composite) and `CalibratedGate` (fitted probability) decide whether to answer
- `question.py`: generates a precise question for the SME when abstaining
- `ledger.py`: writes `KnowledgeGap` records to JSONL (or an extensible backend)
- `prompts.py`: system and user prompts for the LLM calls

**Key types**:
- `ConfidenceBreakdown`: three evidence scores (retrieval support, citation coverage, ensemble agreement), a composite, and a `components` dict for auditability. `ensemble_agreement` is `None` when unmeasured (fewer than two models), never 0.0.
- `Answer`: text, confidence breakdown, citations, `answered` bool, model versions.
- `KnowledgeGap`: query, confidence, generated SME question, status (open/answered/indexed), SME answer once provided.

### Learning (`learning/`)
- `calibration.py`: computes ECE, Brier score, and implied accuracy at the gate from evaluation examples
- `classifier.py`: `AbstentionClassifier` fits logistic regression over evidence features to predict correctness probability; `CalibratedGate` wraps it
- `training_data.py`: converts gap ledger and evaluation data into training examples
- `sagemaker/finetune.py`: exports datasets to S3, submits SageMaker fine-tune jobs, polls status, versions models with rollback
- `sagemaker/calibration_job.py`: trains the abstention classifier as a SageMaker job, registers the artefact

**Critical constraint**: the ledger records only negatives (every gap is a refused answer), so the classifier cannot be fitted until accepted-answer feedback is collected. The shipped gate is therefore uncalibrated.

### Configuration (`config.py`)
Two thresholds with distinct meanings:
- `evidence_gate` (0.55): threshold for the **raw composite** used by `ThresholdGate`. This is not a probability. Retrieval support is a TF-IDF cosine around 0.5-0.7 even for strong matches, so a 0.90 threshold here would abstain on everything.
- `confidence_gate` (0.90): threshold for the **calibrated probability** produced by `CalibratedGate`. This can honestly mean "90% of allowed answers are correct" after calibration.

### Application Layer
- `app.py`: FastAPI server for HTTP queries
- `cli.py`: command-line interface for ingestion, queries, and SME answer ingestion

## Design Decisions

The core architecture follows from two constraints:

1. **Abstention is the feature**: low-confidence answers do more harm than "I don't know, let me ask an expert." This requires measurable evidence, not LLM self-report.
2. **Learning from production**: every refused answer becomes a training case once the SME responds, so the system improves with use.

Retrieval backends are swappable because no single deployment environment is universal: local development needs zero-cloud operation, cloud deployments benefit from vector search, and hybrid scenarios exist.

The confidence composite is explicitly not a probability before calibration, because measuring raw evidence (cosine similarities, span coverage) produces scores on scales that do not sum to one. Logistic regression over those features is what turns evidence into probability.

The dual-threshold design exists because a single threshold would force a choice: either use the raw composite and accept that 0.90 means "this much evidence" rather than "90% correct," or require calibration before the system can run at all. Shipping both lets the system operate immediately (threshold gate) while providing an honest calibrated mode once training data exists (calibrated gate).
