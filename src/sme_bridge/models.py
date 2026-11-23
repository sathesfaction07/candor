from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class ConfidenceBreakdown(BaseModel):
    # ensemble_agreement is None when it could not be MEASURED (fewer than two models
    # answered). Storing 0.0 in that case would claim the models disagreed, which is a
    # different and much stronger statement than "we do not know".
    retrieval_support: float
    ensemble_agreement: float | None
    citation_coverage: float
    composite: float
    # composite is a raw evidence score, NOT a probability. It only becomes a probability
    # once mapped through a fitted calibrator, at which point calibrated is set True.
    calibrated: bool = False
    components: dict = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page: int | None
    quoted_span: str


class Answer(BaseModel):
    text: str
    confidence: ConfidenceBreakdown
    citations: list[Citation]
    answered: bool
    model_versions: dict = Field(default_factory=dict)


class KnowledgeGap(BaseModel):
    gap_id: str
    query: str
    asked_at: datetime
    confidence: ConfidenceBreakdown
    sme_question: str
    retrieved_chunk_ids: list[str]
    status: Literal["open", "answered", "indexed"]
    sme_answer: str | None = None
    answered_at: datetime | None = None
    answered_by: str | None = None
    indexed_at: datetime | None = None
