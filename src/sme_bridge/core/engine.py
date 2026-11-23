import logging
import uuid
from collections.abc import Callable
from datetime import datetime

from sme_bridge.core.citations import extract_citations
from sme_bridge.core.confidence import compute_confidence
from sme_bridge.core.ensemble import EnsembleClient
from sme_bridge.core.gate import Gate, ThresholdGate
from sme_bridge.core.ledger import GapLedger
from sme_bridge.core.question import generate_sme_question
from sme_bridge.models import Answer, Chunk, ConfidenceBreakdown, KnowledgeGap
from sme_bridge.retrieval.base import RetrievalBackend

logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        retrieval: RetrievalBackend,
        ledger: GapLedger,
        ensemble: EnsembleClient | None = None,
        gate: Gate | None = None,
        question_generator: Callable[[str], str] | None = None,
    ):
        self.retrieval = retrieval
        self.ledger = ledger
        self.ensemble = ensemble
        self.gate = gate or ThresholdGate()
        self.question_generator = question_generator

    def answer(self, query: str, top_k: int = 5) -> Answer:
        retrieved = self.retrieval.search(query, top_k=top_k)
        if not retrieved:
            return self._abstain_no_results(query)
        # Best supporting evidence, not the mean: averaging over top_k means retrieving one
        # extra weak chunk LOWERS confidence, which is backwards.
        retrieval_support = max(r.score for r in retrieved)
        if self.ensemble:
            context = "\n\n".join([f"[{r.chunk.doc_id}]\n{r.chunk.text}" for r in retrieved])
            system = "Answer based on the provided context."
            user = f"Context:\n{context}\n\nQuestion: {query}"
            answers, agreement = self.ensemble.query(system, user)
            answer_text = answers[0] if answers else ""
        else:
            answer_text = self._simple_answer(query, retrieved)
            agreement = None
        citations, citation_coverage = extract_citations(answer_text, retrieved)
        confidence = compute_confidence(retrieval_support, citation_coverage, agreement)
        decision = self.gate.decide(confidence)
        if decision.calibrated:
            confidence.calibrated = True
            confidence.components["calibrated_probability"] = decision.probability
        confidence.components["gate_reason"] = decision.reason
        if decision.should_answer:
            return Answer(
                text=answer_text,
                confidence=confidence,
                citations=citations,
                answered=True,
                model_versions={},
            )
        else:
            sme_question = generate_sme_question(
                query, retrieved, confidence, self.question_generator
            )
            gap = KnowledgeGap(
                gap_id=str(uuid.uuid4()),
                query=query,
                asked_at=datetime.now(),
                confidence=confidence,
                sme_question=sme_question,
                retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved],
                status="open",
            )
            self.ledger.record(gap)
            return Answer(
                text=sme_question,
                confidence=confidence,
                citations=[],
                answered=False,
                model_versions={},
            )

    def _abstain_no_results(self, query: str) -> Answer:
        confidence = ConfidenceBreakdown(
            retrieval_support=0.0,
            ensemble_agreement=None,
            citation_coverage=0.0,
            composite=0.0,
            components={
                "components_available": [],
                "gate_reason": "No documents were retrieved, so there was no evidence to weigh",
            },
        )
        gap = KnowledgeGap(
            gap_id=str(uuid.uuid4()),
            query=query,
            asked_at=datetime.now(),
            confidence=confidence,
            sme_question=f"No relevant documents found for: {query}. Please provide information.",
            retrieved_chunk_ids=[],
            status="open",
        )
        self.ledger.record(gap)
        return Answer(
            text=gap.sme_question,
            confidence=confidence,
            citations=[],
            answered=False,
            model_versions={},
        )

    def _simple_answer(self, query: str, retrieved: list) -> str:
        if not retrieved:
            return ""
        top = retrieved[0]
        return f"Based on {top.chunk.doc_id}: {top.chunk.text[:200]}"

    def ingest_sme_answer(self, gap_id: str, sme_answer: str, answered_by: str = "sme") -> None:
        self.ledger.mark_answered(gap_id, sme_answer, answered_by)
        gap = self.ledger.get(gap_id)
        if not gap:
            return
        chunk = Chunk(
            chunk_id=f"sme-{gap_id}",
            doc_id=f"sme_answer_{gap_id}",
            text=f"Question: {gap.query}\n\nAnswer: {sme_answer}",
            metadata={"source": "sme", "gap_id": gap_id},
        )
        self.retrieval.index([chunk])
        self.ledger.mark_indexed(gap_id)
