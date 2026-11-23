import logging
from collections.abc import Callable

from sme_bridge.core.prompts import SME_QUESTION_GENERATION
from sme_bridge.models import ConfidenceBreakdown, RetrievedChunk

logger = logging.getLogger(__name__)


def generate_sme_question(
    query: str,
    retrieved: list[RetrievedChunk],
    confidence: ConfidenceBreakdown,
    llm_generate: Callable[[str], str] | None = None,
) -> str:
    context = "\n\n".join(
        f"[{r.chunk.doc_id}, page {r.chunk.page}]\n{r.chunk.text}" for r in retrieved
    )
    ensemble_str = (
        f"{confidence.ensemble_agreement:.2f}"
        if confidence.ensemble_agreement is not None
        else "not measured (fewer than two models answered)"
    )
    prompt = SME_QUESTION_GENERATION.format(
        query=query,
        context=context,
        retrieval_support=confidence.retrieval_support,
        citation_coverage=confidence.citation_coverage,
        ensemble_agreement=ensemble_str,
    )
    if llm_generate:
        try:
            return llm_generate(prompt)
        except Exception as e:
            logger.warning(f"LLM question generation failed, using fallback: {e}")
    return _fallback_question(query, retrieved, confidence)


def _fallback_question(
    query: str,
    retrieved: list[RetrievedChunk],
    confidence: ConfidenceBreakdown,
) -> str:
    docs = ", ".join(dict.fromkeys(r.chunk.doc_id for r in retrieved[:3])) or "no documents"
    agreement = (
        f"{confidence.ensemble_agreement:.2f}"
        if confidence.ensemble_agreement is not None
        else "not measured"
    )
    measured = {
        "retrieval support": confidence.retrieval_support,
        "citation coverage": confidence.citation_coverage,
    }
    if confidence.ensemble_agreement is not None:
        measured["ensemble agreement"] = confidence.ensemble_agreement
    weakest = min(measured, key=lambda k: measured[k])
    asks = {
        "retrieval support": (
            "the documents on file do not appear to cover this case at all, so please state "
            "which document or clause governs it, or confirm that none does"
        ),
        "citation coverage": (
            "the retrieved text is related but does not directly state an answer, so please "
            "confirm the correct answer and which passage supports it"
        ),
        "ensemble agreement": (
            "the available sources support more than one reading, so please state which "
            "reading is correct and why the other is wrong"
        ),
    }
    return f"""A user asked: "{query}"

Closest material on file: {docs}

Not answered: the evidence was too weak to rely on (composite {confidence.composite:.2f}):
- retrieval support: {confidence.retrieval_support:.2f}
- citation coverage: {confidence.citation_coverage:.2f}
- ensemble agreement: {agreement}

The weakest signal was {weakest}, meaning {asks[weakest]}.

Your answer will be added to the knowledge base, so please make it self-contained."""
