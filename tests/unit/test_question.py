from sme_bridge.core.question import generate_sme_question
from sme_bridge.models import ConfidenceBreakdown


def test_llm_path(retrieved_chunks):
    confidence = ConfidenceBreakdown(
        retrieval_support=0.5,
        ensemble_agreement=0.6,
        citation_coverage=0.4,
        composite=0.5,
        components={},
    )

    def fake_llm(prompt):
        return "Custom question from LLM"

    question = generate_sme_question(
        "Are batteries safe?",
        retrieved_chunks,
        confidence,
        llm_generate=fake_llm,
    )
    assert question == "Custom question from LLM"


def test_deterministic_fallback(retrieved_chunks):
    confidence = ConfidenceBreakdown(
        retrieval_support=0.5,
        ensemble_agreement=0.6,
        citation_coverage=0.4,
        composite=0.5,
        components={},
    )

    question = generate_sme_question(
        "Are batteries safe?",
        retrieved_chunks,
        confidence,
        llm_generate=None,
    )
    assert "Are batteries safe?" in question
    assert "0.50" in question


def test_fallback_on_llm_failure(retrieved_chunks):
    confidence = ConfidenceBreakdown(
        retrieval_support=0.5,
        ensemble_agreement=0.6,
        citation_coverage=0.4,
        composite=0.5,
        components={},
    )

    def failing_llm(prompt):
        raise Exception("LLM unavailable")

    question = generate_sme_question(
        "Are batteries safe?",
        retrieved_chunks,
        confidence,
        llm_generate=failing_llm,
    )
    assert "Are batteries safe?" in question


def test_question_includes_query_and_ambiguity(retrieved_chunks):
    confidence = ConfidenceBreakdown(
        retrieval_support=0.3,
        ensemble_agreement=0.4,
        citation_coverage=0.2,
        composite=0.3,
        components={},
    )
    question = generate_sme_question(
        "What are the battery requirements?",
        retrieved_chunks,
        confidence,
        llm_generate=None,
    )
    assert "battery requirements" in question.lower()
    assert "0.30" in question
