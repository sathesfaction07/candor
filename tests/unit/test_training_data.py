from datetime import datetime

from sme_bridge.core.ledger import InMemoryLedger
from sme_bridge.learning.training_data import build_training_examples
from sme_bridge.models import ConfidenceBreakdown, KnowledgeGap


def test_answered_gap_becomes_example():
    ledger = InMemoryLedger()

    gap = KnowledgeGap(
        gap_id="gap1",
        query="What is the policy?",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.55,
            composite=0.55,
        ),
        sme_question="What is the policy?",
        retrieved_chunk_ids=["chunk1"],
        status="answered",
        sme_answer="The policy is X",
        answered_at=datetime.now(),
        answered_by="sme1",
    )

    ledger.record(gap)

    examples = build_training_examples(ledger)

    assert len(examples) == 1
    breakdown, is_correct = examples[0]
    assert breakdown.composite == 0.55
    assert is_correct is False


def test_open_gap_not_included():
    ledger = InMemoryLedger()

    gap = KnowledgeGap(
        gap_id="gap1",
        query="What is the policy?",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.55,
            composite=0.55,
        ),
        sme_question="What is the policy?",
        retrieved_chunk_ids=["chunk1"],
        status="open",
    )

    ledger.record(gap)

    examples = build_training_examples(ledger)

    assert len(examples) == 0


def test_multiple_gaps():
    ledger = InMemoryLedger()

    for i in range(3):
        gap = KnowledgeGap(
            gap_id=f"gap{i}",
            query=f"Query {i}",
            asked_at=datetime.now(),
            confidence=ConfidenceBreakdown(
                retrieval_support=0.5,
                ensemble_agreement=0.6,
                citation_coverage=0.55,
                composite=0.55,
            ),
            sme_question=f"Question {i}",
            retrieved_chunk_ids=[f"chunk{i}"],
            status="answered",
            sme_answer=f"Answer {i}",
            answered_at=datetime.now(),
            answered_by="sme1",
        )
        ledger.record(gap)

    examples = build_training_examples(ledger)

    assert len(examples) == 3


def test_empty_ledger():
    ledger = InMemoryLedger()

    examples = build_training_examples(ledger)

    assert len(examples) == 0


def test_indexed_gap_not_included():
    ledger = InMemoryLedger()

    gap = KnowledgeGap(
        gap_id="gap1",
        query="What is the policy?",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.55,
            composite=0.55,
        ),
        sme_question="What is the policy?",
        retrieved_chunk_ids=["chunk1"],
        status="indexed",
        sme_answer="The policy is X",
        answered_at=datetime.now(),
        answered_by="sme1",
        indexed_at=datetime.now(),
    )

    ledger.record(gap)

    examples = build_training_examples(ledger)

    assert len(examples) == 0
