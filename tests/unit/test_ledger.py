from datetime import datetime

from sme_bridge.core.ledger import InMemoryLedger, JSONLLedger
from sme_bridge.models import ConfidenceBreakdown, KnowledgeGap


def test_record_and_get():
    ledger = InMemoryLedger()
    gap = KnowledgeGap(
        gap_id="gap-001",
        query="What are the requirements?",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Please clarify requirements",
        retrieved_chunk_ids=["chunk-001"],
        status="open",
    )
    ledger.record(gap)
    retrieved = ledger.get("gap-001")
    assert retrieved is not None
    assert retrieved.query == "What are the requirements?"


def test_list_open():
    ledger = InMemoryLedger()
    gap1 = KnowledgeGap(
        gap_id="gap-001",
        query="Question 1",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question 1",
        retrieved_chunk_ids=[],
        status="open",
    )
    gap2 = KnowledgeGap(
        gap_id="gap-002",
        query="Question 2",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question 2",
        retrieved_chunk_ids=[],
        status="answered",
    )
    ledger.record(gap1)
    ledger.record(gap2)
    open_gaps = ledger.list_open()
    assert len(open_gaps) == 1
    assert open_gaps[0].gap_id == "gap-001"


def test_mark_answered():
    ledger = InMemoryLedger()
    gap = KnowledgeGap(
        gap_id="gap-001",
        query="Question",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question",
        retrieved_chunk_ids=[],
        status="open",
    )
    ledger.record(gap)
    ledger.mark_answered("gap-001", "The answer is...", "expert")
    retrieved = ledger.get("gap-001")
    assert retrieved.status == "answered"
    assert retrieved.sme_answer == "The answer is..."
    assert retrieved.answered_by == "expert"


def test_mark_indexed():
    ledger = InMemoryLedger()
    gap = KnowledgeGap(
        gap_id="gap-001",
        query="Question",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question",
        retrieved_chunk_ids=[],
        status="answered",
    )
    ledger.record(gap)
    ledger.mark_indexed("gap-001")
    retrieved = ledger.get("gap-001")
    assert retrieved.status == "indexed"


def test_statistics():
    ledger = InMemoryLedger()
    gap1 = KnowledgeGap(
        gap_id="gap-001",
        query="Question",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question",
        retrieved_chunk_ids=[],
        status="open",
    )
    ledger.record(gap1)
    ledger.mark_answered("gap-001", "Answer", "expert")
    stats = ledger.statistics()
    assert stats["total"] == 1
    assert stats["open"] == 0
    assert stats["resolution_rate"] == 1.0


def test_jsonl_round_trip(tmp_path):
    ledger = JSONLLedger(path=str(tmp_path / "gaps.jsonl"))
    gap = KnowledgeGap(
        gap_id="gap-001",
        query="Question",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question",
        retrieved_chunk_ids=[],
        status="open",
    )
    ledger.record(gap)
    ledger2 = JSONLLedger(path=str(tmp_path / "gaps.jsonl"))
    retrieved = ledger2.get("gap-001")
    assert retrieved is not None
    assert retrieved.query == "Question"


def test_jsonl_append_only(tmp_path):
    ledger = JSONLLedger(path=str(tmp_path / "gaps.jsonl"))
    gap = KnowledgeGap(
        gap_id="gap-001",
        query="Question",
        asked_at=datetime.now(),
        confidence=ConfidenceBreakdown(
            retrieval_support=0.5,
            ensemble_agreement=0.6,
            citation_coverage=0.4,
            composite=0.5,
            components={},
        ),
        sme_question="Question",
        retrieved_chunk_ids=[],
        status="open",
    )
    ledger.record(gap)
    ledger.mark_answered("gap-001", "Answer", "expert")
    lines = (tmp_path / "gaps.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
