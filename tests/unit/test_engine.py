from sme_bridge.core.engine import Engine
from sme_bridge.core.gate import ThresholdGate
from sme_bridge.core.ledger import InMemoryLedger
from sme_bridge.models import Chunk, ConfidenceBreakdown
from sme_bridge.retrieval.keyword import KeywordBackend


def test_abstain_path_produces_gap():
    backend = KeywordBackend()
    backend.index(
        [
            Chunk(
                chunk_id="c1",
                doc_id="doc.pdf",
                text="Some vague information",
            )
        ]
    )
    ledger = InMemoryLedger()
    gate = ThresholdGate(threshold=0.99)
    engine = Engine(retrieval=backend, ledger=ledger, gate=gate)
    answer = engine.answer("What are the specific requirements?")
    assert answer.answered is False
    assert len(ledger.list_open()) == 1
    gap = ledger.list_open()[0]
    assert "requirements" in gap.query


QUERY = "What are the battery compartment requirements?"


def test_learn_and_evolve():
    backend = KeywordBackend()
    backend.index(
        [
            Chunk(
                chunk_id="c1",
                doc_id="doc.pdf",
                text="General safety information",
            )
        ]
    )
    ledger = InMemoryLedger()
    gate = ThresholdGate(threshold=0.99)
    engine = Engine(retrieval=backend, ledger=ledger, gate=gate)
    answer1 = engine.answer(QUERY)
    assert answer1.answered is False
    gaps = ledger.list_open()
    assert len(gaps) == 1
    gap_id = gaps[0].gap_id
    sme_answer = (
        "Battery compartments must be secured with a screw and "
        "can only be opened with a tool."
    )
    engine.ingest_sme_answer(gap_id, sme_answer, "expert")
    assert ledger.get(gap_id).status == "indexed"

    answer2 = engine.answer(QUERY)
    assert answer2.confidence.composite > answer1.confidence.composite


def test_sme_answer_is_used_once_indexed():
    backend = KeywordBackend()
    backend.index([Chunk(chunk_id="c1", doc_id="doc.pdf", text="General safety information")])
    ledger = InMemoryLedger()
    strict = Engine(retrieval=backend, ledger=ledger, gate=ThresholdGate(threshold=0.99))
    assert strict.answer(QUERY).answered is False
    gap_id = ledger.list_open()[0].gap_id
    strict.ingest_sme_answer(
        gap_id,
        "Battery compartments must be secured with a screw and can only be opened with a tool.",
        "expert",
    )

    permissive = Engine(retrieval=backend, ledger=ledger, gate=ThresholdGate(threshold=0.0))
    answer = permissive.answer(QUERY)
    assert answer.answered is True
    assert "tool" in answer.text.lower()


def test_no_results_path():
    backend = KeywordBackend()
    ledger = InMemoryLedger()
    engine = Engine(retrieval=backend, ledger=ledger)
    answer = engine.answer("Completely unrelated query about quantum mechanics")
    assert answer.answered is False
    assert "No relevant documents" in answer.text
    assert len(ledger.list_open()) == 1


def test_answer_path_with_high_confidence():
    backend = KeywordBackend()
    backend.index(
        [
            Chunk(
                chunk_id="c1",
                doc_id="doc.pdf",
                text=(
                    "Button-cell batteries must be secured in battery "
                    "compartments that require a tool to open."
                ),
            )
        ]
    )
    ledger = InMemoryLedger()
    gate = ThresholdGate(threshold=0.01)
    engine = Engine(retrieval=backend, ledger=ledger, gate=gate)
    answer = engine.answer("What are the battery requirements?")
    assert answer.answered is True
    assert "batter" in answer.text.lower()


RELEVANT = "Battery compartment requirements state that compartments must be secured."


def test_threshold_gate_leaves_breakdown_marked_uncalibrated():
    backend = KeywordBackend()
    backend.index([Chunk(chunk_id="c1", doc_id="doc.pdf", text=RELEVANT)])
    engine = Engine(
        retrieval=backend, ledger=InMemoryLedger(), gate=ThresholdGate(threshold=0.0)
    )
    answer = engine.answer(QUERY)
    assert answer.confidence.calibrated is False
    assert "calibrated_probability" not in answer.confidence.components
    assert answer.confidence.components["gate_reason"]


def test_calibrated_gate_marks_breakdown_and_records_probability():
    from sme_bridge.learning.classifier import AbstentionClassifier, CalibratedGate

    classifier = AbstentionClassifier()
    classifier.fit(
        [
            (ConfidenceBreakdown(
                retrieval_support=s, ensemble_agreement=None, citation_coverage=s,
                composite=s,
            ), s > 0.5)
            for s in (0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95)
        ]
    )
    backend = KeywordBackend()
    backend.index([Chunk(chunk_id="c1", doc_id="doc.pdf", text=RELEVANT)])
    engine = Engine(
        retrieval=backend,
        ledger=InMemoryLedger(),
        gate=CalibratedGate(classifier=classifier, threshold=0.90),
    )
    answer = engine.answer(QUERY)
    assert answer.confidence.calibrated is True
    assert answer.confidence.components["calibrated_probability"] is not None
    assert "Calibrated probability" in answer.confidence.components["gate_reason"]
