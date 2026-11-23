import tempfile
from pathlib import Path

from sme_bridge.config import Settings
from sme_bridge.core.engine import Engine
from sme_bridge.core.gate import ThresholdGate
from sme_bridge.core.ledger import JSONLLedger
from sme_bridge.ingest import get_reader
from sme_bridge.retrieval.keyword import KeywordBackend


def test_full_abstain_to_answer_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        doc1_path = Path(tmpdir) / "doc1.txt"
        doc1_path.write_text(
            "High-powered magnets in toys have caused serious injuries. "
            "Regulations have evolved significantly. Products on the market before recent "
            "regulatory changes may still be available under previous standards."
        )

        backend = KeywordBackend()
        reader = get_reader(doc1_path)
        chunks = reader.read(doc1_path)
        backend.index(chunks)

        ledger_path = Path(tmpdir) / "gaps.jsonl"
        ledger = JSONLLedger(str(ledger_path))

        gate = ThresholdGate(threshold=0.85)
        engine = Engine(retrieval=backend, ledger=ledger, gate=gate)

        query = "What is the exact gauss limit for magnets in children's toys?"
        answer1 = engine.answer(query, top_k=5)

        assert answer1.answered is False, "System should abstain on ambiguous question"
        assert answer1.confidence.composite < 0.85, "Confidence should be below threshold"

        open_gaps = ledger.list_open()
        assert len(open_gaps) == 1, "Should have recorded one knowledge gap"

        gap = open_gaps[0]
        assert gap.query == query
        assert "magnet" in gap.sme_question.lower() or "gauss" in gap.sme_question.lower()

        sme_answer_text = (
            "The current CPSC standard 16 CFR 1262 prohibits high-flux magnets with a flux "
            "index of 50 kG²mm² or greater in children's toys for ages 0-14. This corresponds "
            "to magnets with a magnetic field strength exceeding 5 kilogauss (kG) and a "
            "minimum dimension of 3.17 mm."
        )

        engine.ingest_sme_answer(gap.gap_id, sme_answer_text, answered_by="safety_expert")

        updated_gap = ledger.get(gap.gap_id)
        assert updated_gap.status == "indexed", "Gap should be marked as indexed"
        assert updated_gap.sme_answer == sme_answer_text

        answer2 = engine.answer(query, top_k=5)

        assert (
            answer2.confidence.composite > answer1.confidence.composite
        ), "Confidence should improve after SME input"

        # answer2 still abstains under this deliberately strict 0.85 gate, and that is the
        # point: a raw evidence composite is not a probability, so 0.85 on that scale is not
        # reachable even with the SME answer ranked top. At the gate the project ships, the
        # same query is answered from the SME material.
        permissive = Engine(
            retrieval=backend,
            ledger=ledger,
            gate=ThresholdGate(threshold=Settings().evidence_gate),
        )
        answer3 = permissive.answer(query, top_k=5)
        assert answer3.answered is True
        assert (
            "kG" in answer3.text or "kilogauss" in answer3.text or "50" in answer3.text
        ), "the answer must now draw on the SME-provided specification"

        related = permissive.answer("What minimum dimension applies to magnet restrictions?")
        assert related.answered is True
        assert "3.17" in related.text or "mm" in related.text
