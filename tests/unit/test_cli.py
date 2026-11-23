import tempfile
from datetime import datetime
from pathlib import Path

from click.testing import CliRunner

from sme_bridge.cli import cli
from sme_bridge.core.ledger import JSONLLedger
from sme_bridge.models import Chunk, ConfidenceBreakdown, KnowledgeGap
from sme_bridge.retrieval.keyword import KeywordBackend


def test_ingest_single_file():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / "test.txt"
        doc_path.write_text("Battery compartments must be secured with screws.")

        result = runner.invoke(cli, ["ingest", str(doc_path)])
        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert "chunks" in result.output


def test_ask_command_with_high_confidence():
    backend = KeywordBackend()
    backend.index(
        [
            Chunk(
                chunk_id="c1",
                doc_id="safety.pdf",
                text="Button-cell batteries must be secured in compartments that require a tool.",
            )
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "What are the battery requirements?"])
    assert result.exit_code == 0
    assert "Confidence Breakdown" in result.output
    assert "Retrieval support" in result.output


def test_gaps_command_with_no_gaps():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "gaps.jsonl"
        result = runner.invoke(cli, ["gaps"], env={"SB_LEDGER_PATH": str(ledger_path)})
        assert result.exit_code == 0
        assert "No open knowledge gaps" in result.output


def test_calibrate_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = Path(tmpdir) / "eval.jsonl"
        eval_path.write_text(
            '{"query": "test", "prediction_confidence": 0.9, "is_correct": true}\n'
            '{"query": "test2", "prediction_confidence": 0.8, "is_correct": true}\n'
        )
        output_path = Path(tmpdir) / "report.json"

        result = runner.invoke(cli, ["calibrate", str(eval_path), "--output", str(output_path)])
        assert result.exit_code == 0
        assert "Calibration Report" in result.output
        assert "Samples:" in result.output
        assert "ECE:" in result.output
        assert output_path.exists()


def test_train_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "gaps.jsonl"
        ledger = JSONLLedger(str(ledger_path))
        for n in range(4):
            score = 0.2 + 0.2 * n
            ledger.record(
                KnowledgeGap(
                    gap_id=f"gap-{n}",
                    query=f"question {n}",
                    asked_at=datetime.now(),
                    confidence=ConfidenceBreakdown(
                        retrieval_support=score,
                        ensemble_agreement=None,
                        citation_coverage=score,
                        composite=score,
                    ),
                    sme_question=f"please clarify {n}",
                    retrieved_chunk_ids=[],
                    status="open",
                )
            )
            ledger.mark_answered(f"gap-{n}", f"answer {n}", "expert")
        eval_path = Path(tmpdir) / "eval.jsonl"
        eval_path.write_text(
            '{"query": "test", "prediction_confidence": 0.9, "is_correct": true}\n'
            '{"query": "test2", "prediction_confidence": 0.8, "is_correct": true}\n'
            '{"query": "test3", "prediction_confidence": 0.3, "is_correct": false}\n'
            '{"query": "test4", "prediction_confidence": 0.2, "is_correct": false}\n'
        )
        output_path = Path(tmpdir) / "classifier.joblib"

        result = runner.invoke(
            cli,
            ["train", str(eval_path), "--output", str(output_path)],
            env={"SB_LEDGER_PATH": str(ledger_path)},
        )
        # Gap-derived training data is all negatives by construction, so this must fail
        # loudly rather than produce a classifier that has learned nothing.
        assert result.exit_code != 0
        assert "both classes" in str(result.exception)
        assert not output_path.exists()


def test_ask_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "--help"])
    assert result.exit_code == 0
    assert "query" in result.output.lower()
