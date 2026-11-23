import tempfile
from pathlib import Path

from sme_bridge.learning.calibration import (
    EvalExample,
    assess_calibration_quality,
    compute_calibration_report,
    compute_ece,
    load_eval_examples,
)


def test_empty_eval_set():
    report = compute_calibration_report([])
    assert report.sample_count == 0
    assert report.ece == 0.0
    assert report.brier_score == 0.0
    assert report.bins == []


def test_single_sample():
    examples = [EvalExample(query="test", prediction_confidence=0.8, is_correct=True)]
    report = compute_calibration_report(examples)
    assert report.sample_count == 1
    assert report.ece >= 0.0
    assert report.brier_score >= 0.0


def test_perfectly_calibrated():
    examples = []
    for i in range(10):
        conf = 0.1 + (i * 0.1)
        correct_count = int(conf * 10)
        for j in range(10):
            examples.append(EvalExample(
                query=f"q{i}_{j}",
                prediction_confidence=conf,
                is_correct=(j < correct_count)
            ))

    report = compute_calibration_report(examples, n_bins=10)
    assert report.ece < 0.05


def test_overconfident_set():
    examples = []
    for i in range(100):
        examples.append(EvalExample(query=f"q{i}", prediction_confidence=0.95, is_correct=i < 50))

    report = compute_calibration_report(examples, n_bins=10)
    assert report.ece > 0.2


def test_accuracy_at_gate():
    examples = [
        EvalExample(query="q1", prediction_confidence=0.95, is_correct=True),
        EvalExample(query="q2", prediction_confidence=0.92, is_correct=True),
        EvalExample(query="q3", prediction_confidence=0.88, is_correct=False),
        EvalExample(query="q4", prediction_confidence=0.50, is_correct=False),
    ]

    report = compute_calibration_report(examples, gate_threshold=0.90)
    assert report.accuracy_at_gate == 1.0
    assert report.gate_threshold == 0.90


def test_accuracy_at_gate_filters_correctly():
    examples = [
        EvalExample(query="q1", prediction_confidence=0.95, is_correct=True),
        EvalExample(query="q2", prediction_confidence=0.85, is_correct=False),
        EvalExample(query="q3", prediction_confidence=0.92, is_correct=False),
    ]

    report = compute_calibration_report(examples, gate_threshold=0.90)
    assert report.accuracy_at_gate == 0.5


def test_empty_bin_handling():
    examples = [
        EvalExample(query="q1", prediction_confidence=0.05, is_correct=True),
        EvalExample(query="q2", prediction_confidence=0.95, is_correct=True),
    ]

    report = compute_calibration_report(examples, n_bins=10)
    assert len(report.bins) == 2


def test_all_correct():
    examples = [
        EvalExample(query=f"q{i}", prediction_confidence=0.9, is_correct=True)
        for i in range(10)
    ]
    report = compute_calibration_report(examples)
    assert report.sample_count == 10


def test_all_incorrect():
    examples = [
        EvalExample(query=f"q{i}", prediction_confidence=0.9, is_correct=False)
        for i in range(10)
    ]
    report = compute_calibration_report(examples)
    assert report.sample_count == 10


def test_compute_ece():
    examples = [EvalExample(query="q1", prediction_confidence=0.8, is_correct=True)]
    ece = compute_ece(examples)
    assert ece >= 0.0


def test_assess_calibration_quality_empty():
    report = compute_calibration_report([])
    assessment = assess_calibration_quality(report)
    assert "No samples" in assessment


def test_assess_calibration_quality_without_gate():
    examples = [EvalExample(query="q1", prediction_confidence=0.8, is_correct=True)]
    report = compute_calibration_report(examples)
    assessment = assess_calibration_quality(report)
    assert "No gate threshold" in assessment


def test_assess_calibration_quality_well_calibrated():
    examples = [
        EvalExample(query=f"q{i}", prediction_confidence=0.90, is_correct=True) for i in range(90)
    ]
    examples.extend([
        EvalExample(query=f"q{i}", prediction_confidence=0.90, is_correct=False) for i in range(10)
    ])
    report = compute_calibration_report(examples, gate_threshold=0.90)
    assessment = assess_calibration_quality(report)
    assert "well-calibrated" in assessment


def test_assess_calibration_quality_overconfident():
    examples = [
        EvalExample(query=f"q{i}", prediction_confidence=0.90, is_correct=True) for i in range(50)
    ]
    examples.extend([
        EvalExample(query=f"q{i}", prediction_confidence=0.90, is_correct=False) for i in range(50)
    ])
    report = compute_calibration_report(examples, gate_threshold=0.90)
    assessment = assess_calibration_quality(report)
    assert "over-confident" in assessment


def test_load_eval_examples():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval.jsonl"
        with path.open("w") as f:
            f.write('{"query": "test", "prediction_confidence": 0.8, "is_correct": true}\n')
            f.write('{"query": "test2", "prediction_confidence": 0.6, "is_correct": false}\n')

        examples = load_eval_examples(str(path))
        assert len(examples) == 2
        assert examples[0].query == "test"
        assert examples[0].prediction_confidence == 0.8
        assert examples[0].is_correct is True


def test_load_eval_examples_nonexistent():
    examples = load_eval_examples("/nonexistent/path.jsonl")
    assert examples == []
