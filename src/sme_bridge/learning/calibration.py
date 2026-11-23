import json
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalExample(BaseModel):
    query: str
    prediction_confidence: float
    is_correct: bool


class CalibrationReport(BaseModel):
    bins: list[dict]
    ece: float
    brier_score: float
    sample_count: int
    accuracy_at_gate: float | None = None
    gate_threshold: float | None = None


def load_eval_examples(path: str) -> list[EvalExample]:
    examples = []
    p = Path(path)
    if not p.exists():
        return examples
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(EvalExample(**json.loads(line)))
    return examples


def compute_calibration_report(
    examples: list[EvalExample],
    n_bins: int = 10,
    gate_threshold: float | None = None,
) -> CalibrationReport:
    if not examples:
        return CalibrationReport(
            bins=[],
            ece=0.0,
            brier_score=0.0,
            sample_count=0,
            accuracy_at_gate=None,
            gate_threshold=gate_threshold,
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = [
        {"lower": bin_edges[i], "upper": bin_edges[i + 1], "samples": []}
        for i in range(n_bins)
    ]

    for ex in examples:
        conf = max(0.0, min(1.0, ex.prediction_confidence))
        for i, bd in enumerate(bin_data):
            if i == n_bins - 1:
                if bd["lower"] <= conf <= bd["upper"]:
                    bd["samples"].append(ex)
                    break
            elif bd["lower"] <= conf < bd["upper"]:
                bd["samples"].append(ex)
                break

    bins = []
    ece_terms = []
    for bd in bin_data:
        if bd["samples"]:
            correct_count = sum(1 for s in bd["samples"] if s.is_correct)
            accuracy = correct_count / len(bd["samples"])
            mean_conf = sum(s.prediction_confidence for s in bd["samples"]) / len(bd["samples"])
            bins.append(
                {
                    "lower": bd["lower"],
                    "upper": bd["upper"],
                    "count": len(bd["samples"]),
                    "accuracy": accuracy,
                    "mean_confidence": mean_conf,
                }
            )
            ece_terms.append(len(bd["samples"]) * abs(accuracy - mean_conf))

    ece = sum(ece_terms) / len(examples) if ece_terms else 0.0

    brier_components = [
        (ex.prediction_confidence - (1.0 if ex.is_correct else 0.0)) ** 2 for ex in examples
    ]
    brier_score = sum(brier_components) / len(examples)

    accuracy_at_gate = None
    if gate_threshold is not None:
        gated_examples = [ex for ex in examples if ex.prediction_confidence >= gate_threshold]
        if gated_examples:
            correct_gated = sum(1 for ex in gated_examples if ex.is_correct)
            accuracy_at_gate = correct_gated / len(gated_examples)

    return CalibrationReport(
        bins=bins,
        ece=ece,
        brier_score=brier_score,
        sample_count=len(examples),
        accuracy_at_gate=accuracy_at_gate,
        gate_threshold=gate_threshold,
    )


def compute_ece(examples: list[EvalExample], n_bins: int = 10) -> float:
    report = compute_calibration_report(examples, n_bins=n_bins)
    return report.ece


def assess_calibration_quality(report: CalibrationReport) -> str:
    if report.sample_count == 0:
        return "No samples available for calibration assessment."

    if report.accuracy_at_gate is None or report.gate_threshold is None:
        return (
            f"ECE: {report.ece:.4f}, Brier: {report.brier_score:.4f}. "
            "No gate threshold specified; cannot assess accuracy at gate."
        )

    delta = report.accuracy_at_gate - report.gate_threshold
    abs_delta = abs(delta)

    if abs_delta < 0.05:
        quality = "well-calibrated"
    elif delta > 0:
        quality = f"under-confident (actual accuracy {abs_delta:.2%} higher than threshold)"
    else:
        quality = f"over-confident (actual accuracy {abs_delta:.2%} lower than threshold)"

    return (
        f"ECE: {report.ece:.4f}, Brier: {report.brier_score:.4f}. "
        f"At gate {report.gate_threshold:.2f}, observed accuracy is {report.accuracy_at_gate:.2%}. "
        f"System is {quality}."
    )
