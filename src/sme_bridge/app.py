import logging
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from pydantic import BaseModel, ValidationError

from sme_bridge.config import settings
from sme_bridge.core.engine import Engine
from sme_bridge.core.ensemble import EnsembleClient
from sme_bridge.core.gate import ThresholdGate
from sme_bridge.core.ledger import get_ledger
from sme_bridge.learning.calibration import CalibrationReport, load_eval_examples
from sme_bridge.learning.classifier import CalibratedGate
from sme_bridge.retrieval import get_retrieval_backend

logging.basicConfig(
    level=settings.log_level,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sme_bridge")

app = Flask(__name__)

retrieval = get_retrieval_backend(
    settings.retrieval_backend,
    bedrock_kb_id=None,
    region=settings.bedrock_region,
)
ledger = get_ledger(settings.ledger_backend, path=settings.ledger_path)

ensemble_client: EnsembleClient | None = None
gate: ThresholdGate | CalibratedGate = ThresholdGate(threshold=settings.evidence_gate)
calibration_report: CalibrationReport | None = None

engine = Engine(
    retrieval=retrieval,
    ledger=ledger,
    ensemble=ensemble_client,
    gate=gate,
)

START_TIME = time.time()
request_count = 0


class QueryRequest(BaseModel):
    query: str


class AnswerGapRequest(BaseModel):
    answer: str
    answered_by: str = "sme"


@app.get("/health")
def health() -> ResponseReturnValue:
    return jsonify(
        {
            "status": "healthy",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "requests_served": request_count,
            "retrieval_backend": settings.retrieval_backend,
            "gate_calibrated": isinstance(gate, CalibratedGate),
        }
    )


@app.post("/api/query")
def handle_query() -> ResponseReturnValue:
    global request_count
    request_count += 1

    try:
        req = QueryRequest(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": e.errors()}), 400

    if not req.query.strip():
        return jsonify({"error": "Query cannot be empty"}), 400

    logger.info(f"Query: {req.query}")
    answer = engine.answer(req.query, top_k=settings.top_k)

    response = {
        "answered": answer.answered,
        "text": answer.text,
        "confidence": {
            "retrieval_support": answer.confidence.retrieval_support,
            "ensemble_agreement": answer.confidence.ensemble_agreement,
            "citation_coverage": answer.confidence.citation_coverage,
            "composite": answer.confidence.composite,
            "calibrated": answer.confidence.calibrated,
            "components_available": answer.confidence.components.get("components_available", []),
            "gate_reason": answer.confidence.components.get("gate_reason", ""),
        },
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page": c.page,
                "quoted_span": c.quoted_span,
            }
            for c in answer.citations
        ],
    }

    if answer.confidence.calibrated:
        response["confidence"]["calibrated_probability"] = answer.confidence.components.get(
            "calibrated_probability"
        )

    if not answer.answered:
        gaps = ledger.list_open()
        if gaps:
            latest = gaps[-1]
            response["gap_id"] = latest.gap_id
            response["sme_question"] = latest.sme_question

    return jsonify(response)


@app.get("/api/gaps")
def list_gaps() -> ResponseReturnValue:
    gaps = ledger.list_open()
    gaps_sorted = sorted(gaps, key=lambda g: g.asked_at, reverse=True)
    return jsonify(
        {
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "query": g.query,
                    "asked_at": g.asked_at.isoformat(),
                    "sme_question": g.sme_question,
                    "confidence_composite": g.confidence.composite,
                }
                for g in gaps_sorted
            ],
            "count": len(gaps_sorted),
        }
    )


@app.post("/api/gaps/<gap_id>/answer")
def answer_gap(gap_id: str) -> ResponseReturnValue:
    gap = ledger.get(gap_id)
    if not gap:
        return jsonify({"error": f"Gap not found: {gap_id}"}), 404

    if gap.status != "open":
        return jsonify({"error": f"Gap already answered: {gap_id}"}), 400

    try:
        req = AnswerGapRequest(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": e.errors()}), 400

    if not req.answer.strip():
        return jsonify({"error": "Answer cannot be empty"}), 400

    engine.ingest_sme_answer(gap_id, req.answer, req.answered_by)
    updated_gap = ledger.get(gap_id)

    logger.info(f"Gap {gap_id} answered and indexed")

    return jsonify(
        {
            "gap_id": gap_id,
            "status": updated_gap.status if updated_gap else "unknown",
            "indexed": updated_gap.status == "indexed" if updated_gap else False,
        }
    )


@app.get("/api/calibration")
def get_calibration() -> ResponseReturnValue:
    if calibration_report is None:
        return jsonify(
            {
                "calibrated": False,
                "message": (
                    "The gate is UNCALIBRATED. The threshold of "
                    f"{settings.confidence_gate:.2f} operates on a raw evidence composite, "
                    "not a probability. To obtain a calibrated gate where 0.90 means '90% "
                    "likely correct,' run the 'calibrate' command against a labelled "
                    "evaluation dataset (query, prediction_confidence, is_correct). Until "
                    "then, the gate's threshold is a heuristic tuning knob, not a statistical "
                    "guarantee."
                ),
            }
        )

    return jsonify(
        {
            "calibrated": True,
            "ece": calibration_report.ece,
            "brier_score": calibration_report.brier_score,
            "sample_count": calibration_report.sample_count,
            "accuracy_at_gate": calibration_report.accuracy_at_gate,
            "gate_threshold": calibration_report.gate_threshold,
            "bins": calibration_report.bins,
        }
    )


def load_calibration(report_path: str | Path) -> None:
    global calibration_report
    from sme_bridge.learning.calibration import compute_calibration_report

    examples = load_eval_examples(str(report_path))
    if examples:
        calibration_report = compute_calibration_report(
            examples, gate_threshold=settings.evidence_gate
        )
        logger.info(f"Loaded calibration report: {len(examples)} examples")


def set_gate(new_gate: ThresholdGate | CalibratedGate) -> None:
    global gate
    gate = new_gate
    engine.gate = gate
    logger.info(f"Gate updated: {type(gate).__name__}")
