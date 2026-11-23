import json
import logging
from pathlib import Path

import click

from sme_bridge.config import Settings, settings
from sme_bridge.core.engine import Engine
from sme_bridge.core.gate import ThresholdGate
from sme_bridge.core.ledger import get_ledger
from sme_bridge.ingest import get_reader
from sme_bridge.learning.calibration import (
    compute_calibration_report,
    load_eval_examples,
)
from sme_bridge.learning.classifier import AbstentionClassifier
from sme_bridge.learning.training_data import build_training_examples
from sme_bridge.retrieval import get_retrieval_backend

logging.basicConfig(
    level="INFO",
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _ledger():
    # Settings are re-read per invocation: a module-level singleton would freeze SB_* env
    # vars at import time, so documented overrides would silently do nothing.
    s = Settings()
    return get_ledger(s.ledger_backend, path=s.ledger_path)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--backend", default="keyword", help="Retrieval backend")
def ingest(path: str, backend: str) -> None:
    doc_path = Path(path)
    retrieval = get_retrieval_backend(backend, bedrock_kb_id=None, region=settings.bedrock_region)

    if doc_path.is_dir():
        all_chunks = []
        for file_path in sorted(doc_path.iterdir()):
            if file_path.is_file():
                reader = get_reader(file_path)
                chunks = reader.read(file_path)
                all_chunks.extend(chunks)
        count = retrieval.index(all_chunks)
        click.echo(f"Indexed {count} chunks from {len(list(doc_path.iterdir()))} files")
    else:
        reader = get_reader(doc_path)
        chunks = reader.read(doc_path)
        count = retrieval.index(chunks)
        click.echo(f"Indexed {count} chunks from {doc_path.name}")


@cli.command()
@click.argument("query")
@click.option("--backend", default="keyword", help="Retrieval backend")
@click.option("--top-k", default=5, help="Number of results to retrieve")
def ask(query: str, backend: str, top_k: int) -> None:
    retrieval = get_retrieval_backend(backend, bedrock_kb_id=None, region=settings.bedrock_region)
    ledger = _ledger()
    gate = ThresholdGate(threshold=settings.evidence_gate)
    engine = Engine(retrieval=retrieval, ledger=ledger, gate=gate)

    answer = engine.answer(query, top_k=top_k)

    click.secho("\n=== Confidence Breakdown ===", fg="blue", bold=True)
    click.echo(f"Retrieval support:   {answer.confidence.retrieval_support:.2f}")
    if answer.confidence.ensemble_agreement is not None:
        click.echo(f"Ensemble agreement:  {answer.confidence.ensemble_agreement:.2f}")
    else:
        click.echo("Ensemble agreement:  not measured")
    click.echo(f"Citation coverage:   {answer.confidence.citation_coverage:.2f}")
    click.echo(f"Composite:           {answer.confidence.composite:.2f}")
    click.echo(f"Calibrated:          {answer.confidence.calibrated}")
    click.echo(f"Gate reason:         {answer.confidence.components.get('gate_reason', 'N/A')}")

    if answer.answered:
        click.secho("\n=== Answer ===", fg="green", bold=True)
        click.echo(answer.text)
        if answer.citations:
            click.secho("\n=== Citations ===", fg="cyan")
            for i, cit in enumerate(answer.citations, 1):
                page_str = f", page {cit.page}" if cit.page else ""
                click.echo(f"{i}. [{cit.doc_id}{page_str}] \"{cit.quoted_span}\"")
    else:
        click.secho("\n=== ABSTAINED ===", fg="yellow", bold=True)
        click.echo("The system lacks confidence to answer. Generated SME question:\n")
        click.echo(answer.text)


@cli.command()
@click.option("--backend", default="keyword", help="Retrieval backend")
def gaps(backend: str) -> None:
    ledger = _ledger()
    open_gaps = ledger.list_open()

    if not open_gaps:
        click.echo("No open knowledge gaps")
        return

    click.secho(f"\n=== Open Knowledge Gaps ({len(open_gaps)}) ===\n", fg="yellow", bold=True)
    for gap in sorted(open_gaps, key=lambda g: g.asked_at, reverse=True):
        click.secho(f"Gap ID: {gap.gap_id}", fg="cyan")
        click.echo(f"Query:  {gap.query}")
        click.echo(f"Asked:  {gap.asked_at.isoformat()}")
        click.echo(f"Confidence: {gap.confidence.composite:.2f}")
        click.echo(f"Question for SME:\n{gap.sme_question}")
        click.echo("")


@cli.command()
@click.argument("gap_id")
@click.argument("sme_answer")
@click.option("--backend", default="keyword", help="Retrieval backend")
@click.option("--answered-by", default="sme", help="Who answered this gap")
def answer(gap_id: str, sme_answer: str, backend: str, answered_by: str) -> None:
    retrieval = get_retrieval_backend(backend, bedrock_kb_id=None, region=settings.bedrock_region)
    ledger = _ledger()
    gate = ThresholdGate(threshold=settings.evidence_gate)
    engine = Engine(retrieval=retrieval, ledger=ledger, gate=gate)

    gap = ledger.get(gap_id)
    if not gap:
        click.secho(f"Error: Gap not found: {gap_id}", fg="red")
        return

    if gap.status != "open":
        click.secho(f"Error: Gap already answered: {gap_id}", fg="red")
        return

    engine.ingest_sme_answer(gap_id, sme_answer, answered_by)
    updated = ledger.get(gap_id)

    click.secho(f"\n✓ Gap {gap_id} answered and indexed", fg="green", bold=True)
    click.echo(f"Status: {updated.status if updated else 'unknown'}")


@cli.command()
@click.argument("eval_path", type=click.Path(exists=True))
@click.option("--output", default="calibration_report.json", help="Output report path")
@click.option("--n-bins", default=10, help="Number of calibration bins")
def calibrate(eval_path: str, output: str, n_bins: int) -> None:
    examples = load_eval_examples(eval_path)
    if not examples:
        click.secho("Error: No evaluation examples found", fg="red")
        return

    report = compute_calibration_report(
        examples, n_bins=n_bins, gate_threshold=settings.evidence_gate
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)

    click.secho("\n=== Calibration Report ===", fg="blue", bold=True)
    click.echo(f"Samples:             {report.sample_count}")
    click.echo(f"ECE:                 {report.ece:.4f}")
    click.echo(f"Brier score:         {report.brier_score:.4f}")
    if report.accuracy_at_gate is not None:
        click.echo(f"Gate threshold:      {report.gate_threshold:.2f}")
        click.echo(f"Accuracy at gate:    {report.accuracy_at_gate:.2%}")
        delta = abs(report.accuracy_at_gate - report.gate_threshold)
        if delta < 0.05:
            quality = "well-calibrated"
            color = "green"
        elif report.accuracy_at_gate > report.gate_threshold:
            quality = f"under-confident (actual {delta:.2%} higher)"
            color = "yellow"
        else:
            quality = f"over-confident (actual {delta:.2%} lower)"
            color = "red"
        click.secho(f"Assessment:          {quality}", fg=color)

    click.echo(f"\nReport saved to {output}")


@cli.command()
@click.argument("eval_path", type=click.Path(exists=True))
@click.option("--output", default="classifier.joblib", help="Output classifier path")
def train(eval_path: str, output: str) -> None:
    examples = load_eval_examples(eval_path)
    if not examples:
        click.secho("Error: No evaluation examples found", fg="red")
        return

    training_examples = build_training_examples(_ledger())
    if not training_examples:
        click.secho("Error: Could not build training examples", fg="red")
        return

    classifier = AbstentionClassifier()
    classifier.fit(training_examples)

    out_path = Path(output)
    classifier.save(str(out_path))

    click.secho(
        f"\n✓ Trained classifier on {len(training_examples)} examples",
        fg="green",
        bold=True,
    )
    click.echo(f"Saved to {output}")


if __name__ == "__main__":
    cli()
