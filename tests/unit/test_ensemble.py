import time

import pytest

from sme_bridge.core.ensemble import EnsembleClient


def test_single_model_responding_agreement_is_none():
    def fake_invoke(model_id, system, user):
        if model_id == "model-a":
            return "Answer from model A"
        raise Exception("Model failed")

    client = EnsembleClient(models=["model-a", "model-b"], invoke=fake_invoke)
    answers, agreement = client.query("system", "user")
    assert len(answers) == 1
    assert agreement is None


def test_all_models_failing():
    def fake_invoke(model_id, system, user):
        raise Exception("All models down")

    client = EnsembleClient(models=["model-a", "model-b"], invoke=fake_invoke)
    answers, agreement = client.query("system", "user")
    assert len(answers) == 0
    assert agreement is None


def test_one_model_timing_out():
    def fake_invoke(model_id, system, user):
        if model_id == "slow":
            time.sleep(5)
        return f"Answer from {model_id}"

    client = EnsembleClient(models=["fast", "slow"], invoke=fake_invoke, timeout=0.05)
    started = time.monotonic()
    answers, agreement = client.query("system", "user")
    elapsed = time.monotonic() - started
    assert answers == ["Answer from fast"]
    assert agreement is None
    assert elapsed < 2.0


def test_identical_answers_high_agreement():
    def fake_invoke(model_id, system, user):
        return "Battery products require secure compartments"

    client = EnsembleClient(models=["model-a", "model-b"], invoke=fake_invoke)
    answers, agreement = client.query("system", "user")
    assert len(answers) == 2
    assert agreement is not None
    assert agreement > 0.9


def test_unrelated_answers_low_agreement():
    responses = {
        "model-a": "Button-cell batteries must be secured",
        "model-b": "Window coverings present strangulation hazards",
    }

    def fake_invoke(model_id, system, user):
        return responses[model_id]

    client = EnsembleClient(models=["model-a", "model-b"], invoke=fake_invoke)
    answers, agreement = client.query("system", "user")
    assert len(answers) == 2
    assert agreement is not None
    assert agreement < 0.5


def test_requires_at_least_two_models():
    def fake_invoke(model_id, system, user):
        return "answer"

    with pytest.raises(ValueError, match="at least 2 models"):
        EnsembleClient(models=["single"], invoke=fake_invoke)
