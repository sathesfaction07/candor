import os

from sme_bridge.config import Settings


def test_config_ensemble_models_property():
    settings = Settings(ensemble_models_str="model-a,model-b,model-c")
    models = settings.ensemble_models

    assert isinstance(models, list)
    assert len(models) == 3
    assert models == ["model-a", "model-b", "model-c"]


def test_config_ensemble_models_with_spaces():
    settings = Settings(ensemble_models_str="model-a , model-b , model-c")
    models = settings.ensemble_models

    assert models == ["model-a", "model-b", "model-c"]


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("SB_CONFIDENCE_GATE", "0.75")
    monkeypatch.setenv("SB_TOP_K", "10")
    monkeypatch.setenv("SB_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.confidence_gate == 0.75
    assert settings.top_k == 10
    assert settings.log_level == "DEBUG"


def test_config_defaults():
    for key in os.environ:
        if key.startswith("SB_"):
            del os.environ[key]

    settings = Settings()

    assert settings.retrieval_backend == "keyword"
    assert settings.confidence_gate == 0.90
    assert settings.top_k == 5
    assert settings.ledger_backend == "jsonl"
