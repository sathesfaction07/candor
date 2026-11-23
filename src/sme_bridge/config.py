from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SB_")

    retrieval_backend: str = "keyword"
    bedrock_region: str = "us-east-1"
    ensemble_models_str: str = "us.anthropic.claude-sonnet-4,us.anthropic.claude-haiku-3-5"
    # Threshold for the CALIBRATED probability that CalibratedGate produces. 0.90 here can
    # honestly mean "roughly 90% of the answers this allows are correct" -- but only after
    # `calibrate` has been run against labelled data.
    confidence_gate: float = 0.90
    # Threshold for the UNCALIBRATED evidence composite that ThresholdGate uses. This is NOT
    # a probability. Retrieval support is a raw TF-IDF cosine, which sits around 0.5-0.7 even
    # for a strong match, so reusing 0.90 here would abstain on essentially everything -- the
    # gate would look strict while actually being broken.
    evidence_gate: float = 0.55
    top_k: int = 5
    ledger_path: str = "knowledge_gaps.jsonl"
    ledger_backend: str = "jsonl"
    log_level: str = "INFO"

    @property
    def ensemble_models(self) -> list[str]:
        return [m.strip() for m in self.ensemble_models_str.split(",")]


settings = Settings()
