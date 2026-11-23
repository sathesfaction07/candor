import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODEL_TIMEOUT_SECONDS = 30.0


class ModelInvoker(Protocol):
    def __call__(self, model_id: str, system: str, user: str) -> str: ...


class EnsembleClient:
    def __init__(
        self,
        models: list[str],
        invoke: ModelInvoker,
        timeout: float = MODEL_TIMEOUT_SECONDS,
    ):
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        self.models = models
        self.invoke = invoke
        self.timeout = timeout

    def query(self, system: str, user: str) -> tuple[list[str], float | None]:
        answers: list[str] = []
        executor = ThreadPoolExecutor(max_workers=len(self.models))
        try:
            futures = {
                executor.submit(self.invoke, model_id, system, user): model_id
                for model_id in self.models
            }
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    model_id = futures[future]
                    try:
                        answer = future.result()
                        if answer:
                            answers.append(answer)
                    except Exception as exc:
                        logger.error(f"Model {model_id} failed: {exc}")
            except TimeoutError:
                # as_completed raises from the ITERATOR, not from inside the loop body, so
                # catching it below the loop is what actually keeps a slow model from taking
                # the whole ensemble down with it.
                unfinished = [m for f, m in futures.items() if not f.done()]
                logger.warning(f"Ensemble timed out waiting for: {', '.join(unfinished)}")
        finally:
            # wait=False so one hung model cannot block the caller for the full timeout again
            # on the way out.
            executor.shutdown(wait=False, cancel_futures=True)

        if len(answers) < 2:
            logger.warning(
                f"Ensemble degraded: {len(answers)} of {len(self.models)} models answered, "
                "so agreement cannot be measured"
            )
            return answers, None
        return answers, self._measure_agreement(answers)

    def _measure_agreement(self, answers: list[str]) -> float:
        # TF-IDF cosine over the returned answers: dependency-light (sklearn is already
        # required) and deterministic. Deliberately NOT an LLM judging agreement, which would
        # only move the calibration problem somewhere less inspectable.
        if len(answers) < 2:
            return 0.0
        try:
            tfidf = TfidfVectorizer().fit_transform(answers)
        except ValueError:
            # every answer was stop words only, so there is no vocabulary to compare
            return 0.0
        similarities = cosine_similarity(tfidf)
        pairs = [
            similarities[i, j]
            for i in range(len(answers))
            for j in range(i + 1, len(answers))
        ]
        return float(sum(pairs) / len(pairs)) if pairs else 0.0
