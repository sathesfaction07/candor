import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sme_bridge.models import Chunk, RetrievedChunk

from .base import RetrievalBackend

logger = logging.getLogger(__name__)


class KeywordBackend(RetrievalBackend):
    def __init__(self, **kwargs):
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.vectors = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks or self.vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append(RetrievedChunk(chunk=self.chunks[idx], score=score))

        return results

    def index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Indexed {len(chunks)} chunks")
        return len(chunks)
