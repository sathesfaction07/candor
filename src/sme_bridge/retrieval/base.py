from abc import ABC, abstractmethod

from sme_bridge.models import Chunk, RetrievedChunk


class RetrievalBackend(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> int: ...
