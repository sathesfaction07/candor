from abc import ABC, abstractmethod
from pathlib import Path

from sme_bridge.models import Chunk


class DocumentReader(ABC):
    @abstractmethod
    def read(self, path: Path) -> list[Chunk]: ...
