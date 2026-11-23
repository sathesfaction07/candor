import logging
from pathlib import Path

from sme_bridge.models import Chunk

from .base import DocumentReader
from .chunking import chunk_text

logger = logging.getLogger(__name__)


class TextReader(DocumentReader):
    def read(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, doc_id=path.name, page=None, section=None, metadata={})
        logger.info(f"Extracted {len(chunks)} chunks from {path.name}")
        return chunks
