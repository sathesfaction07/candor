import logging
from pathlib import Path

import fitz

from sme_bridge.models import Chunk

from .base import DocumentReader
from .chunking import chunk_text

logger = logging.getLogger(__name__)


class PDFReader(DocumentReader):
    def read(self, path: Path) -> list[Chunk]:
        doc = fitz.open(path)
        chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue

            page_chunks = chunk_text(
                text, doc_id=path.name, page=page_num + 1, section=None, metadata={}
            )
            chunks.extend(page_chunks)

        doc.close()
        logger.info(f"Extracted {len(chunks)} chunks from {len(doc)} pages in {path.name}")
        return chunks
