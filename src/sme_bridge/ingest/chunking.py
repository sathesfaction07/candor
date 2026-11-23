import re
import uuid

from sme_bridge.models import Chunk

CHUNK_SIZE = 500
OVERLAP = 100


def chunk_text(
    text: str,
    doc_id: str,
    page: int | None,
    section: str | None,
    metadata: dict,
) -> list[Chunk]:
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > CHUNK_SIZE and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text_str,
                    page=page,
                    section=section,
                    metadata=metadata.copy(),
                )
            )
            overlap_text = chunk_text_str[-OVERLAP:] if len(chunk_text_str) > OVERLAP else ""
            current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
            current_length = len(overlap_text) + sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len

    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text_str,
                page=page,
                section=section,
                metadata=metadata.copy(),
            )
        )

    return chunks
