import logging
import re

from sme_bridge.models import Citation, RetrievedChunk

logger = logging.getLogger(__name__)


def extract_citations(
    answer_text: str,
    retrieved: list[RetrievedChunk],
) -> tuple[list[Citation], float]:
    if not answer_text.strip():
        return [], 0.0
    sentences = _split_sentences(answer_text)
    if not sentences:
        return [], 0.0
    citations = []
    supported_count = 0
    for sentence in sentences:
        found_support = False
        for ret in retrieved:
            spans = _find_supporting_spans(sentence, ret.chunk.text)
            if spans:
                for span in spans:
                    citations.append(
                        Citation(
                            chunk_id=ret.chunk.chunk_id,
                            doc_id=ret.chunk.doc_id,
                            page=ret.chunk.page,
                            quoted_span=span,
                        )
                    )
                found_support = True
                break
        if found_support:
            supported_count += 1
    coverage = supported_count / len(sentences)
    return citations, coverage


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _find_supporting_spans(sentence: str, chunk_text: str, min_word_overlap: int = 3) -> list[str]:
    sentence_words = set(re.findall(r'\w+', sentence.lower()))
    if len(sentence_words) < min_word_overlap:
        return []
    chunk_windows = _sliding_windows(chunk_text, window_size=50)
    matches = []
    for window in chunk_windows:
        window_words = set(re.findall(r'\w+', window.lower()))
        overlap = len(sentence_words & window_words)
        if overlap >= min_word_overlap:
            matches.append(window)
    if not matches:
        return []
    return [matches[0]]


def _sliding_windows(text: str, window_size: int) -> list[str]:
    words = text.split()
    if len(words) <= window_size:
        return [text]
    windows = []
    for i in range(len(words) - window_size + 1):
        windows.append(" ".join(words[i : i + window_size]))
    return windows
