from sme_bridge.core.citations import extract_citations

SUPPORTED = (
    "Button-cell batteries must be secured. Products with "
    "accessible batteries are prohibited."
)


def test_zero_support(retrieved_chunks):
    answer = "This answer discusses completely unrelated topics like quantum physics."
    citations, coverage = extract_citations(answer, retrieved_chunks)
    assert coverage == 0.0
    assert citations == []


def test_full_support(retrieved_chunks):
    citations, coverage = extract_citations(SUPPORTED, retrieved_chunks)
    assert coverage >= 0.5
    assert len(citations) > 0


def test_partial_support(retrieved_chunks):
    answer = "Batteries must be secured. Also, quantum entanglement is fascinating."
    citations, coverage = extract_citations(answer, retrieved_chunks)
    assert 0.0 < coverage < 1.0
    assert len(citations) > 0


def test_empty_answer(retrieved_chunks):
    citations, coverage = extract_citations("", retrieved_chunks)
    assert coverage == 0.0
    assert citations == []


def test_citation_includes_chunk_metadata(retrieved_chunks):
    citations, coverage = extract_citations(SUPPORTED, retrieved_chunks)
    assert coverage > 0.0
    assert citations, "supported text must yield at least one citation"
    assert citations[0].chunk_id == "chunk-001"
    assert citations[0].doc_id == "safety.pdf"
