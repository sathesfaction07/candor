from sme_bridge.ingest.chunking import chunk_text


def test_chunk_text_with_overlap():
    text = "First sentence. " * 50 + "Last sentence."
    chunks = chunk_text(text, doc_id="test.pdf", page=1, section=None, metadata={})

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        chunk_text_str = chunks[i].text
        next_chunk_text = chunks[i + 1].text
        assert any(word in next_chunk_text for word in chunk_text_str.split()[-10:])


def test_chunk_text_respects_sentence_boundaries():
    text = "Short one. " * 100
    chunks = chunk_text(text, doc_id="test.pdf", page=1, section=None, metadata={})

    for chunk in chunks:
        assert chunk.text.strip().endswith(".")


def test_chunk_text_empty_input():
    chunks = chunk_text("", doc_id="test.pdf", page=1, section=None, metadata={})
    assert chunks == []

    chunks = chunk_text("   \n  \t  ", doc_id="test.pdf", page=1, section=None, metadata={})
    assert chunks == []


def test_chunk_text_single_huge_string():
    text = "x" * 5000
    chunks = chunk_text(text, doc_id="test.pdf", page=1, section=None, metadata={})

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == "test.pdf"
        assert chunk.page == 1


def test_chunk_text_preserves_metadata():
    text = "Test sentence."
    meta = {"category": "test", "priority": "high"}
    chunks = chunk_text(text, doc_id="test.pdf", page=3, section="Intro", metadata=meta)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_id == "test.pdf"
    assert chunk.page == 3
    assert chunk.section == "Intro"
    assert chunk.metadata == meta
