
from sme_bridge.ingest import get_reader


def test_text_reader_basic(tmp_path, tiny_corpus_text):
    text_file = tmp_path / "test.txt"
    text_file.write_text(tiny_corpus_text)

    reader = get_reader(text_file)
    chunks = reader.read(text_file)

    assert len(chunks) > 0
    assert all(c.doc_id == "test.txt" for c in chunks)
    assert all(c.page is None for c in chunks)
    assert any("Battery Products" in c.text for c in chunks)
