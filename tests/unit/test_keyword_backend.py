from sme_bridge.retrieval import get_retrieval_backend


def test_keyword_backend_finds_relevant_chunk(sample_chunks):
    backend = get_retrieval_backend("keyword")
    backend.index(sample_chunks)

    results = backend.search("button battery safety", top_k=2)

    assert len(results) > 0
    assert "button-cell" in results[0].chunk.text.lower()
    assert results[0].score > 0


def test_keyword_backend_finds_the_relevant_chunk(sample_chunks):
    backend = get_retrieval_backend("keyword")
    backend.index(sample_chunks)
    results = backend.search("strangulation hazard cords", top_k=3)
    assert len(results) == 1
    assert "strangulation" in results[0].chunk.text.lower()


def test_keyword_backend_ranks_sensibly(sample_chunks):
    # A query that genuinely matches several chunks. The previous version used a query
    # matching exactly one, then guarded the ordering assertion behind `if len(results) > 1`
    # -- so the test named "ranks sensibly" never checked an ordering at all.
    backend = get_retrieval_backend("keyword")
    backend.index(sample_chunks)
    results = backend.search("accessible products safety", top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_keyword_backend_empty_index():
    backend = get_retrieval_backend("keyword")
    results = backend.search("anything", top_k=5)
    assert results == []


def test_keyword_backend_no_match(sample_chunks):
    backend = get_retrieval_backend("keyword")
    backend.index(sample_chunks)

    results = backend.search("xyzabc123nonexistent", top_k=5)
    assert results == []


def test_keyword_backend_index_returns_count(sample_chunks):
    backend = get_retrieval_backend("keyword")
    count = backend.index(sample_chunks)
    assert count == len(sample_chunks)


def test_keyword_backend_index_empty():
    backend = get_retrieval_backend("keyword")
    count = backend.index([])
    assert count == 0
