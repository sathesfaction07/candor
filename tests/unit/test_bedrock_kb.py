from unittest.mock import MagicMock, patch

from sme_bridge.retrieval import get_retrieval_backend


@patch("boto3.client")
def test_bedrock_kb_search_no_kb_id(mock_boto_client, sample_chunks):
    backend = get_retrieval_backend("bedrock_kb", knowledge_base_id="", region="us-east-1")
    results = backend.search("test query", top_k=5)

    assert results == []
    mock_boto_client.return_value.retrieve.assert_not_called()


@patch("boto3.client")
def test_bedrock_kb_search_with_results(mock_boto_client, sample_chunks):
    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": sample_chunks[0].text},
                "metadata": {"chunk_id": sample_chunks[0].chunk_id},
                "score": 0.95,
            }
        ]
    }

    backend = get_retrieval_backend("bedrock_kb", knowledge_base_id="kb-123", region="us-east-1")
    backend.index(sample_chunks)

    results = backend.search("battery safety", top_k=3)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == sample_chunks[0].chunk_id
    assert results[0].score == 0.95


@patch("boto3.client")
def test_bedrock_kb_search_handles_exception(mock_boto_client):
    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client
    mock_client.retrieve.side_effect = Exception("API Error")

    backend = get_retrieval_backend("bedrock_kb", knowledge_base_id="kb-123", region="us-east-1")
    results = backend.search("test", top_k=5)

    assert results == []


@patch("boto3.client")
def test_bedrock_kb_index_only_caches(mock_boto_client, sample_chunks):
    backend = get_retrieval_backend("bedrock_kb", knowledge_base_id="kb-123", region="us-east-1")
    count = backend.index(sample_chunks)

    assert count == 0
    assert len(backend._chunk_map) == len(sample_chunks)
