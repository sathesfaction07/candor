import logging

import boto3

from sme_bridge.models import Chunk, RetrievedChunk

from .base import RetrievalBackend

logger = logging.getLogger(__name__)


class BedrockKBBackend(RetrievalBackend):
    def __init__(self, knowledge_base_id: str = "", region: str = "us-east-1", **kwargs):
        self.kb_id = knowledge_base_id
        self.client = boto3.client("bedrock-agent-runtime", region_name=region)
        self._chunk_map: dict[str, Chunk] = {}

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.kb_id:
            logger.error("Bedrock KB ID not configured")
            return []

        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
            )
        except Exception as e:
            logger.error(f"Bedrock KB retrieval failed: {e}")
            return []

        results = []
        for result in response.get("retrievalResults", []):
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)

            chunk_id = metadata.get("chunk_id", "")
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                results.append(RetrievedChunk(chunk=chunk, score=score))

        return results

    def index(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self._chunk_map[chunk.chunk_id] = chunk
        logger.info(
            "Bedrock KB indexing is managed via S3 data source sync "
            "— not supported via this interface"
        )
        return 0
