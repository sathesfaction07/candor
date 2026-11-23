import logging

from sme_bridge.models import Chunk, RetrievedChunk

from .base import RetrievalBackend

logger = logging.getLogger(__name__)


class ChromaDBBackend(RetrievalBackend):
    def __init__(
        self, persist_dir: str = ".chromadb", embedding_model: str = "all-MiniLM-L6-v2", **kwargs
    ):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="sme_bridge_chunks", embedding_function=self.embedding_fn
        )
        self._chunk_map: dict[str, Chunk] = {}

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(query_texts=[query], n_results=top_k)

        retrieved = []
        for i, chunk_id in enumerate(results["ids"][0]):
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                distance = results["distances"][0][i] if results.get("distances") else 0
                score = 1.0 - distance
                retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{"doc_id": c.doc_id, "page": c.page or 0} for c in chunks]

        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        for chunk in chunks:
            self._chunk_map[chunk.chunk_id] = chunk

        logger.info(f"Indexed {len(chunks)} chunks in ChromaDB")
        return len(chunks)
