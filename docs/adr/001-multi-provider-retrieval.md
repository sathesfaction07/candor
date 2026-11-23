# ADR 001: Multi-Provider Retrieval Strategy

## Status

Accepted

## Context

The system must work in multiple environments: local development with no cloud access, cloud deployments with vector search, and hybrid scenarios. A single retrieval backend would force all users onto one path.

## Decision

Implement three retrieval backends sharing a common interface:

1. **Keyword** (local, zero-dependency): TF-IDF with cosine similarity via scikit-learn
2. **ChromaDB** (optional): local vector search with sentence transformers
3. **Bedrock Knowledge Base** (cloud): AWS-managed vector store

All backends implement `RetrievalBackend` ABC with `search()` and `index()` methods. Factory function `get_retrieval_backend()` selects by name.

## Consequences

- New deployments work immediately with keyword backend
- Optional dependencies (`chromadb`, `sentence-transformers`) don't break package import
- Lazy imports in ChromaDBBackend constructor prevent import errors when extras not installed
- Tests run with no credentials and no optional dependencies
- Adding a new backend requires: implementing the ABC, registering in `BACKENDS`, adding to `__all__`
