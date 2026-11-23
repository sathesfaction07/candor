import pytest

from sme_bridge.models import Chunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk-001",
            doc_id="sop-product-safety.pdf",
            text="Button-cell batteries must be secured in battery compartments that require "
            "a tool to open. Products with accessible button-cell batteries are prohibited.",
            page=1,
            section="Battery Safety Requirements",
            metadata={"category": "battery_products"},
        ),
        Chunk(
            chunk_id="chunk-002",
            doc_id="sop-product-safety.pdf",
            text="Corded window coverings with accessible cords present a strangulation hazard. "
            "All such products must comply with safety standards or use cordless mechanisms.",
            page=2,
            section="Window Covering Safety",
            metadata={"category": "window_coverings"},
        ),
        Chunk(
            chunk_id="chunk-003",
            doc_id="sop-textile-safety.pdf",
            text="Infant sleepwear must meet flammability standards. Polyester fleece garments "
            "must be labeled as either flame-resistant or tight-fitting.",
            page=1,
            section="Textile Flammability",
            metadata={"category": "infant_products"},
        ),
        Chunk(
            chunk_id="chunk-004",
            doc_id="sop-general.pdf",
            text="All products must include clear safety warnings in the primary language "
            "of the target market. Warnings must be legible and durable.",
            page=5,
            section="General Requirements",
            metadata={"category": "general"},
        ),
    ]


@pytest.fixture
def tiny_corpus_text():
    return """Safety Protocol Document

Battery Products
Products containing button-cell batteries require secure compartments.

Window Coverings
Corded products must meet safety standards to prevent strangulation.

Textile Products
Infant sleepwear must meet flammability requirements.
"""


@pytest.fixture
def retrieved_chunks():
    from sme_bridge.models import RetrievedChunk

    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="chunk-001",
                doc_id="safety.pdf",
                text=(
                    "Button-cell batteries must be secured. Products with "
                    "accessible batteries are prohibited."
                ),
                page=1,
            ),
            score=0.85,
        ),
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="chunk-002",
                doc_id="safety.pdf",
                text="Window coverings with cords present a strangulation hazard.",
                page=2,
            ),
            score=0.65,
        ),
    ]
