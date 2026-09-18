"""Unit tests for Embeddings and ChromaVectorStore."""

from pathlib import Path
import pytest
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings, EmbeddingFactory
from knowledge_assistant.rag.vector_store import ChromaVectorStore


def test_deterministic_mock_embeddings():
    """Verify mock embeddings generate valid unit vectors with semantic alignment."""
    embedder = DeterministicMockEmbeddings(dimension=64)
    assert embedder.dimension == 64

    v1 = embedder.embed_query("Security and passwords")
    v2 = embedder.embed_query("Security and passwords")
    v3 = embedder.embed_query("Completely unrelated cafeteria food")

    assert len(v1) == 64
    assert v1 == v2  # Deterministic!

    # Dot product of identical unit vectors should equal 1.0
    dot_identical = sum(a * b for a, b in zip(v1, v2))
    assert pytest.approx(dot_identical, 0.001) == 1.0


def test_chroma_vector_store_crud(tmp_path: Path):
    """Test adding chunks, querying, and counting in ChromaVectorStore."""
    db_path = tmp_path / "test_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            content="Multi-Factor Authentication (MFA) is mandatory for corporate login.",
            metadata={"file_name": "security.md", "category": "auth"},
        ),
        DocumentChunk(
            chunk_id="c2",
            content="Full-time employees receive a $500 home office equipment stipend.",
            metadata={"file_name": "faq.txt", "category": "finance"},
        ),
    ]

    # Insert chunks
    inserted_ids = store.add_chunks(chunks)
    assert len(inserted_ids) == 2
    assert store.count() == 2

    # Query for authentication
    results = store.similarity_search("MFA login authentication", k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert "MFA" in results[0].content
    assert results[0].score > 0.5


def test_chroma_metadata_filtering(tmp_path: Path):
    """Test searching with metadata where filters."""
    db_path = tmp_path / "test_chroma_filter"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="filter_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            content="Internal server security policy rules.",
            metadata={"department": "engineering", "access": "restricted"},
        ),
        DocumentChunk(
            chunk_id="c2",
            content="Marketing branding guidelines and color palette.",
            metadata={"department": "marketing", "access": "public"},
        ),
    ]
    store.add_chunks(chunks)

    # Search with filter
    filtered_results = store.similarity_search(
        "guidelines",
        k=2,
        filter_metadata={"department": "marketing"}
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk.chunk_id == "c2"
