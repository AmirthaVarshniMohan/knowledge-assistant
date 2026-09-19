"""Unit tests for the end-to-end RAG Pipeline."""

from pathlib import Path
import pytest
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.llm_base import MockLLMProvider
from knowledge_assistant.rag.models import SearchResult
from knowledge_assistant.rag.prompts import format_context
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore
from knowledge_assistant.rag.pipeline import RAGPipeline, RAGResponse


def test_format_context_with_chunks():
    """Verify format_context creates ordered, numbered blocks with metadata."""
    chunk1 = DocumentChunk(
        chunk_id="chk_1",
        content="Password must have at least 16 characters.",
        metadata={"file_name": "security.md", "page": 2},
    )
    search_res = [SearchResult(chunk=chunk1, score=0.92, distance=0.08)]

    formatted = format_context(search_res)
    assert "[1] Document: security.md | Page: 2 | Relevance: 0.92" in formatted
    assert "Password must have at least 16 characters." in formatted


def test_format_context_empty():
    """Verify format_context handles empty results safely."""
    formatted = format_context([])
    assert "NO_CONTEXT_AVAILABLE" in formatted


def test_rag_pipeline_end_to_end(tmp_path: Path):
    """Verify full RAG pipeline flow from vector retrieval to LLM answer generation."""
    db_path = tmp_path / "rag_test_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="rag_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )

    # Ingest test knowledge
    chunks = [
        DocumentChunk(
            chunk_id="sec_1",
            content="Multi-Factor Authentication (MFA) is mandatory for corporate login.",
            metadata={"file_name": "security.md", "page": 1, "source": "/docs/security.md"},
        ),
        DocumentChunk(
            chunk_id="faq_1",
            content="Employees receive a $500 home office equipment stipend upon joining.",
            metadata={"file_name": "faq.txt", "page": 1, "source": "/docs/faq.txt"},
        ),
    ]
    store.add_chunks(chunks)

    retriever = Retriever(vector_store=store, default_k=2)
    mock_llm = MockLLMProvider(model_name="mock-enterprise-llm")
    pipeline = RAGPipeline(retriever=retriever, llm_provider=mock_llm)

    # Ask question
    response = pipeline.run("What is the MFA policy?")

    assert isinstance(response, RAGResponse)
    assert response.has_context is True
    assert "Multi-Factor Authentication" in response.answer or "MFA" in response.answer
    assert len(response.citations) >= 1
    assert response.citations[0]["file_name"] == "security.md"
    assert response.model_used == "mock-enterprise-llm"
    assert response.token_usage is not None


def test_rag_pipeline_empty_knowledge_fallback(tmp_path: Path):
    """Verify RAG pipeline handles queries with no matching documents gracefully."""
    db_path = tmp_path / "empty_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="empty_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )

    retriever = Retriever(vector_store=store, default_k=2)
    mock_llm = MockLLMProvider()
    pipeline = RAGPipeline(retriever=retriever, llm_provider=mock_llm)

    response = pipeline.run("What is quantum computing?")
    assert response.has_context is False
    assert len(response.citations) == 0
    assert "sufficient information" in response.answer.lower()


def test_rag_pipeline_streaming(tmp_path: Path):
    """Verify streaming RAG response generator."""
    db_path = tmp_path / "stream_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="stream_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="c1",
            content="MFA is mandatory.",
            metadata={"file_name": "security.md", "page": 1},
        )
    ])

    retriever = Retriever(vector_store=store, default_k=1)
    pipeline = RAGPipeline(retriever=retriever, llm_provider=MockLLMProvider())

    stream = list(pipeline.stream_run("What is MFA?"))
    assert len(stream) > 0
    full_text = "".join(stream)
    assert "MFA" in full_text
