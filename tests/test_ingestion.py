"""Unit tests for Document Ingestion Pipeline."""

from pathlib import Path
import pytest
from knowledge_assistant.ingestion.models import Document, DocumentChunk
from knowledge_assistant.ingestion.parsers import TextParser, MarkdownParser, DocumentParserRegistry
from knowledge_assistant.ingestion.splitter import RecursiveTextSplitter
from knowledge_assistant.ingestion.pipeline import IngestionPipeline


def test_document_hash_generation():
    """Verify SHA-256 hash is computed and stored in metadata."""
    doc = Document(content="Confidential Financial Plan")
    assert doc.doc_hash is not None
    assert len(doc.doc_hash) == 64
    assert doc.metadata.get("doc_hash") == doc.doc_hash


def test_markdown_parser(tmp_path: Path):
    """Test markdown parser extracts content and source metadata."""
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Heading 1\n\nSome paragraph text.", encoding="utf-8")

    parser = MarkdownParser()
    assert parser.can_handle(sample_file) is True

    docs = parser.parse(sample_file)
    assert len(docs) == 1
    assert "Heading 1" in docs[0].content
    assert docs[0].metadata["file_name"] == "test.md"
    assert docs[0].metadata["file_type"] == "markdown"


def test_text_splitter_chunking():
    """Test text splitting into small chunks with overlap and token counting."""
    splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=10)
    long_text = (
        "Artificial Intelligence is transforming enterprise operations. "
        "Retrieval Augmented Generation combines external knowledge bases with large language models. "
        "Agents can use tools to perform complex multi-step tasks autonomously."
    )
    doc = Document(content=long_text, metadata={"source": "ai_overview.txt", "page": 1})
    chunks = splitter.split_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.token_count > 0
        assert chunk.metadata["source"] == "ai_overview.txt"
        assert chunk.metadata["chunk_index"] is not None


def test_ingestion_pipeline_deduplication(tmp_path: Path):
    """Verify duplicate document files are not ingested multiple times."""
    test_file = tmp_path / "policy.txt"
    test_file.write_text("Password must be at least 16 characters.", encoding="utf-8")

    pipeline = IngestionPipeline(chunk_size=50, chunk_overlap=10)
    
    # First ingestion
    chunks_1 = pipeline.ingest_file(test_file)
    assert len(chunks_1) > 0

    # Second ingestion of exact same content
    chunks_2 = pipeline.ingest_file(test_file)
    assert len(chunks_2) == 0  # Deduplicated!


def test_sample_docs_directory_ingestion():
    """Verify pipeline successfully ingests the data/sample_docs folder."""
    sample_dir = Path("data/sample_docs")
    if sample_dir.exists():
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        chunks = pipeline.ingest_directory(sample_dir)
        assert len(chunks) >= 2
        file_names = {c.metadata["file_name"] for c in chunks}
        assert "security_policy.md" in file_names or "faq.txt" in file_names
