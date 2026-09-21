"""Unit and integration tests for FastAPI backend."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from knowledge_assistant.api.dependencies import get_vector_store
from knowledge_assistant.api.main import create_app
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.vector_store import ChromaVectorStore


@pytest.fixture
def client(tmp_path: Path):
    """TestClient fixture with isolated temporary ChromaDB vector store."""
    app = create_app()
    test_db_path = tmp_path / "api_test_chroma"
    test_store = ChromaVectorStore(
        collection_name="api_test_col",
        persist_dir=test_db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )

    app.dependency_overrides[get_vector_store] = lambda: test_store

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks" in data


def test_root_endpoint(client: TestClient):
    """Verify root / returns 200 with documentation paths."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_upload_document_and_query_flow(client: TestClient):
    """Test uploading a document and subsequent RAG querying."""
    # 1. Upload sample policy file
    file_content = b"# Remote Work Policy\n\nEmployees receive a $500 home office equipment stipend."
    file_obj = io.BytesIO(file_content)
    
    upload_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("remote_policy.md", file_obj, "text/markdown")},
    )
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["status"] == "success"
    assert upload_data["chunks_created"] >= 1

    # 2. Query knowledge base
    query_response = client.post(
        "/api/v1/query",
        json={"question": "What is the equipment stipend?", "provider": "mock"},
    )
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert query_data["question"] == "What is the equipment stipend?"
    assert query_data["has_context"] is True
    assert len(query_data["citations"]) >= 1
    assert query_data["latency_ms"] >= 0.0


def test_query_streaming_endpoint(client: TestClient):
    """Test SSE streaming query endpoint."""
    response = client.post(
        "/api/v1/query/stream",
        json={"question": "What is the MFA policy?", "provider": "mock"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data:" in text
    assert "data: [DONE]" in text


def test_upload_unsupported_format(client: TestClient):
    """Test upload endpoint rejects unsupported binary files with 400."""
    fake_exe = io.BytesIO(b"binary content")
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", fake_exe, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_clear_documents(client: TestClient):
    """Test deleting all documents in vector store."""
    response = client.delete("/api/v1/documents/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
