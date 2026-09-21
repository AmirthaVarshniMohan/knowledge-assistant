"""Unit tests for AI Agent, Safe Calculator, Custom Tools, and Agent API endpoint."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from knowledge_assistant.agents.tools import (
    SafeCalculator,
    calculate,
    create_knowledge_tool,
    create_system_status_tool,
)
from knowledge_assistant.agents.agent import KnowledgeAgent
from knowledge_assistant.api.dependencies import get_vector_store
from knowledge_assistant.api.main import create_app
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


def test_safe_calculator_valid_expressions():
    """Verify safe calculator handles arithmetic operations properly."""
    assert SafeCalculator.evaluate("15 * 500") == 7500.0
    assert SafeCalculator.evaluate("(1000 - 250) / 3") == 250.0
    assert SafeCalculator.evaluate("2 ** 4") == 16.0
    assert SafeCalculator.evaluate("10 % 3") == 1.0


def test_safe_calculator_rejects_unsafe_code():
    """Verify calculator blocks malicious code injection."""
    with pytest.raises(ValueError):
        SafeCalculator.evaluate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        SafeCalculator.evaluate("open('/etc/passwd').read()")


def test_calculate_tool_invocation():
    """Test calculate LangChain tool wrapper."""
    assert calculate.invoke({"expression": "12 * 500"}) == "6000.0"
    err_res = calculate.invoke({"expression": "bad_syntax +++"})
    assert "Calculation error" in err_res


def test_knowledge_and_system_tools(tmp_path: Path):
    """Test custom knowledge search and system inspector tools."""
    db_path = tmp_path / "agent_tool_chroma"
    store = ChromaVectorStore(
        collection_name="agent_tools_col",
        persist_dir=db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="chk_1",
            content="Full-time employees receive a $500 home office equipment stipend.",
            metadata={"file_name": "stipend_policy.md", "page": 1},
        )
    ])
    retriever = Retriever(vector_store=store, default_k=1)

    # 1. Knowledge Tool
    knowledge_tool = create_knowledge_tool(retriever)
    res = knowledge_tool.invoke({"query": "equipment stipend"})
    assert "$500" in res
    assert "stipend_policy.md" in res

    # 2. System Status Tool
    status_tool = create_system_status_tool(store)
    stat = status_tool.invoke({})
    assert "Total Indexed Chunks: 1" in stat


def test_knowledge_agent_multi_step_reasoning(tmp_path: Path):
    """Test agent multi-step workflow combining policy lookup and arithmetic calculation."""
    db_path = tmp_path / "agent_reason_chroma"
    store = ChromaVectorStore(
        collection_name="reason_col",
        persist_dir=db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="c1",
            content="Employees receive a $500 home office equipment stipend.",
            metadata={"file_name": "faq.txt", "page": 1},
        )
    ])
    retriever = Retriever(vector_store=store, default_k=1)
    agent = KnowledgeAgent(retriever=retriever, vector_store=store)

    response = agent.run("What is the equipment stipend total for 15 employees?")
    assert "$7,500" in response.output or "7500" in response.output
    assert "search_enterprise_knowledge" in response.tools_used
    assert "calculator" in response.tools_used
    assert len(response.intermediate_steps) >= 2


def test_agent_api_endpoint(tmp_path: Path):
    """Test POST /api/v1/agent/query endpoint."""
    app = create_app()
    db_path = tmp_path / "agent_api_chroma"
    store = ChromaVectorStore(
        collection_name="agent_api_col",
        persist_dir=db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )
    app.dependency_overrides[get_vector_store] = lambda: store

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/agent/query",
            json={"query": "What is the system status and total indexed documents?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "output" in data
        assert "tools_used" in data
        assert "intermediate_steps" in data
        assert data["latency_ms"] >= 0.0

    app.dependency_overrides.clear()
