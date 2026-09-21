"""Unit tests for Model Context Protocol (MCP) JSON-RPC Server."""

import json
from pathlib import Path
import pytest
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.mcp.server import EnterpriseMCPServer
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


@pytest.fixture
def mcp_server(tmp_path: Path) -> EnterpriseMCPServer:
    """Fixture providing an EnterpriseMCPServer with pre-populated Chroma vector store."""
    db_path = tmp_path / "mcp_test_chroma"
    store = ChromaVectorStore(
        collection_name="mcp_collection",
        persist_dir=db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="c1",
            content="MFA is mandatory for all corporate systems and email accounts.",
            metadata={"file_name": "security_policy.md", "page": 1},
        )
    ])
    retriever = Retriever(vector_store=store, default_k=1)
    return EnterpriseMCPServer(retriever=retriever, vector_store=store)


def test_mcp_initialize(mcp_server: EnterpriseMCPServer):
    """Verify MCP initialize handshake returns server info and capabilities."""
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    res_str = mcp_server.handle_message(req)
    res = json.loads(res_str)

    assert res["id"] == 1
    assert "result" in res
    assert res["result"]["serverInfo"]["name"] == "knowledge-assistant-mcp"
    assert "tools" in res["result"]["capabilities"]


def test_mcp_tools_list(mcp_server: EnterpriseMCPServer):
    """Verify tools/list returns registered tools with JSON schemas."""
    req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    res_str = mcp_server.handle_message(req)
    res = json.loads(res_str)

    assert res["id"] == 2
    tools = res["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "search_enterprise_knowledge" in tool_names
    assert "calculate_expression" in tool_names
    assert "get_system_status" in tool_names


def test_mcp_tools_call_calculate(mcp_server: EnterpriseMCPServer):
    """Verify tools/call executes calculate_expression correctly."""
    req = json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "calculate_expression", "arguments": {"expression": "25 * 40"}},
    })
    res_str = mcp_server.handle_message(req)
    res = json.loads(res_str)

    assert res["id"] == 3
    content = res["result"]["content"]
    assert content[0]["text"] == "1000.0"


def test_mcp_tools_call_search(mcp_server: EnterpriseMCPServer):
    """Verify tools/call executes search_enterprise_knowledge."""
    req = json.dumps({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "search_enterprise_knowledge", "arguments": {"query": "MFA policy"}},
    })
    res_str = mcp_server.handle_message(req)
    res = json.loads(res_str)

    assert res["id"] == 4
    content = res["result"]["content"]
    assert "security_policy.md" in content[0]["text"]
    assert "MFA" in content[0]["text"]


def test_mcp_resources_list_and_read(mcp_server: EnterpriseMCPServer):
    """Verify resources/list and resources/read."""
    # List resources
    list_req = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "resources/list", "params": {}})
    list_res = json.loads(mcp_server.handle_message(list_req))
    assert len(list_res["result"]["resources"]) >= 1
    assert list_res["result"]["resources"][0]["uri"] == "knowledge://status"

    # Read resource
    read_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/read",
        "params": {"uri": "knowledge://status"},
    })
    read_res = json.loads(mcp_server.handle_message(read_req))
    contents = read_res["result"]["contents"]
    assert contents[0]["uri"] == "knowledge://status"
    data = json.loads(contents[0]["text"])
    assert data["status"] == "healthy"
    assert data["indexed_chunks"] == 1


def test_mcp_unknown_method_and_tool_errors(mcp_server: EnterpriseMCPServer):
    """Verify error responses for unknown methods and unknown tools."""
    # Unknown method
    bad_method_req = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "invalid/method"})
    bad_res = json.loads(mcp_server.handle_message(bad_method_req))
    assert bad_res["error"]["code"] == -32601

    # Unknown tool
    bad_tool_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    })
    bad_tool_res = json.loads(mcp_server.handle_message(bad_tool_req))
    assert bad_tool_res["error"]["code"] == -32601


def test_mcp_invalid_json_parse_error(mcp_server: EnterpriseMCPServer):
    """Verify JSON parse error returns -32700."""
    res_str = mcp_server.handle_message("{invalid json")
    res = json.loads(res_str)
    assert res["error"]["code"] == -32700
