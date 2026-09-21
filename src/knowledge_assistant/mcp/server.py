"""Model Context Protocol (MCP) Server for Knowledge Assistant."""

import json
import sys
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
from knowledge_assistant.agents.tools import SafeCalculator
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.mcp.models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPResourceDefinition,
    MCPToolDefinition,
)
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import BaseVectorStore


class EnterpriseMCPServer:
    """Production MCP Server exposing Knowledge Base retrieval and tools via standard JSON-RPC."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.retriever = retriever
        self.vector_store = vector_store
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools_and_resources()

    def _register_default_tools_and_resources(self):
        """Register the standard enterprise tools and resource endpoints."""
        # 1. Tool: search_enterprise_knowledge
        self.register_tool(
            name="search_enterprise_knowledge",
            description="Search internal company documents, security policies, and operational FAQs.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to look up in internal documentation",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top matching chunks to retrieve (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_search_knowledge,
        )

        # 2. Tool: calculate_expression
        self.register_tool(
            name="calculate_expression",
            description="Perform safe arithmetic calculations (e.g. '15 * 500' or '(1000 - 250) / 3').",
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid mathematical expression string",
                    }
                },
                "required": ["expression"],
            },
            handler=self._handle_calculate,
        )

        # 3. Tool: get_system_status
        self.register_tool(
            name="get_system_status",
            description="Inspect the operational health, vector database metrics, and indexed chunk count.",
            parameters_schema={"type": "object", "properties": {}},
            handler=self._handle_system_status,
        )

        # 4. Resource: knowledge://status
        self.register_resource(
            uri="knowledge://status",
            name="System Health & Metrics",
            description="Live system uptime, environment, and indexed chunk statistics",
            handler=self._handle_resource_status,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
    ):
        """Register an MCP tool."""
        self.tools[name] = {
            "definition": MCPToolDefinition(
                name=name,
                description=description,
                inputSchema=parameters_schema,
            ),
            "handler": handler,
        }

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable[[], Dict[str, Any]],
    ):
        """Register an MCP resource."""
        self.resources[uri] = {
            "definition": MCPResourceDefinition(
                uri=uri,
                name=name,
                description=description,
            ),
            "handler": handler,
        }

    # Tool Handlers
    def _handle_search_knowledge(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        k = int(params.get("k", 3))
        if not self.retriever:
            return "Knowledge retriever is not configured on this MCP server."

        results = self.retriever.retrieve(query=query, k=k)
        if not results:
            return "No matching documentation found."

        formatted = []
        for r in results:
            file_name = r.metadata.get("file_name", "unknown")
            page = r.metadata.get("page", 1)
            formatted.append(f"[{file_name} - Page {page}]: {r.content.strip()}")
        return "\n\n".join(formatted)

    def _handle_calculate(self, params: Dict[str, Any]) -> str:
        expression = params.get("expression", "")
        try:
            val = SafeCalculator.evaluate(expression)
            return str(val)
        except Exception as e:
            return f"Calculation error: {str(e)}"

    def _handle_system_status(self, params: Dict[str, Any]) -> str:
        settings = get_settings()
        count = self.vector_store.count() if self.vector_store else 0
        return (
            f"Application: {settings.app_name} | "
            f"Environment: {settings.environment} | "
            f"Indexed Chunks: {count}"
        )

    # Resource Handlers
    def _handle_resource_status(self) -> Dict[str, Any]:
        settings = get_settings()
        count = self.vector_store.count() if self.vector_store else 0
        return {
            "app_name": settings.app_name,
            "environment": settings.environment,
            "collection_name": settings.collection_name,
            "indexed_chunks": count,
            "status": "healthy",
        }

    def handle_message(self, request_json: str) -> str:
        """Process incoming JSON-RPC 2.0 message and return JSON response string."""
        try:
            req_data = json.loads(request_json)
            request = JSONRPCRequest(**req_data)
        except Exception as e:
            err_resp = JSONRPCResponse(
                id=None,
                error=JSONRPCError(code=-32700, message=f"Parse error: {str(e)}"),
            )
            return err_resp.model_dump_json()

        method = request.method
        req_id = request.id
        params = request.params or {}

        try:
            if method == "initialize":
                settings = get_settings()
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "knowledge-assistant-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {}, "resources": {}},
                    "instructions": f"Enterprise knowledge assistant for {settings.app_name}.",
                }
                return JSONRPCResponse(id=req_id, result=result).model_dump_json()

            elif method == "tools/list":
                tool_list = [t["definition"].model_dump() for t in self.tools.values()]
                return JSONRPCResponse(id=req_id, result={"tools": tool_list}).model_dump_json()

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                if tool_name not in self.tools:
                    return JSONRPCResponse(
                        id=req_id,
                        error=JSONRPCError(code=-32601, message=f"Tool not found: '{tool_name}'"),
                    ).model_dump_json()

                tool_handler = self.tools[tool_name]["handler"]
                output_text = tool_handler(arguments)

                return JSONRPCResponse(
                    id=req_id,
                    result={"content": [{"type": "text", "text": str(output_text)}]},
                ).model_dump_json()

            elif method == "resources/list":
                res_list = [r["definition"].model_dump() for r in self.resources.values()]
                return JSONRPCResponse(id=req_id, result={"resources": res_list}).model_dump_json()

            elif method == "resources/read":
                uri = params.get("uri")
                if uri not in self.resources:
                    return JSONRPCResponse(
                        id=req_id,
                        error=JSONRPCError(code=-32602, message=f"Resource URI not found: '{uri}'"),
                    ).model_dump_json()

                res_handler = self.resources[uri]["handler"]
                res_data = res_handler()
                return JSONRPCResponse(
                    id=req_id,
                    result={
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(res_data, indent=2),
                            }
                        ]
                    },
                ).model_dump_json()

            elif method == "ping":
                return JSONRPCResponse(id=req_id, result={}).model_dump_json()

            else:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32601, message=f"Method not found: '{method}'"),
                ).model_dump_json()

        except Exception as e:
            logger.error(f"[MCP] Internal error processing '{method}': {e}")
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32603, message=f"Internal error: {str(e)}"),
            ).model_dump_json()

    def run_stdio(self):
        """Main loop reading JSON-RPC messages from standard input and writing to standard output."""
        logger.info("Starting Knowledge Assistant MCP Server (stdio mode)...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response_json = self.handle_message(line)
            sys.stdout.write(response_json + "\n")
            sys.stdout.flush()


def main():
    """CLI entrypoint for MCP Server."""
    from knowledge_assistant.api.dependencies import get_retriever, get_vector_store
    server = EnterpriseMCPServer(
        retriever=get_retriever(),
        vector_store=get_vector_store(),
    )
    server.run_stdio()


if __name__ == "__main__":
    main()
