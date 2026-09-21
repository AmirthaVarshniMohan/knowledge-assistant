"""Data models and schemas for Model Context Protocol (MCP) and JSON-RPC 2.0."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request."""

    jsonrpc: str = Field(default="2.0")
    id: Optional[Union[str, int]] = Field(default=None)
    method: str = Field(description="Method name (e.g. tools/list, tools/call, resources/list)")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 Error response."""

    code: int = Field(description="Error code")
    message: str = Field(description="Error message")
    data: Optional[Any] = Field(default=None)


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response."""

    jsonrpc: str = Field(default="2.0")
    id: Optional[Union[str, int]] = Field(default=None)
    result: Optional[Any] = Field(default=None)
    error: Optional[JSONRPCError] = Field(default=None)


class MCPToolParameter(BaseModel):
    """JSON Schema definition for MCP tool parameter."""

    type: str = Field(default="object")
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class MCPToolDefinition(BaseModel):
    """MCP Tool metadata."""

    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human readable explanation of what the tool does")
    inputSchema: Dict[str, Any] = Field(description="JSON Schema specifying acceptable arguments")


class MCPResourceDefinition(BaseModel):
    """MCP Resource metadata."""

    uri: str = Field(description="URI identifier (e.g. knowledge://status)")
    name: str = Field(description="Name of the resource")
    description: Optional[str] = Field(default=None)
    mimeType: str = Field(default="application/json")
