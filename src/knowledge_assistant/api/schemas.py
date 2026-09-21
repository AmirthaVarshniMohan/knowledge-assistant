"""Pydantic schemas for FastAPI request and response validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for RAG question answering."""

    question: str = Field(..., min_length=2, max_length=2000, description="The user query or question")
    k: int = Field(default=4, ge=1, le=20, description="Number of top relevant chunks to retrieve")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata key-value filters (e.g. {'department': 'finance'})"
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional override for LLM provider ('openai', 'huggingface', 'mock')"
    )


class CitationItem(BaseModel):
    """Schema for individual document citation."""

    file_name: str = Field(description="Name of cited document")
    page: int = Field(default=1, description="Page number of cited document")
    source: Optional[str] = Field(default=None, description="Full file path or URI")
    score: float = Field(default=0.0, description="Relevance score (0.0 to 1.0)")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier")


class QueryResponse(BaseModel):
    """Response schema for RAG queries."""

    question: str = Field(description="The original user query")
    answer: str = Field(description="Grounded LLM-generated answer with citations")
    citations: List[CitationItem] = Field(default_factory=list, description="Source citations")
    has_context: bool = Field(description="True if matching documents were found")
    model_used: str = Field(description="Model identifier that generated the response")
    latency_ms: float = Field(description="End-to-end query latency in milliseconds")


class AgentStep(BaseModel):
    """Schema for individual tool execution steps taken by the agent."""

    tool: str = Field(description="Name of tool called")
    input: str = Field(description="Input passed to tool")
    output: str = Field(description="Output returned from tool")


class AgentQueryRequest(BaseModel):
    """Request schema for AI Agent invocations."""

    query: str = Field(..., min_length=2, max_length=2000, description="Task or question for the agent")
    provider: Optional[str] = Field(default=None, description="Optional LLM provider override")


class AgentQueryResponse(BaseModel):
    """Response schema from AI Agent execution."""

    query: str = Field(description="Original user query")
    output: str = Field(description="Synthesized final answer from agent")
    tools_used: List[str] = Field(default_factory=list, description="Tools executed")
    intermediate_steps: List[AgentStep] = Field(default_factory=list, description="Trace of thought/tool steps")
    latency_ms: float = Field(description="Execution latency in milliseconds")


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload and ingestion."""

    file_name: str = Field(description="Uploaded file name")
    chunks_created: int = Field(description="Total chunks generated and indexed into vector DB")
    status: str = Field(default="success", description="Status string")
    message: str = Field(description="Human readable confirmation")


class HealthResponse(BaseModel):
    """System health and status response."""

    status: str = Field(default="healthy", description="Application health status")
    app_name: str = Field(description="Application name")
    environment: str = Field(description="Active environment")
    indexed_chunks: int = Field(description="Count of indexed vectors currently in ChromaDB")
    version: str = Field(default="0.1.0", description="API Version")
