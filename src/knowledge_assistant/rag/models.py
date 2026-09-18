"""Data models for RAG retrieval and vector search."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from knowledge_assistant.ingestion.models import DocumentChunk


class SearchResult(BaseModel):
    """Represents a retrieved document chunk with similarity scoring."""

    chunk: DocumentChunk = Field(description="The retrieved document chunk")
    score: float = Field(
        default=0.0,
        description="Normalized similarity score (0.0 to 1.0, higher means more relevant)"
    )
    distance: Optional[float] = Field(
        default=None,
        description="Raw vector distance returned by the vector index"
    )

    @property
    def content(self) -> str:
        """Shortcut to chunk content."""
        return self.chunk.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Shortcut to chunk metadata."""
        return self.chunk.metadata
