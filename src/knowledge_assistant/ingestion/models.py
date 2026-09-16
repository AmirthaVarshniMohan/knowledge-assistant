"""Data models for document ingestion and chunking."""

import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a raw document or document page before chunking."""
    
    content: str = Field(description="The textual content of the document or page")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the document (e.g., source, page, file_type)"
    )
    doc_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the content to detect duplicates"
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute doc_hash automatically if not provided."""
        if not self.doc_hash and self.content:
            self.doc_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
            self.metadata["doc_hash"] = self.doc_hash


class DocumentChunk(BaseModel):
    """Represents a processed chunk ready for embedding and vector storage."""
    
    chunk_id: str = Field(description="Unique identifier for the chunk (e.g. hash_index)")
    content: str = Field(description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rich metadata including source file, page, chunk index, etc."
    )
    token_count: int = Field(default=0, description="Approximate number of tokens in chunk")
