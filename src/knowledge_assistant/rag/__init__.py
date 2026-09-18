"""RAG (Retrieval-Augmented Generation) subpackage."""

from knowledge_assistant.rag.models import SearchResult
from knowledge_assistant.rag.embeddings import (
    BaseEmbeddingProvider,
    DeterministicMockEmbeddings,
    SentenceTransformerEmbeddings,
    OpenAIEmbeddingProvider,
    EmbeddingFactory,
)
from knowledge_assistant.rag.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
)

__all__ = [
    "SearchResult",
    "BaseEmbeddingProvider",
    "DeterministicMockEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddingProvider",
    "EmbeddingFactory",
    "BaseVectorStore",
    "ChromaVectorStore",
]
