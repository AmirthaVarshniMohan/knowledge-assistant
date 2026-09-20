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
from knowledge_assistant.rag.llm_base import (
    BaseLLMProvider,
    LLMResponse,
    MockLLMProvider,
)
from knowledge_assistant.rag.openai_provider import OpenAILLMProvider
from knowledge_assistant.rag.huggingface_provider import HuggingFaceLLMProvider
from knowledge_assistant.rag.llm_factory import LLMFactory
from knowledge_assistant.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    format_context,
)
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.pipeline import RAGPipeline, RAGResponse
from knowledge_assistant.rag.langchain_adapter import (
    LangChainRAGAdapter,
    LangChainVectorRetriever,
    CustomLoggingCallbackHandler,
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
    "BaseLLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "HuggingFaceLLMProvider",
    "LLMFactory",
    "DEFAULT_SYSTEM_PROMPT",
    "RAG_USER_TEMPLATE",
    "format_context",
    "Retriever",
    "RAGPipeline",
    "RAGResponse",
    "LangChainRAGAdapter",
    "LangChainVectorRetriever",
    "CustomLoggingCallbackHandler",
]
