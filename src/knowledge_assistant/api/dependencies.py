"""Dependency Injection providers for FastAPI routes."""

from functools import lru_cache
from typing import Optional
from fastapi import Depends
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.ingestion.pipeline import IngestionPipeline
from knowledge_assistant.rag.embeddings import EmbeddingFactory
from knowledge_assistant.rag.llm_factory import LLMFactory
from knowledge_assistant.rag.pipeline import RAGPipeline
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


@lru_cache()
def get_vector_store() -> ChromaVectorStore:
    """Return cached singleton vector store instance."""
    settings = get_settings()
    embedder = EmbeddingFactory.get_provider()
    return ChromaVectorStore(
        collection_name=settings.collection_name,
        persist_dir=settings.vector_db_dir,
        embedding_provider=embedder,
    )


def get_retriever(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> Retriever:
    """Return Retriever instance wrapping the active vector store."""
    return Retriever(vector_store=vector_store, default_k=4)


def get_ingestion_pipeline() -> IngestionPipeline:
    """Return IngestionPipeline instance."""
    return IngestionPipeline(chunk_size=500, chunk_overlap=50)


def create_rag_pipeline(
    retriever: Retriever = Depends(get_retriever),
) -> RAGPipeline:
    """Create RAGPipeline with injected retriever."""
    llm = LLMFactory.get_provider()
    return RAGPipeline(retriever=retriever, llm_provider=llm)
