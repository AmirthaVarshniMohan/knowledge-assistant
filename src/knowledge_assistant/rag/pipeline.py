"""End-to-end RAG pipeline coordinating retrieval, prompt augmentation, and generation."""

from typing import Any, Dict, Generator, List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from knowledge_assistant.rag.llm_base import BaseLLMProvider, MockLLMProvider
from knowledge_assistant.rag.models import SearchResult
from knowledge_assistant.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    format_context,
)
from knowledge_assistant.rag.retriever import Retriever


class RAGResponse(BaseModel):
    """Structured output from a RAG pipeline execution."""

    question: str = Field(description="The original user query")
    answer: str = Field(description="Grounded LLM-generated answer")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of cited documents and page numbers"
    )
    sources: List[SearchResult] = Field(
        default_factory=list,
        description="Underlying raw retrieved chunks with scores"
    )
    has_context: bool = Field(
        default=True,
        description="Whether relevant context was found in the vector store"
    )
    model_used: str = Field(description="Name of the LLM model that generated the answer")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage stats")


class RAGPipeline:
    """Production RAG orchestrator linking document retrieval and LLM generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: Optional[BaseLLMProvider] = None,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider or MockLLMProvider()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def run(
        self,
        question: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> RAGResponse:
        """Execute full RAG workflow: Retrieve -> Format -> Augment -> Generate."""
        logger.info(f"Executing RAG pipeline for question: '{question}'")

        # 1. Retrieve relevant chunks
        search_results = self.retriever.retrieve(
            query=question,
            k=k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

        has_context = len(search_results) > 0
        citations = self.retriever.extract_citations(search_results)

        # 2. Format context and assemble prompt
        formatted_context = format_context(search_results)
        augmented_prompt = RAG_USER_TEMPLATE.format(
            context=formatted_context,
            question=question,
        )

        # 3. Generate response using configured LLM
        llm_response = self.llm_provider.generate(
            prompt=augmented_prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
        )

        logger.info(
            f"RAG query answered successfully. Model: {llm_response.model_name}, Citations: {len(citations)}"
        )

        return RAGResponse(
            question=question,
            answer=llm_response.content,
            citations=citations,
            sources=search_results,
            has_context=has_context,
            model_used=llm_response.model_name,
            token_usage=llm_response.token_usage,
        )

    def stream_run(
        self,
        question: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Stream the generated RAG response chunk by chunk."""
        search_results = self.retriever.retrieve(
            query=question,
            k=k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )
        formatted_context = format_context(search_results)
        augmented_prompt = RAG_USER_TEMPLATE.format(
            context=formatted_context,
            question=question,
        )

        for token in self.llm_provider.generate_stream(
            prompt=augmented_prompt,
            system_prompt=self.system_prompt,
        ):
            yield token
