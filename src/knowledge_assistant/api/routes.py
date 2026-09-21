"""FastAPI Router for RAG queries, streaming, and document management."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from knowledge_assistant.core.config import Settings, get_settings
from knowledge_assistant.api.dependencies import (
    get_ingestion_pipeline,
    get_retriever,
    get_vector_store,
)
from knowledge_assistant.api.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentStep,
    CitationItem,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from knowledge_assistant.ingestion.pipeline import IngestionPipeline
from knowledge_assistant.rag.llm_factory import LLMFactory
from knowledge_assistant.rag.pipeline import RAGPipeline
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and vector index statistics",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Return application health, version, and vector store count."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        environment=settings.environment,
        indexed_chunks=vector_store.count(),
        version="0.1.0",
    )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Submit a query to the RAG pipeline",
)
async def query_knowledge_base(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
):
    """Execute a RAG query and return grounded answer with citations."""
    start_time = time.perf_counter()
    llm = LLMFactory.get_provider(provider_type=request.provider)
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm)

    try:
        response = pipeline.run(
            question=request.question,
            k=request.k,
            min_score=request.min_score,
            filter_metadata=request.filter_metadata,
        )

        latency = (time.perf_counter() - start_time) * 1000

        citation_items = [
            CitationItem(
                file_name=c.get("file_name", "unknown"),
                page=c.get("page", 1),
                source=c.get("source"),
                score=c.get("score", 0.0),
                chunk_id=c.get("chunk_id"),
            )
            for c in response.citations
        ]

        return QueryResponse(
            question=response.question,
            answer=response.answer,
            citations=citation_items,
            has_context=response.has_context,
            model_used=response.model_used,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post(
    "/api/v1/query/stream",
    tags=["RAG"],
    summary="Stream RAG response tokens via Server-Sent Events (SSE)",
)
async def stream_query_knowledge_base(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
):
    """Stream response tokens chunk-by-chunk."""
    llm = LLMFactory.get_provider(provider_type=request.provider)
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm)

    def event_generator() -> Generator[str, None, None]:
        try:
            for token in pipeline.stream_run(
                question=request.question,
                k=request.k,
                min_score=request.min_score,
                filter_metadata=request.filter_metadata,
            ):
                # Standard SSE format: data: <token>\n\n
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/api/v1/agent/query",
    response_model=AgentQueryResponse,
    tags=["Agent"],
    summary="Submit a task to the autonomous AI Agent",
)
async def run_agent(
    request: AgentQueryRequest,
    retriever: Retriever = Depends(get_retriever),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Execute multi-step AI Agent with tool calling and intermediate step traces."""
    start_time = time.perf_counter()
    from knowledge_assistant.agents.agent import KnowledgeAgent

    agent = KnowledgeAgent(retriever=retriever, vector_store=vector_store)
    try:
        result = agent.run(request.query)
        latency = (time.perf_counter() - start_time) * 1000

        steps = [
            AgentStep(
                tool=s.get("tool", "unknown"),
                input=str(s.get("input", "")),
                output=str(s.get("output", "")),
            )
            for s in result.intermediate_steps
        ]

        return AgentQueryResponse(
            query=result.query,
            output=result.output,
            tools_used=result.tools_used,
            intermediate_steps=steps,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}",
        )


@router.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload and ingest a document (PDF, TXT, MD)",
)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Upload a file, parse into chunks, compute embeddings, and index into ChromaDB."""
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown", ".log", ".csv"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {suffix}. Allowed: {allowed_extensions}",
        )

    # Save to temporary file for parser processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chunks = ingestion_pipeline.ingest_file(tmp_path)
        # Update metadata to reflect actual uploaded filename
        for chunk in chunks:
            chunk.metadata["file_name"] = file.filename

        inserted_ids = vector_store.add_chunks(chunks)

        return DocumentUploadResponse(
            file_name=file.filename,
            chunks_created=len(inserted_ids),
            status="success",
            message=f"Successfully indexed {len(inserted_ids)} chunks into knowledge base.",
        )
    except Exception as e:
        logger.error(f"Failed to ingest uploaded document {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


@router.delete(
    "/api/v1/documents/clear",
    tags=["Documents"],
    summary="Clear all indexed documents from vector database",
)
async def clear_documents(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Wipe all vector embeddings and chunks from the active collection."""
    vector_store.clear()
    return {"status": "success", "message": "Knowledge base collection wiped successfully."}
