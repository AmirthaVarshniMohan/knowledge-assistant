"""Retriever interface for querying vector store and extracting source citations."""

from typing import Any, Dict, List, Optional
from loguru import logger
from knowledge_assistant.rag.models import SearchResult
from knowledge_assistant.rag.vector_store import BaseVectorStore


class Retriever:
    """Handles semantic retrieval of document chunks with scoring and citation extraction."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        default_k: int = 4,
        default_min_score: float = 0.0,
    ):
        self.vector_store = vector_store
        self.default_k = default_k
        self.default_min_score = default_min_score

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute semantic search against the vector database."""
        top_k = k or self.default_k
        score_thresh = min_score if min_score is not None else self.default_min_score

        logger.debug(f"Retrieving top {top_k} chunks for query: '{query}' (min_score={score_thresh})")
        
        results = self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter_metadata=filter_metadata,
            min_score=score_thresh,
        )

        logger.info(f"Retrieved {len(results)} relevant chunks from vector store")
        return results

    def extract_citations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract unique source citations from search results."""
        citations = []
        seen = set()

        for res in results:
            file_name = res.metadata.get("file_name", "unknown")
            page = res.metadata.get("page", 1)
            source_key = f"{file_name}:{page}"

            if source_key not in seen:
                seen.add(source_key)
                citations.append({
                    "file_name": file_name,
                    "page": page,
                    "source": res.metadata.get("source", ""),
                    "score": res.score,
                    "chunk_id": res.chunk.chunk_id,
                })

        return citations
