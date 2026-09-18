"""Vector Store implementation with ChromaDB and pluggable embedding models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import BaseEmbeddingProvider, DeterministicMockEmbeddings
from knowledge_assistant.rag.models import SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and retrieval."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Index a list of document chunks and return their assigned IDs."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Perform semantic similarity search for a query string."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed chunks in the collection."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all vectors in the collection."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Production vector store using ChromaDB with cosine distance metric."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Union[str, Path]] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir = str(Path(persist_dir or settings.vector_db_dir).resolve())
        self.embedding_provider = embedding_provider or DeterministicMockEmbeddings()

        logger.info(
            f"Initializing ChromaDB vector store at: {self.persist_dir} (Collection: {self.collection_name})"
        )

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection configured with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """ChromaDB metadata only accepts primitive types (str, int, float, bool)."""
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> List[str]:
        """Embed and insert document chunks into ChromaDB in batches."""
        if not chunks:
            return []

        logger.info(f"Adding {len(chunks)} chunks to vector store collection '{self.collection_name}'")
        chunk_ids: List[str] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [self._sanitize_metadata(c.metadata) for c in batch]
            
            # Generate embeddings
            embeddings = self.embedding_provider.embed_documents(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            chunk_ids.extend(ids)

        logger.info(f"Successfully indexed {len(chunk_ids)} chunks into ChromaDB")
        return chunk_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search ChromaDB using query embeddings and compute normalized similarity scores."""
        if self.count() == 0:
            logger.warning("Vector store is empty. Returning 0 search results.")
            return []

        query_vector = self.embedding_provider.embed_query(query)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.collection.query(**query_kwargs)
        search_results: List[SearchResult] = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                # In Chroma with cosine space: distance = 1 - cosine_similarity (0 = identical, 2 = opposite)
                # Similarity score normalized between 0.0 and 1.0:
                similarity_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                if similarity_score >= min_score:
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=doc_text,
                        metadata=meta,
                        token_count=len(doc_text.split()),
                    )
                    search_results.append(
                        SearchResult(
                            chunk=chunk,
                            score=round(similarity_score, 4),
                            distance=round(dist, 4),
                        )
                    )

        # Sort descending by similarity score
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    def count(self) -> int:
        """Return count of items in collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Reset and wipe current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
