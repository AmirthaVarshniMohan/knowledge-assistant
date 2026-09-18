"""Embedding providers supporting local Hugging Face models, OpenAI API, and test fallbacks."""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger
from knowledge_assistant.core.config import get_settings


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single search query."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced by this model."""
        pass


class DeterministicMockEmbeddings(BaseEmbeddingProvider):
    """Fast, deterministic pseudo-embedding provider for offline testing and CI.
    
    Generates unit-normalized 64-dimensional vectors based on MD5/character hashes.
    Texts with similar tokens produce higher cosine similarity.
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for i in range(self._dimension):
                val = ((h >> (i % 32)) & 0xFF) / 255.0 - 0.5
                vec[i] += val

        # Normalize to unit sphere (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]


class SentenceTransformerEmbeddings(BaseEmbeddingProvider):
    """Local embedding provider using Hugging Face Sentence Transformers."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.hf_embedding_model
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading local SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension or 384

    def embed_query(self, text: str) -> List[float]:
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI API embedding provider using text-embedding-3-small / text-embedding-ada-002."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name
        self.api_key = api_key or settings.openai_api_key
        self._client = None
        self._dimension = 1536

    def _load_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment or passed directly.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        self._load_client()
        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_client()
        # Batch in sizes of 100 to avoid OpenAI payload limits
        results: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            results.extend([item.embedding for item in response.data])
        return results


class EmbeddingFactory:
    """Factory to create and retrieve embedding providers."""

    @staticmethod
    def get_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
        settings = get_settings()
        p_type = (provider_type or settings.environment).lower()

        if p_type in {"mock", "test"}:
            return DeterministicMockEmbeddings()
        elif p_type in {"openai"}:
            return OpenAIEmbeddingProvider()
        else:
            # Default to local HuggingFace / sentence-transformers
            return SentenceTransformerEmbeddings()
