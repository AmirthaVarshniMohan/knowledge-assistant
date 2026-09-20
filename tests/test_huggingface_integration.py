"""Unit tests for Hugging Face LLM Provider and Factory resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.huggingface_provider import HuggingFaceLLMProvider
from knowledge_assistant.rag.llm_factory import LLMFactory
from knowledge_assistant.rag.pipeline import RAGPipeline
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


def test_huggingface_provider_generation_mocked():
    """Test Hugging Face chat completion with mocked API response."""
    provider = HuggingFaceLLMProvider(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_token="hf_mock_token_12345"
    )

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Based on policy: MFA is strictly required for email and VPN access."
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 30
    mock_response.usage.completion_tokens = 15
    mock_response.usage.total_tokens = 45

    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response
        mock_client_cls.return_value = mock_client

        response = provider.generate(
            prompt="What is MFA policy?",
            system_prompt="You are a security assistant.",
            temperature=0.2,
        )

        assert response.content == "Based on policy: MFA is strictly required for email and VPN access."
        assert response.model_name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert response.token_usage["total_tokens"] == 45
        mock_client.chat_completion.assert_called_once()


def test_huggingface_provider_streaming_mocked():
    """Test Hugging Face streaming response generator."""
    provider = HuggingFaceLLMProvider(api_token="hf_mock_token_12345")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="MFA "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="is active."))]

    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = iter([chunk1, chunk2])
        mock_client_cls.return_value = mock_client

        tokens = list(provider.generate_stream("What is MFA?"))
        assert tokens == ["MFA ", "is active."]


def test_huggingface_factory_resolution():
    """Verify LLMFactory resolves 'huggingface' and 'hf' to HuggingFaceLLMProvider."""
    provider_hf = LLMFactory.get_provider(provider_type="huggingface")
    assert isinstance(provider_hf, HuggingFaceLLMProvider)

    provider_short = LLMFactory.get_provider(provider_type="hf")
    assert isinstance(provider_short, HuggingFaceLLMProvider)


def test_rag_pipeline_with_huggingface_provider(tmp_path: Path):
    """Verify RAG pipeline works seamlessly with Hugging Face open-source model provider."""
    db_path = tmp_path / "hf_rag_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="hf_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="sec_1",
            content="MFA is required for all remote employees.",
            metadata={"file_name": "security.md", "page": 1},
        )
    ])

    retriever = Retriever(vector_store=store, default_k=1)
    hf_provider = HuggingFaceLLMProvider(api_token="hf_mock_token")

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "According to [Source: security.md, Page: 1], MFA is required for all remote employees."
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response
        mock_client_cls.return_value = mock_client

        pipeline = RAGPipeline(retriever=retriever, llm_provider=hf_provider)
        response = pipeline.run("Who needs MFA?")

        assert response.has_context is True
        assert len(response.citations) == 1
        assert "security.md" in response.citations[0]["file_name"]
        assert "MFA is required" in response.answer
