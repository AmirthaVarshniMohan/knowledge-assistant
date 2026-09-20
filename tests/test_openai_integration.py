"""Unit tests for OpenAI LLM Provider and LLM Factory."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field
from knowledge_assistant.rag.llm_base import MockLLMProvider
from knowledge_assistant.rag.llm_factory import LLMFactory
from knowledge_assistant.rag.openai_provider import OpenAILLMProvider


class SampleStructuredAnalysis(BaseModel):
    summary: str = Field(description="Summary of policy")
    compliance_score: int = Field(description="Score from 1 to 100")


def test_openai_provider_missing_key():
    """Verify provider raises ValueError when initialized without API key."""
    provider = OpenAILLMProvider(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        provider._get_client()


def test_openai_provider_generation_mocked():
    """Test standard OpenAI generation with mocked API response."""
    provider = OpenAILLMProvider(api_key="sk-mock-key-12345", model_name="gpt-4o-mini")

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Multi-Factor Authentication is required for all VPN connections."
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 12
    mock_response.usage.total_tokens = 37

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        response = provider.generate(
            prompt="What is the MFA policy?",
            system_prompt="You are a helpful assistant.",
        )

        assert response.content == "Multi-Factor Authentication is required for all VPN connections."
        assert response.model_name == "gpt-4o-mini"
        assert response.token_usage["total_tokens"] == 37
        mock_client.chat.completions.create.assert_called_once()


def test_openai_provider_streaming_mocked():
    """Test OpenAI streaming response with mocked generator chunks."""
    provider = OpenAILLMProvider(api_key="sk-mock-key-12345")

    # Create mock chunk generator
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="MFA "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="is required."))]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
        mock_openai_cls.return_value = mock_client

        tokens = list(provider.generate_stream("What is MFA?"))
        assert tokens == ["MFA ", "is required."]


def test_openai_structured_output_mocked():
    """Test OpenAI structured JSON parsing into Pydantic model."""
    provider = OpenAILLMProvider(api_key="sk-mock-key-12345")

    expected_schema_instance = SampleStructuredAnalysis(
        summary="Security policy mandates MFA and 90-day password rotation.",
        compliance_score=95
    )

    mock_parsed_response = MagicMock()
    mock_parsed_response.choices = [MagicMock(message=MagicMock(parsed=expected_schema_instance))]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_parsed_response
        mock_openai_cls.return_value = mock_client

        result = provider.generate_structured(
            prompt="Analyze this security policy.",
            schema_cls=SampleStructuredAnalysis,
        )

        assert isinstance(result, SampleStructuredAnalysis)
        assert result.compliance_score == 95
        assert "MFA" in result.summary


def test_llm_factory():
    """Verify factory returns appropriate provider."""
    mock_provider = LLMFactory.get_provider(provider_type="mock")
    assert isinstance(mock_provider, MockLLMProvider)

    openai_provider = LLMFactory.get_provider(provider_type="openai")
    assert isinstance(openai_provider, OpenAILLMProvider)
