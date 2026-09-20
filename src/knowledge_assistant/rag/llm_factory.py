"""Factory for instantiating LLM providers based on application configuration."""

from typing import Optional
from loguru import logger
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.rag.llm_base import BaseLLMProvider, MockLLMProvider
from knowledge_assistant.rag.openai_provider import OpenAILLMProvider


class LLMFactory:
    """Creates the appropriate LLM provider based on configuration or explicit request."""

    @staticmethod
    def get_provider(
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> BaseLLMProvider:
        """Resolve and instantiate the requested LLM provider."""
        settings = get_settings()
        ptype = (provider_type or settings.environment).lower()

        if ptype in {"mock", "test"}:
            logger.info("Instantiating Mock LLM provider.")
            return MockLLMProvider(model_name=model_name or "mock-enterprise-llm")
        elif ptype in {"openai"}:
            logger.info("Instantiating OpenAI LLM provider.")
            return OpenAILLMProvider(model_name=model_name or settings.openai_model_name)
        else:
            # If OPENAI_API_KEY is available in settings, default to OpenAI, otherwise fallback to Mock
            if settings.openai_api_key:
                return OpenAILLMProvider(model_name=model_name or settings.openai_model_name)
            else:
                logger.warning("No OPENAI_API_KEY detected. Using MockLLMProvider fallback.")
                return MockLLMProvider(model_name=model_name or "mock-enterprise-llm")
