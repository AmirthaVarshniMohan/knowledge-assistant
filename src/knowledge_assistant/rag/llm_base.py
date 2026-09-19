"""Base LLM provider interfaces and response models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Structured response object from an LLM invocation."""

    content: str = Field(description="Generated textual response from LLM")
    model_name: str = Field(description="Name or identifier of the LLM model used")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Prompt, completion, and total token usage metrics"
    )
    raw_response: Optional[Any] = Field(
        default=None,
        description="Raw response object from provider API"
    )


class BaseLLMProvider(ABC):
    """Abstract interface for LLM text generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier of the active model."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a complete completion response for the given prompt."""
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Stream response tokens chunk-by-chunk (default fallback implementation)."""
        response = self.generate(prompt, system_prompt, temperature, max_tokens)
        for word in response.content.split(" "):
            yield word + " "


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM provider for unit testing, offline development, and CI."""

    def __init__(self, model_name: str = "mock-llm-v1"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        # If context is empty or says "No relevant context found"
        if "NO_CONTEXT_AVAILABLE" in prompt or "No relevant context found" in prompt:
            content = "I do not have sufficient information in the provided documentation to answer this question."
        elif "MFA" in prompt or "Multi-Factor Authentication" in prompt:
            content = "According to [Source: security.md, Page: 1], Multi-Factor Authentication (MFA) is mandatory for corporate email accounts, VPN tunnels, and code repositories."
        elif "stipend" in prompt or "equipment" in prompt:
            content = "According to [Source: faq.txt, Page: 1], full-time employees receive a one-time $500 home office equipment stipend within 60 days of their start date."
        else:
            content = f"Based on the provided documents: The answer relates to your query: {prompt[:80]}..."

        return LLMResponse(
            content=content,
            model_name=self.model_name,
            token_usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split()), "total_tokens": len(prompt.split()) + len(content.split())}
        )
