"""OpenAI LLM provider supporting chat completions, token streaming, and structured schema outputs."""

from typing import Any, Dict, Generator, List, Optional, Type, TypeVar
from pydantic import BaseModel
from loguru import logger
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.rag.llm_base import BaseLLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)


class OpenAILLMProvider(BaseLLMProvider):
    """Production LLM provider using OpenAI's API (e.g. gpt-4o-mini, gpt-4o)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self._model_name = model_name or settings.openai_model_name
        self.api_key = api_key or settings.openai_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client to avoid failure at import time if key is not configured."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in your .env file or pass it directly."
                )
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a complete completion from OpenAI."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        logger.debug(f"Invoking OpenAI model: {self._model_name} (temperature={temperature})")
        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                model_name=self._model_name,
                token_usage=usage,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Stream response tokens chunk-by-chunk."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        logger.debug(f"Starting OpenAI streaming request: {self._model_name}")
        try:
            stream = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured outputs conforming strictly to a Pydantic schema."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        logger.debug(f"Invoking OpenAI structured parse with schema: {schema_cls.__name__}")
        try:
            completion = client.beta.chat.completions.parse(
                model=self._model_name,
                messages=messages,
                response_format=schema_cls,
                temperature=temperature,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            logger.error(f"OpenAI structured parsing failed: {e}")
            raise
