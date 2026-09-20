"""Hugging Face Open-Source LLM Provider supporting Serverless Inference and Custom Endpoints."""

from typing import Any, Dict, Generator, List, Optional
from loguru import logger
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.rag.llm_base import BaseLLMProvider, LLMResponse


class HuggingFaceLLMProvider(BaseLLMProvider):
    """LLM provider using Hugging Face Serverless Inference API or custom TGI/vLLM endpoints."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        settings = get_settings()
        self._model_name = model_name or settings.hf_llm_model
        self.api_token = api_token or settings.huggingfacehub_api_token
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy load huggingface_hub InferenceClient."""
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self.endpoint_url or self._model_name,
                token=self.api_token,
                timeout=self.timeout,
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
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a response using Hugging Face Chat Completion API."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        logger.debug(f"Invoking Hugging Face model: {self._model_name} (temperature={temperature})")
        try:
            # Temperature must be > 0.0 for many Hugging Face models
            safe_temp = max(0.01, temperature)
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=safe_temp,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            else:
                usage = {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(prompt.split()) + len(content.split()),
                }

            return LLMResponse(
                content=content,
                model_name=self._model_name,
                token_usage=usage,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Hugging Face Inference call failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Stream response tokens chunk-by-chunk from Hugging Face Inference API."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        logger.debug(f"Starting Hugging Face streaming request: {self._model_name}")
        try:
            safe_temp = max(0.01, temperature)
            stream = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=safe_temp,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Hugging Face streaming error: {e}")
            raise
