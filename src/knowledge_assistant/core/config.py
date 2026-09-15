"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration class for Knowledge Assistant.
    
    Reads from environment variables and an optional .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application settings
    app_name: str = Field(default="Enterprise AI Knowledge Assistant", description="Application Name")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    log_level: str = Field(default="INFO", description="Logging level")

    # LLM Settings (OpenAI)
    openai_api_key: Optional[str] = Field(default=None, description="API Key for OpenAI services")
    openai_model_name: str = Field(default="gpt-4o-mini", description="OpenAI LLM Model Name")

    # Hugging Face Settings
    huggingfacehub_api_token: Optional[str] = Field(default=None, description="Hugging Face API Token")
    hf_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Hugging Face Embedding Model")
    hf_llm_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3", description="Hugging Face open LLM")

    # Vector Database Settings
    vector_db_dir: str = Field(default="./data/vector_store", description="Directory where vector DB is persisted")
    collection_name: str = Field(default="enterprise_knowledge", description="Default vector collection name")

    # API Server Settings
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of the Settings."""
    return Settings()
