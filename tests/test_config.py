"""Unit tests for core configuration and logging."""

from knowledge_assistant.core.config import Settings, get_settings


def test_default_settings():
    """Verify default configuration loads with expected defaults."""
    settings = Settings()
    assert settings.app_name == "Enterprise AI Knowledge Assistant"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.openai_model_name == "gpt-4o-mini"
    assert settings.collection_name == "enterprise_knowledge"


def test_get_settings_singleton():
    """Verify get_settings returns cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
