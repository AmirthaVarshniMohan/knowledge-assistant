"""Centralized logging configuration."""

import sys
from loguru import logger
from knowledge_assistant.core.config import get_settings


def setup_logging():
    """Configure loguru logging based on application settings."""
    settings = get_settings()

    # Remove default loguru handler
    logger.remove()

    # Console handler with formatting
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Optional file logging for persistent audit in development/production
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )

    return logger
