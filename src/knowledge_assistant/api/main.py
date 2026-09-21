"""FastAPI Application entry point, middleware, and lifecycle configuration."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.core.logging import setup_logging
from knowledge_assistant.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan events."""
    setup_logging()
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} in [{settings.environment}] mode...")
    yield
    logger.info("Shutting down Knowledge Assistant API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Enterprise AI Knowledge & Operations Assistant with RAG, Multi-Model Routing, Agents, and MCP.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for frontend integrations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "app": settings.app_name,
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
