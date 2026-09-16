"""Document ingestion, parsing, and chunking subpackage."""

from knowledge_assistant.ingestion.models import Document, DocumentChunk
from knowledge_assistant.ingestion.parsers import (
    BaseParser,
    TextParser,
    MarkdownParser,
    PDFParser,
    DocumentParserRegistry,
)
from knowledge_assistant.ingestion.splitter import RecursiveTextSplitter
from knowledge_assistant.ingestion.pipeline import IngestionPipeline

__all__ = [
    "Document",
    "DocumentChunk",
    "BaseParser",
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "DocumentParserRegistry",
    "RecursiveTextSplitter",
    "IngestionPipeline",
]
