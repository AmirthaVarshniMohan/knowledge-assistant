"""Document parsers for extracting text and metadata from various file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from pypdf import PdfReader
from loguru import logger
from knowledge_assistant.ingestion.models import Document


class BaseParser(ABC):
    """Abstract base class for document format parsers."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Return True if this parser supports the given file extension."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> List[Document]:
        """Extract and return documents/pages from the file."""
        pass


class TextParser(BaseParser):
    """Parser for plain text files (.txt, .log, .csv)."""

    SUPPORTED_EXTENSIONS = {".txt", ".log", ".csv", ".json"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> List[Document]:
        logger.debug(f"Parsing text file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            return [
                Document(
                    content=content,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "page": 1,
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise


class MarkdownParser(BaseParser):
    """Parser for Markdown files (.md, .markdown)."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> List[Document]:
        logger.debug(f"Parsing markdown file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            return [
                Document(
                    content=content,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "file_type": "markdown",
                        "page": 1,
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Failed to read markdown file {file_path}: {e}")
            raise


class PDFParser(BaseParser):
    """Parser for PDF files (.pdf) using pypdf, extracting text per page."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> List[Document]:
        logger.debug(f"Parsing PDF file: {file_path}")
        documents: List[Document] = []
        
        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # Clean basic null bytes and excessive whitespace
                cleaned_text = text.replace("\x00", "").strip()

                if cleaned_text:
                    documents.append(
                        Document(
                            content=cleaned_text,
                            metadata={
                                "source": str(file_path.resolve()),
                                "file_name": file_path.name,
                                "file_type": "pdf",
                                "page": page_idx + 1,
                                "total_pages": total_pages,
                            },
                        )
                    )

            logger.info(f"Successfully extracted {len(documents)} pages from PDF: {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Failed to parse PDF file {file_path}: {e}")
            raise


class DocumentParserRegistry:
    """Registry to resolve the appropriate parser for a given file."""

    def __init__(self):
        self.parsers: List[BaseParser] = [
            TextParser(),
            MarkdownParser(),
            PDFParser(),
        ]

    def get_parser(self, file_path: Path) -> BaseParser:
        """Find a parser capable of handling the file."""
        for parser in self.parsers:
            if parser.can_handle(file_path):
                return parser
        raise ValueError(f"No parser available for unsupported file format: {file_path.suffix}")
