"""Document Ingestion Pipeline orchestrator."""

from pathlib import Path
from typing import List, Optional, Set, Union
from loguru import logger
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.ingestion.parsers import DocumentParserRegistry
from knowledge_assistant.ingestion.splitter import RecursiveTextSplitter


class IngestionPipeline:
    """End-to-end ingestion pipeline from raw files to embedded-ready chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parser_registry: Optional[DocumentParserRegistry] = None,
        splitter: Optional[RecursiveTextSplitter] = None,
    ):
        self.parser_registry = parser_registry or DocumentParserRegistry()
        self.splitter = splitter or RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.processed_hashes: Set[str] = set()

    def ingest_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Parse and chunk a single file."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")

        logger.info(f"Ingesting file: {path.name}")
        parser = self.parser_registry.get_parser(path)
        documents = parser.parse(path)

        # Filter out documents whose hash has already been processed in this session
        new_documents = []
        for doc in documents:
            if doc.doc_hash and doc.doc_hash in self.processed_hashes:
                logger.debug(f"Skipping duplicate document page hash: {doc.doc_hash[:8]}")
                continue
            if doc.doc_hash:
                self.processed_hashes.add(doc.doc_hash)
            new_documents.append(doc)

        if not new_documents:
            logger.info(f"No new content to process for: {path.name}")
            return []

        chunks = self.splitter.split_documents(new_documents)
        logger.info(f"Generated {len(chunks)} chunks from {path.name}")
        return chunks

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[DocumentChunk]:
        """Scan a directory, parse all supported files, and return chunks."""
        directory = Path(dir_path).resolve()
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        logger.info(f"Ingesting directory: {directory} (recursive={recursive})")
        glob_pattern = "**/*" if recursive else "*"
        all_chunks: List[DocumentChunk] = []

        for item in directory.glob(glob_pattern):
            if item.is_file() and not item.name.startswith("."):
                try:
                    chunks = self.ingest_file(item)
                    all_chunks.extend(chunks)
                except ValueError:
                    # File type not supported by registered parsers, skip silently
                    logger.debug(f"Skipping unsupported file: {item.name}")
                except Exception as e:
                    logger.warning(f"Error ingesting {item.name}: {e}")

        logger.info(f"Directory ingestion complete. Total chunks generated: {len(all_chunks)}")
        return all_chunks
