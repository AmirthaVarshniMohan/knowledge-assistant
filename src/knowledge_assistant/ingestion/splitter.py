"""Recursive text splitter with robust token-aware chunking and metadata preservation."""

import re
from typing import List, Optional
from loguru import logger
from knowledge_assistant.ingestion.models import Document, DocumentChunk


class RecursiveTextSplitter:
    """Splits documents hierarchically into semantically sound chunks with overlap."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        use_tiktoken: bool = False,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.tokenizer = None

        if use_tiktoken:
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Could not load tiktoken ({e}), falling back to heuristic token estimation.")
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Calculate token count using tokenizer if available, or fast heuristic approximation (1 token ~ 4 chars / 0.75 words)."""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Standard NLP rule-of-thumb: ~4 characters per token in English, or words * 1.33
        words = len(text.split())
        chars = len(text)
        return max(1, max(words, chars // 4))

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators down to chunk_size."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        if separator != "":
            splits = text.split(separator)
        else:
            # Character-level fallback for giant continuous strings
            splits = list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if separator != "":
                piece = s + separator if not s.endswith(separator) else s
            else:
                piece = s

            if self.count_tokens(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if new_separators:
                    sub_splits = self._split_text(piece, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(piece)

        return self._merge_splits(good_splits)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge fragments into chunks of size chunk_size with chunk_overlap."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for piece in splits:
            piece_tokens = self.count_tokens(piece)
            
            if current_tokens + piece_tokens > self.chunk_size:
                if current_chunk:
                    chunk_str = "".join(current_chunk).strip()
                    if chunk_str:
                        chunks.append(chunk_str)

                    # Build overlap from the end of current_chunk
                    overlap_chunk: List[str] = []
                    overlap_tokens = 0
                    for p in reversed(current_chunk):
                        p_tok = self.count_tokens(p)
                        if overlap_tokens + p_tok <= self.chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_tokens += p_tok
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = overlap_tokens

            current_chunk.append(piece)
            current_tokens += piece_tokens

        if current_chunk:
            chunk_str = "".join(current_chunk).strip()
            if chunk_str and (not chunks or chunks[-1] != chunk_str):
                chunks.append(chunk_str)

        return chunks

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Split a single Document into multiple DocumentChunk instances with rich metadata."""
        raw_chunks = self._split_text(document.content, self.separators)
        total_chunks = len(raw_chunks)
        chunks: List[DocumentChunk] = []

        doc_hash = document.doc_hash or "doc"

        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            chunk_meta = {
                **document.metadata,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "parent_hash": doc_hash,
            }
            
            token_count = self.count_tokens(chunk_text)
            page_num = document.metadata.get("page", 1)
            chunk_id = f"{doc_hash[:12]}_p{page_num}_c{idx}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=chunk_meta,
                    token_count=token_count,
                )
            )

        return chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split a batch of Documents into DocumentChunk instances."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
