"""Prompt templates and context formatting for grounded RAG generation."""

from typing import List
from knowledge_assistant.rag.models import SearchResult

DEFAULT_SYSTEM_PROMPT = """You are an Enterprise AI Knowledge & Operations Assistant.
Your primary objective is to provide precise, truthful, and grounded answers to user questions based exclusively on the provided documentation context.

Guidelines:
1. Strict Grounding: Rely ONLY on the facts directly stated in the context. Do NOT extrapolate, speculate, or introduce external knowledge.
2. Honest Refusal: If the provided context does not contain enough information to answer the question, state:
   "I do not have sufficient information in the provided documentation to answer this question."
3. Precise Citations: Cite your sources using the format [Source: <filename>, Page: <page>] for every factual claim.
4. Tone: Maintain a professional, concise, and helpful tone suitable for enterprise operations.
"""

RAG_USER_TEMPLATE = """Use the following context from internal enterprise documents to answer the user query.

--- START OF CONTEXT ---
{context}
--- END OF CONTEXT ---

User Question: {question}

Answer with source citations:"""


def format_context(search_results: List[SearchResult]) -> str:
    """Format retrieved SearchResult chunks into an ordered, numbered context block with source metadata."""
    if not search_results:
        return "NO_CONTEXT_AVAILABLE (No relevant documents matched the query)."

    formatted_blocks: List[str] = []
    for idx, res in enumerate(search_results, start=1):
        file_name = res.metadata.get("file_name", "unknown_document")
        page = res.metadata.get("page", 1)
        score = res.score
        
        block = (
            f"[{idx}] Document: {file_name} | Page: {page} | Relevance: {score:.2f}\n"
            f"Content:\n{res.content.strip()}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)
