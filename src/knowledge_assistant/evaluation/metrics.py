"""Evaluation metrics for RAG Triad: Context Relevance, Groundedness, Answer Relevance, and Citations."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from knowledge_assistant.rag.models import SearchResult


class BaseMetric(ABC):
    """Abstract base class for RAG evaluation metrics."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> float:
        """Calculate a normalized score between 0.0 and 1.0."""
        pass


class ContextRelevanceMetric(BaseMetric):
    """Measures whether the retrieved context chunks contain information relevant to the user query."""

    def evaluate(self, question: str, retrieved_contexts: List[SearchResult], **kwargs: Any) -> float:
        if not retrieved_contexts:
            return 0.0

        query_tokens = set(re.findall(r"\w+", question.lower()))
        # Remove common stopwords for cleaner lexical alignment
        stopwords = {"what", "is", "the", "for", "a", "an", "and", "in", "of", "to", "how", "do", "i", "are", "does"}
        meaningful_tokens = query_tokens - stopwords
        if not meaningful_tokens:
            meaningful_tokens = query_tokens

        scores = []
        for ctx in retrieved_contexts:
            ctx_text = ctx.content.lower()
            matches = sum(1 for tok in meaningful_tokens if tok in ctx_text)
            overlap_ratio = matches / len(meaningful_tokens) if meaningful_tokens else 0.0
            # Blend retrieval vector score and token overlap
            combined = 0.5 * min(1.0, overlap_ratio) + 0.5 * min(1.0, ctx.score)
            scores.append(combined)

        return round(sum(scores) / len(scores), 4) if scores else 0.0


class GroundednessMetric(BaseMetric):
    """Measures whether the generated answer is faithful to the context without hallucinations."""

    def evaluate(self, answer: str, retrieved_contexts: List[SearchResult], **kwargs: Any) -> float:
        if not answer.strip():
            return 0.0
        
        # If model explicitly reported no context found
        if "not have sufficient information" in answer.lower():
            return 1.0

        if not retrieved_contexts:
            # Answer generated with zero context is ungrounded
            return 0.0

        combined_context = " ".join(c.content.lower() for c in retrieved_contexts)
        
        # Extract meaningful claim keywords from answer
        answer_tokens = re.findall(r"\w+", answer.lower())
        stopwords = {"according", "to", "source", "page", "the", "is", "a", "an", "in", "of", "for", "and", "with", "from", "on", "as", "by", "that", "this", "be", "are", "all"}
        claim_tokens = [t for t in answer_tokens if t not in stopwords and len(t) > 2]

        if not claim_tokens:
            return 1.0

        grounded_count = sum(1 for tok in claim_tokens if tok in combined_context)
        groundedness_score = grounded_count / len(claim_tokens)

        return round(min(1.0, max(0.0, groundedness_score)), 4)


class AnswerRelevanceMetric(BaseMetric):
    """Measures whether the generated response directly addresses the user's initial question."""

    def evaluate(self, question: str, answer: str, **kwargs: Any) -> float:
        if not answer.strip():
            return 0.0

        if "not have sufficient information" in answer.lower():
            return 1.0

        q_tokens = set(re.findall(r"\w+", question.lower()))
        stopwords = {"what", "is", "the", "for", "a", "an", "and", "in", "of", "to", "how", "do", "i", "are", "does", "who", "which"}
        key_q_tokens = q_tokens - stopwords or q_tokens

        ans_lower = answer.lower()
        matched = sum(1 for tok in key_q_tokens if tok in ans_lower)
        relevance_score = matched / len(key_q_tokens) if key_q_tokens else 1.0

        return round(min(1.0, max(0.0, relevance_score)), 4)


class CitationPrecisionMetric(BaseMetric):
    """Measures whether the answer includes valid source citations matching retrieved documents."""

    def evaluate(self, citations: List[Dict[str, Any]], retrieved_contexts: List[SearchResult], **kwargs: Any) -> float:
        if not retrieved_contexts:
            return 1.0 if not citations else 0.0

        if not citations:
            return 0.0

        valid_files = {c.metadata.get("file_name") for c in retrieved_contexts}
        valid_citations = sum(1 for cite in citations if cite.get("file_name") in valid_files)

        return round(valid_citations / len(citations), 4)
