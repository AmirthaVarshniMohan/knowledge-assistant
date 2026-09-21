"""Unit tests for RAG Evaluation metrics, Evaluator engine, and benchmark dataset."""

import json
from pathlib import Path
import pytest
from knowledge_assistant.evaluation.metrics import (
    AnswerRelevanceMetric,
    CitationPrecisionMetric,
    ContextRelevanceMetric,
    GroundednessMetric,
)
from knowledge_assistant.evaluation.evaluator import (
    EvaluationItem,
    RAGEvaluationReport,
    RAGEvaluator,
)
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.llm_base import MockLLMProvider
from knowledge_assistant.rag.models import SearchResult
from knowledge_assistant.rag.pipeline import RAGPipeline
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


def test_context_relevance_metric():
    """Test ContextRelevance scoring."""
    metric = ContextRelevanceMetric()
    chunk = DocumentChunk(
        chunk_id="c1",
        content="Multi-Factor Authentication (MFA) is mandatory for corporate login.",
        metadata={"file_name": "security.md"},
    )
    search_res = [SearchResult(chunk=chunk, score=0.9, distance=0.1)]

    score = metric.evaluate(
        question="What is the MFA policy?",
        retrieved_contexts=search_res,
    )
    assert score >= 0.7


def test_groundedness_metric():
    """Test Groundedness scoring detects hallucinated claims."""
    metric = GroundednessMetric()
    chunk = DocumentChunk(
        chunk_id="c1",
        content="Employees receive a $500 home office equipment stipend.",
        metadata={"file_name": "faq.txt"},
    )
    search_res = [SearchResult(chunk=chunk, score=0.9)]

    # Grounded answer
    grounded_ans = "Employees receive a $500 equipment stipend."
    assert metric.evaluate(answer=grounded_ans, retrieved_contexts=search_res) >= 0.8

    # Hallucinated answer with claims not in context
    hallucinated_ans = "Employees receive free gym memberships and unlimited pizza deliveries."
    assert metric.evaluate(answer=hallucinated_ans, retrieved_contexts=search_res) <= 0.3


def test_answer_relevance_metric():
    """Test AnswerRelevance scoring."""
    metric = AnswerRelevanceMetric()
    q = "What is the equipment stipend?"
    good_ans = "The equipment stipend is $500."
    bad_ans = "The weather in Seattle is rainy today."

    assert metric.evaluate(question=q, answer=good_ans) >= 0.7
    assert metric.evaluate(question=q, answer=bad_ans) <= 0.3


def test_citation_precision_metric():
    """Test CitationPrecision scoring against valid retrieved sources."""
    metric = CitationPrecisionMetric()
    chunk = DocumentChunk(chunk_id="c1", content="Text", metadata={"file_name": "security.md"})
    search_res = [SearchResult(chunk=chunk, score=0.9)]

    valid_cites = [{"file_name": "security.md", "page": 1}]
    invalid_cites = [{"file_name": "imaginary_doc.pdf", "page": 99}]

    assert metric.evaluate(citations=valid_cites, retrieved_contexts=search_res) == 1.0
    assert metric.evaluate(citations=invalid_cites, retrieved_contexts=search_res) == 0.0


def test_rag_evaluator_dataset_benchmark(tmp_path: Path):
    """Run full benchmark dataset through RAGEvaluator and verify scorecard metrics."""
    db_path = tmp_path / "eval_chroma"
    store = ChromaVectorStore(
        collection_name="eval_col",
        persist_dir=db_path,
        embedding_provider=DeterministicMockEmbeddings(dimension=32),
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="c1",
            content="Multi-Factor Authentication (MFA) is mandatory for corporate login.",
            metadata={"file_name": "security.md", "page": 1},
        ),
        DocumentChunk(
            chunk_id="c2",
            content="Full-time employees receive a $500 home office equipment stipend.",
            metadata={"file_name": "faq.txt", "page": 1},
        ),
    ])
    retriever = Retriever(vector_store=store, default_k=2)
    pipeline = RAGPipeline(retriever=retriever, llm_provider=MockLLMProvider())

    evaluator = RAGEvaluator(pipeline=pipeline, min_pass_score=0.60)

    dataset = [
        EvaluationItem(question="What is the MFA policy?", expected_sources=["security.md"]),
        EvaluationItem(question="What is the equipment stipend?", expected_sources=["faq.txt"]),
        EvaluationItem(question="What is the quantum computing policy?", expected_sources=[]),
    ]

    report = evaluator.evaluate_dataset(dataset)
    assert isinstance(report, RAGEvaluationReport)
    assert report.total_test_cases == 3
    assert report.pass_rate >= 0.66
    assert report.mean_overall_score >= 0.60
    assert len(report.results) == 3
