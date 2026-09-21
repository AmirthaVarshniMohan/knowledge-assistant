"""RAG Evaluation orchestrator and comprehensive performance reporting."""

from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field
from knowledge_assistant.evaluation.metrics import (
    AnswerRelevanceMetric,
    CitationPrecisionMetric,
    ContextRelevanceMetric,
    GroundednessMetric,
)
from knowledge_assistant.rag.pipeline import RAGPipeline


class EvaluationItem(BaseModel):
    """A test case in the golden benchmark dataset."""

    question: str = Field(description="User test question")
    expected_keywords: List[str] = Field(default_factory=list, description="Keywords expected in the answer")
    expected_sources: List[str] = Field(default_factory=list, description="Expected source document filenames")


class EvaluationItemResult(BaseModel):
    """Evaluation scoring outcome for a single test question."""

    question: str
    answer: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    citation_precision: float
    overall_score: float
    passed: bool


class RAGEvaluationReport(BaseModel):
    """Aggregate statistics and performance scorecard for a full benchmark dataset."""

    total_test_cases: int
    passed_test_cases: int
    pass_rate: float
    mean_context_relevance: float
    mean_groundedness: float
    mean_answer_relevance: float
    mean_citation_precision: float
    mean_overall_score: float
    results: List[EvaluationItemResult]


class RAGEvaluator:
    """Evaluates RAG Pipelines against benchmark datasets across all RAG Triad dimensions."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        min_pass_score: float = 0.65,
    ):
        self.pipeline = pipeline
        self.min_pass_score = min_pass_score
        self.context_metric = ContextRelevanceMetric()
        self.groundedness_metric = GroundednessMetric()
        self.answer_relevance_metric = AnswerRelevanceMetric()
        self.citation_metric = CitationPrecisionMetric()

    def evaluate_query(self, item: EvaluationItem) -> EvaluationItemResult:
        """Run single question through pipeline and compute RAG Triad scores."""
        response = self.pipeline.run(item.question)

        c_rel = self.context_metric.evaluate(
            question=item.question,
            retrieved_contexts=response.sources,
        )
        grounded = self.groundedness_metric.evaluate(
            answer=response.answer,
            retrieved_contexts=response.sources,
        )
        a_rel = self.answer_relevance_metric.evaluate(
            question=item.question,
            answer=response.answer,
        )
        cite_prec = self.citation_metric.evaluate(
            citations=response.citations,
            retrieved_contexts=response.sources,
        )

        overall = round((c_rel + grounded + a_rel + cite_prec) / 4.0, 4)
        passed = overall >= self.min_pass_score

        return EvaluationItemResult(
            question=item.question,
            answer=response.answer,
            context_relevance=c_rel,
            groundedness=grounded,
            answer_relevance=a_rel,
            citation_precision=cite_prec,
            overall_score=overall,
            passed=passed,
        )

    def evaluate_dataset(self, dataset: List[EvaluationItem]) -> RAGEvaluationReport:
        """Run full evaluation suite over a list of test cases."""
        logger.info(f"[Evaluation] Starting benchmark run over {len(dataset)} test cases...")
        results: List[EvaluationItemResult] = []

        for item in dataset:
            res = self.evaluate_query(item)
            results.append(res)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = round(passed / total, 4) if total else 0.0

        mean_c_rel = round(sum(r.context_relevance for r in results) / total, 4) if total else 0.0
        mean_ground = round(sum(r.groundedness for r in results) / total, 4) if total else 0.0
        mean_a_rel = round(sum(r.answer_relevance for r in results) / total, 4) if total else 0.0
        mean_cite = round(sum(r.citation_precision for r in results) / total, 4) if total else 0.0
        mean_overall = round(sum(r.overall_score for r in results) / total, 4) if total else 0.0

        logger.info(
            f"[Evaluation] Finished. Pass Rate: {pass_rate*100:.1f}%, Mean Triad Score: {mean_overall:.2f}"
        )

        return RAGEvaluationReport(
            total_test_cases=total,
            passed_test_cases=passed,
            pass_rate=pass_rate,
            mean_context_relevance=mean_c_rel,
            mean_groundedness=mean_ground,
            mean_answer_relevance=mean_a_rel,
            mean_citation_precision=mean_cite,
            mean_overall_score=mean_overall,
            results=results,
        )
