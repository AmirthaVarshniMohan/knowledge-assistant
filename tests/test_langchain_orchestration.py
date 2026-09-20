"""Unit tests for LangChain LCEL RAG Adapter and Callbacks."""

from pathlib import Path
import pytest
from langchain_core.language_models.fake import FakeListLLM
from knowledge_assistant.ingestion.models import DocumentChunk
from knowledge_assistant.rag.embeddings import DeterministicMockEmbeddings
from knowledge_assistant.rag.langchain_adapter import (
    LangChainRAGAdapter,
    LangChainVectorRetriever,
    CustomLoggingCallbackHandler,
)
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import ChromaVectorStore


@pytest.fixture
def setup_vector_store(tmp_path: Path):
    """Fixture providing populated vector store and retriever."""
    db_path = tmp_path / "lc_test_chroma"
    embedder = DeterministicMockEmbeddings(dimension=32)
    store = ChromaVectorStore(
        collection_name="lc_collection",
        persist_dir=db_path,
        embedding_provider=embedder,
    )
    store.add_chunks([
        DocumentChunk(
            chunk_id="sec_1",
            content="MFA is mandatory for all corporate email accounts and VPNs.",
            metadata={"file_name": "security_policy.md", "page": 1},
        ),
        DocumentChunk(
            chunk_id="faq_1",
            content="Employees receive a $500 home office equipment stipend.",
            metadata={"file_name": "faq.txt", "page": 1},
        ),
    ])
    retriever = Retriever(vector_store=store, default_k=2)
    return retriever


def test_langchain_vector_retriever(setup_vector_store: Retriever):
    """Verify LangChainVectorRetriever returns standard LCDocuments."""
    lc_retriever = LangChainVectorRetriever(retriever=setup_vector_store, k=2)
    docs = lc_retriever._get_relevant_documents("What is the MFA policy?")

    assert len(docs) > 0
    assert "MFA" in docs[0].page_content
    assert docs[0].metadata["file_name"] == "security_policy.md"


def test_langchain_rag_adapter_single_turn(setup_vector_store: Retriever):
    """Verify single-turn LCEL RAG execution."""
    fake_llm = FakeListLLM(responses=["MFA is required for all corporate emails according to the policy."])
    adapter = LangChainRAGAdapter(retriever=setup_vector_store, llm=fake_llm)

    result = adapter.query("What is the MFA policy?")
    assert result["question"] == "What is the MFA policy?"
    assert "MFA is required" in result["answer"]
    assert len(result["citations"]) > 0
    assert result["has_context"] is True


def test_langchain_rag_adapter_conversational(setup_vector_store: Retriever):
    """Verify multi-turn LCEL RAG execution with chat history."""
    fake_llm = FakeListLLM(responses=["Yes, the $500 stipend must be claimed within 60 days of starting."])
    adapter = LangChainRAGAdapter(retriever=setup_vector_store, llm=fake_llm)

    chat_history = [
        ("human", "What is the equipment stipend?"),
        ("ai", "Full-time employees receive a $500 stipend."),
    ]

    result = adapter.query(
        question="Is there a deadline to claim it?",
        chat_history=chat_history,
    )

    assert "$500 stipend" in result["answer"]
    assert result["question"] == "Is there a deadline to claim it?"


def test_langchain_rag_adapter_streaming(setup_vector_store: Retriever):
    """Verify streaming token output from LCEL chain."""
    fake_llm = FakeListLLM(responses=["MFA is mandatory."])
    adapter = LangChainRAGAdapter(retriever=setup_vector_store, llm=fake_llm)

    tokens = list(adapter.stream_query("What is MFA?"))
    assert len(tokens) > 0
    assert "".join(tokens) == "MFA is mandatory."
