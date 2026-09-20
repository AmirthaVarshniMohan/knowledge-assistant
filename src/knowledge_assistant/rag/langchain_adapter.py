"""LangChain LCEL orchestration adapter, conversational memory, and tracing callbacks."""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from knowledge_assistant.rag.prompts import DEFAULT_SYSTEM_PROMPT
from knowledge_assistant.rag.retriever import Retriever


class CustomLoggingCallbackHandler(BaseCallbackHandler):
    """Custom LangChain callback handler for latency and step tracing."""

    def __init__(self):
        super().__init__()
        self.start_times: Dict[str, float] = {}

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chain_name = serialized.get("name", "LCEL_Chain") if serialized else "LCEL_Chain"
        self.start_times[chain_name] = time.perf_counter()
        logger.debug(f"[LangChain] Chain started: {chain_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.debug("[LangChain] Chain execution finished successfully.")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        logger.debug(f"[LangChain] LLM generation initiated ({len(prompts)} prompts).")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        logger.debug("[LangChain] LLM generation completed.")


from pydantic import ConfigDict


class LangChainVectorRetriever(BaseRetriever):
    """Bridges Knowledge Assistant Retriever into LangChain's BaseRetriever standard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Retriever
    k: int = 4
    min_score: float = 0.0

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[LCDocument]:
        results = self.retriever.retrieve(query=query, k=self.k, min_score=self.min_score)
        lc_docs = []
        for res in results:
            lc_docs.append(
                LCDocument(
                    page_content=res.content,
                    metadata={**res.metadata, "score": res.score, "chunk_id": res.chunk.chunk_id},
                )
            )
        return lc_docs


def format_lc_docs(docs: List[LCDocument]) -> str:
    """Format LangChain documents into ordered context blocks."""
    if not docs:
        return "NO_CONTEXT_AVAILABLE"
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page", 1)
        formatted.append(f"[{idx}] Source: {file_name} (Page {page})\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


class LangChainRAGAdapter:
    """Orchestrates RAG pipelines using modern LangChain LCEL (LangChain Expression Language)."""

    def __init__(
        self,
        retriever: Retriever,
        llm: Any,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.lc_retriever = LangChainVectorRetriever(retriever=retriever)
        self.callback_handler = CustomLoggingCallbackHandler()
        self._build_chains()

    def _build_chains(self):
        """Construct standard single-turn and conversational LCEL chains."""
        # 1. Single-turn standard LCEL RAG prompt
        self.single_turn_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])

        self.single_turn_chain = (
            {"context": self.lc_retriever | format_lc_docs, "question": RunnablePassthrough()}
            | self.single_turn_prompt
            | self.llm
            | StrOutputParser()
        )

        # 2. Multi-turn Conversational LCEL RAG prompt with chat history
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])

        self.conversational_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_lc_docs(self.lc_retriever._get_relevant_documents(x["question"]))
            )
            | self.conversational_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute query through LCEL chain."""
        logger.info(f"[LCEL] Executing LangChain RAG query: '{question}'")
        config = {"callbacks": [self.callback_handler]}

        if chat_history:
            messages = []
            for role, text in chat_history:
                if role.lower() in {"user", "human"}:
                    messages.append(HumanMessage(content=text))
                else:
                    messages.append(AIMessage(content=text))

            answer = self.conversational_chain.invoke(
                {"question": question, "chat_history": messages},
                config=config,
            )
        else:
            answer = self.single_turn_chain.invoke(question, config=config)

        # Retrieve citations for provenance
        docs = self.lc_retriever._get_relevant_documents(question)
        citations = [
            {"file_name": d.metadata.get("file_name"), "page": d.metadata.get("page", 1)}
            for d in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "has_context": len(docs) > 0,
        }

    def stream_query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """Stream LCEL response tokens in real-time."""
        config = {"callbacks": [self.callback_handler]}
        if chat_history:
            messages = []
            for role, text in chat_history:
                if role.lower() in {"user", "human"}:
                    messages.append(HumanMessage(content=text))
                else:
                    messages.append(AIMessage(content=text))

            for token in self.conversational_chain.stream(
                {"question": question, "chat_history": messages},
                config=config,
            ):
                yield token
        else:
            for token in self.single_turn_chain.stream(question, config=config):
                yield token
