# Enterprise AI Knowledge & Operations Assistant

A production-grade AI Knowledge Assistant built with RAG, Multi-Model LLM routing (OpenAI + Hugging Face open-source models), LangChain Agent workflows, Model Context Protocol (MCP), and a FastAPI backend.

## 🚀 Architecture Roadmap

1. **Phase 1: Environment & Project Setup** (Current)
2. **Phase 2: Document Ingestion Pipeline** (PDF, Markdown, Text parsing & Chunking strategies)
3. **Phase 3: Embeddings & Vector Database** (Vector Store, Similarity Search, Metadata filtering)
4. **Phase 4: Retrieval-Augmented Generation (RAG)** (Context Injection, Grounded QA)
5. **Phase 5: OpenAI Integration** (Structured outputs, Function calling)
6. **Phase 6: Hugging Face Open-Source Model** (Local / Inference API fallback)
7. **Phase 7: LangChain Orchestration** (LCEL Chains, Memory, Callbacks)
8. **Phase 8: FastAPI Backend** (Streaming responses, Background tasks, OpenTelemetry/Logging)
9. **Phase 9: AI Agent & Custom Tools** (ReAct agent, Calculator, Search, Vector Retriever tool)
10. **Phase 10: Model Context Protocol (MCP)** (Standardized tool server exposing internal knowledge securely)
11. **Phase 11: Evaluation & Automated Testing** (RAG Triad metrics, Pytest suite)
12. **Phase 12: Production Readiness & Docker** (Multi-stage Dockerfile, CI/CD, deployment config)

## 🛠️ Getting Started (Phase 1)

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Setup Environment Variables
```bash
cp .env.example .env
```
