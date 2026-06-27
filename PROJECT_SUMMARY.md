# LoL Lore RAG Assistant — Project Summary

## Project Status: DEPLOYED & LIVE

A production RAG system for exploring the League of Legends universe lore, deployed on Render.

**Live URL**: https://lol-lore-rag-assistant.onrender.com/

---

## Deliverables

### Core Application Files
- **server.py** — FastAPI application (main entry point)
- **preindex.py** — Standalone index builder (fastembed + ChromaDB, no API key needed; runs at `docker build` time)
- **Dockerfile** — Multi-stage Docker build with layer-cached index
- **requirements.txt** — All Python dependencies
- **.env.example** — Environment template
- **README.md** — Complete user documentation
- **QUICKSTART.md** — Quick setup guide

### Ingestion Module
- **ingestion/document_loader.py** — PDF file loading with metadata
- **ingestion/text_splitter.py** — Document chunking (RecursiveCharacterTextSplitter)
- **ingestion/embeddings.py** — fastembed ONNX embedding generation (singleton, thread-safe)

### Vector Store Module
- **vector_store/chroma_db.py** — ChromaDB HNSW storage, similarity search, sync-with-disk

### Retrieval Module
- **retrieval/retriever.py** — Top-K semantic search and context formatting

### Generation Module
- **generation/llm_factory.py** — Provider selection (Groq / Gemini / OpenRouter / auto)
- **generation/llm.py** — Groq API integration
- **generation/gemini_llm.py** — Google Gemini integration
- **generation/openrouter_llm.py** — OpenRouter integration (200+ models)
- **generation/prompt_template.py** — Prompt engineering and hallucination prevention
- **generation/rag_chain.py** — Complete RAG pipeline orchestration

### Utilities Module
- **utils/config.py** — Centralized configuration management

### Documentation
- **README.md** — Feature, architecture, and usage documentation
- **QUICKSTART.md** — 5-minute local setup guide
- **ACADEMIC_REPORT.md** — Technical analysis and architecture deep dive

### Data
- **data/pdfs/** — Source PDF documents (committed to git)
- **chroma_data/** — HNSW vector index (built inside Docker on Linux, not committed to git)

---

## Architecture Overview

### Deployment Architecture

```
Git push
    ↓
Render Docker build
    ↓
RUN python preindex.py  ← builds Linux-native HNSW index (~15 min, layer-cached)
    ↓
Docker image with pre-built chroma_data/
    ↓
Container starts → loads index instantly → serves requests
```

### Request Pipeline

```
User Question
    ↓
Query Embedding (fastembed ONNX — BAAI/bge-small-en-v1.5)
    ↓
HNSW Similarity Search (ChromaDB 1.5.9)
    ↓
Retrieve Top-K Chunks (default: 3)
    ↓
Format Context
    ↓
Build Prompt (Context + Question + Grounding Instructions)
    ↓
LLM (Groq / Gemini / OpenRouter)
    ↓
Response Generation + Grounding Validation
    ↓
Return Answer + Sources via REST API
```

### Component Integration

```
┌──────────────────────────────────────┐
│   Web Frontend (HTML/JS)             │
│   (served from /frontend/)           │
└────────────┬─────────────────────────┘
             │ HTTP (port 8000)
        ┌────▼────────────────────┐
        │   FastAPI (server.py)   │
        │   /api/chat             │
        │   /api/reindex          │
        │   /api/sync             │
        │   /api/stats            │
        └────┬────────────────────┘
             │
        ┌────▼────────────────────┐
        │   RAG Chain             │
        │   (Orchestration)       │
        └────┬────┬───────────────┘
             │    │
    ┌─────────▼─┐  └──────────────────────┐
    │ Retriever  │                        │
    ├────────────┤        ┌───────────────▼────┐
    │ ChromaDB   │        │ Generation         │
    │ fastembed  │        ├────────────────────┤
    │ HNSW index │        │ LLM Factory        │
    └────────────┘        │ Prompt Template    │
                          └────────────────────┘
```

---

## Technical Specifications

### Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| LLM | Groq / Gemini / OpenRouter | Multi-provider, per-request override |
| Embeddings | fastembed (ONNX) | BAAI/bge-small-en-v1.5, ~22 MB, no GPU |
| Vector DB | ChromaDB 1.5.9 | Rust HNSW, index built on Linux at build time |
| Framework | LangChain | RAG orchestration |
| Backend | FastAPI + uvicorn | Async, background startup task |
| Deployment | Docker on Render | Python 3.11-slim, free tier (512 MB RAM) |
| PDF | pypdf | Document loading and metadata extraction |

### Configuration Defaults
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top-K Retrieval**: 3 chunks
- **Temperature**: 0.7
- **Max Tokens**: 1024
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384 dimensions)

### Performance
- Index build (docker): ~15 min (PDF loading + ONNX embedding + HNSW construction)
- Index load at startup: <2 seconds
- Query embedding: ~5 ms (ONNX CPU)
- HNSW search: <10 ms
- End-to-end chat response: 2–5 s (LLM API latency dominates)

---

## Key Engineering Decisions

### fastembed over sentence-transformers
Replaced PyTorch-based `sentence-transformers` with `fastembed` (ONNX Runtime). Result: no PyTorch dependency (~600 MB saved), faster inference on CPU, no GPU required, same embedding quality (BAAI/bge-small-en-v1.5).

### Index built at `docker build` time
ChromaDB 1.5.9 uses a Rust-based HNSW implementation that produces platform-specific binary files. A Windows-built index cannot be loaded on Linux (Render). Solving this by building the index inside `docker build` on Render's own Linux servers — the image ships with a Linux-native index, so startup is instant.

### Docker layer caching
The Dockerfile copies PDFs and embedding code before running `preindex.py`, so the index build layer is only invalidated when source documents or embedding code change — not on every application code change.

### _EmbeddingsWrapper singleton
`FastEmbedEmbeddings` from `langchain-community` has a tqdm thread-safety bug on Linux in async contexts. Replaced with a custom `_EmbeddingsWrapper` that delegates to the `EmbeddingsGenerator` singleton already loaded at startup, bypassing tqdm entirely.

---

## Setup (3 steps, locally)

1. Set an API key in `.env` (Groq, Gemini, or OpenRouter)
2. Run: `pip install -r requirements.txt && python preindex.py`
3. Run: `uvicorn server:app --reload --port 8000`

---

**Built for Generative AI Education**
**Deployed on Render — https://lol-lore-rag-assistant.onrender.com/**
