# LoL Lore RAG Assistant — Project Index

## Getting Started

**Try it live**: https://lol-lore-rag-assistant.onrender.com/

**Run locally**: See [QUICKSTART.md](QUICKSTART.md)

**Full documentation**: [README.md](README.md)

---

## Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | Local setup + adding PDFs | 5 min |
| [README.md](README.md) | Complete user documentation | 10 min |
| [ACADEMIC_REPORT.md](ACADEMIC_REPORT.md) | Technical deep dive | 20 min |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Project completion report | 10 min |

---

## Quick Setup

```bash
# 1. Configure
cp .env.example .env
# Add your LLM API key to .env

# 2. Install
pip install -r requirements.txt

# 3. Build vector index
python preindex.py

# 4. Run
uvicorn server:app --reload --port 8000

# 5. Open
# http://localhost:8000
```

---

## Project Structure

### Core Application Files
- **server.py** — FastAPI application (main entry point)
- **preindex.py** — Builds the ChromaDB HNSW index from PDFs (run at `docker build` time)
- **Dockerfile** — Docker build: installs deps, builds index, copies app
- **requirements.txt** — Python dependencies
- **.env.example** — Configuration template

### Python Modules

#### Ingestion (`ingestion/`)
- **document_loader.py** — Load PDF files with metadata
- **text_splitter.py** — Chunk documents with configurable size/overlap
- **embeddings.py** — fastembed ONNX embedding generation (BAAI/bge-small-en-v1.5)

#### Vector Store (`vector_store/`)
- **chroma_db.py** — ChromaDB persistent HNSW storage and similarity search

#### Retrieval (`retrieval/`)
- **retriever.py** — Top-K semantic search and context formatting

#### Generation (`generation/`)
- **llm_factory.py** — Provider selection (Groq / Gemini / OpenRouter / auto)
- **llm.py** — Groq API integration
- **gemini_llm.py** — Google Gemini integration
- **openrouter_llm.py** — OpenRouter integration (200+ models)
- **prompt_template.py** — Prompt engineering and grounding instructions
- **rag_chain.py** — Complete RAG pipeline orchestration

#### Utilities (`utils/`)
- **config.py** — Centralized configuration management

### Data & Build Artifacts
- **data/pdfs/** — Source PDF documents (committed to git)
- **chroma_data/** — Vector index (built inside Docker, not committed to git)
- **frontend/** — Static web frontend

---

## Key Features

**Document Processing**
- PDF loading with metadata extraction
- Recursive character-based chunking
- Configurable chunk size and overlap

**Vector Storage**
- ChromaDB 1.5.9 with Rust HNSW index
- Index built on Linux at `docker build` time (no cross-platform issues)
- Fast similarity search at query time

**LLM Integration**
- Groq, Google Gemini, and OpenRouter support
- Auto provider selection
- Configurable temperature and max tokens

**REST API**
- FastAPI backend at `http://localhost:8000`
- `/api/chat`, `/api/reindex`, `/api/sync`, `/api/stats`, `/api/providers`, `/api/health`

---

## RAG Pipeline

```
User Question
    ↓
Query Embedding (fastembed ONNX)
    ↓
HNSW Similarity Search (ChromaDB)
    ↓
Retrieve Top-K Chunks
    ↓
Format Context
    ↓
Build Prompt (Context + Question + Instructions)
    ↓
Call LLM (Groq / Gemini / OpenRouter)
    ↓
Generate Grounded Response
    ↓
Return Answer + Sources
```

---

## Configuration & Tuning

**Chunking**
- `CHUNK_SIZE` (default: 1000) — characters per chunk
- `CHUNK_OVERLAP` (default: 200) — character overlap between chunks

**Retrieval**
- `RETRIEVER_TOP_K` (default: 3) — chunks to retrieve per query

**LLM**
- `LLM_PROVIDER` — groq | gemini | openrouter | auto
- `LLM_TEMPERATURE` (default: 0.7)
- `LLM_MAX_TOKENS` (default: 1024)

Edit `.env` to change these values.

---

## Common Questions

**Q: Where do I add documents?**
A: Place PDF files in `data/pdfs/`, then run `python preindex.py` locally or push to git to trigger a Render rebuild.

**Q: Where do I set the API key?**
A: Edit `.env` and set `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`.

**Q: How does the index get built?**
A: `preindex.py` runs during `docker build` on Render's Linux servers, producing a Linux-native HNSW index baked into the Docker image. The app never re-processes PDFs at runtime.

**Q: Why is the first request slow?**
A: Render's free tier sleeps after 15 min of inactivity. The first request wakes the container (~30 s). Once running, responses are fast.

**Q: How do I re-index after adding documents (locally)?**
A: Run `python preindex.py` then restart the server. Or call `POST /api/reindex` while the server is running.

---

**LoL Lore RAG Assistant** | LangChain · ChromaDB · fastembed · FastAPI · Docker · Render
