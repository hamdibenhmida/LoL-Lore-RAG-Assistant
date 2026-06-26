# RAG Chatbot - Project Index & Navigation

## Getting Started

### First Time Here?
**START HERE**: Read [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide

### Already Familiar?
**RUN THIS**: `uvicorn server:app --reload --port 8000` (after setting an API key in .env)

### Want All Details?
**READ THIS**: [README.md](README.md) for complete documentation

---

## Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | Quick setup guide | 5 min |
| [README.md](README.md) | Complete user documentation | 15 min |
| [ACADEMIC_REPORT.md](ACADEMIC_REPORT.md) | Technical deep dive | 20 min |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Project completion report | 10 min |

---

## Quick Setup

```bash
# Step 1: Configure
# Edit .env and set your LLM API key

# Step 2: Install
pip install -r requirements.txt

# Step 3: Run
uvicorn server:app --reload --port 8000

# Step 4: Open
# http://localhost:8000
```

---

## Project Structure

### Core Application Files
- **server.py** — FastAPI application (main entry point)
- **app.py** — Convenience launcher (runs uvicorn)
- **requirements.txt** — Python dependencies
- **.env** — Local configuration (add your API key here)
- **.env.example** — Configuration template
- **start_server.bat** — Windows start script

### Python Modules

#### Ingestion (`ingestion/`)
- **document_loader.py** — Load PDF and TXT files
- **text_splitter.py** — Chunk documents with configurable parameters
- **embeddings.py** — Generate embeddings using SentenceTransformers

#### Vector Store (`vector_store/`)
- **chroma_db.py** — ChromaDB persistent storage, sync-with-disk

#### Retrieval (`retrieval/`)
- **retriever.py** — Semantic similarity search and ranking

#### Generation (`generation/`)
- **llm_factory.py** — Provider selection and LLM instantiation
- **llm.py** — Groq API integration
- **gemini_llm.py** — Google Gemini integration
- **openrouter_llm.py** — OpenRouter integration (200+ models)
- **prompt_template.py** — Prompt engineering and response validation
- **rag_chain.py** — Complete RAG pipeline orchestration

#### Utilities (`utils/`)
- **config.py** — Centralized configuration management

### Data Directories
- **data/pdfs/** — Place your PDF documents here
- **data/txt/** — Place your TXT files here
- **chroma_data/** — Generated vector store (created at first run)
- **frontend/** — Static web frontend files

---

## Key Features

**Document Processing**
- Support for PDF and TXT files
- Intelligent chunking with overlap
- Metadata extraction

**Vector Storage**
- ChromaDB with persistent local storage
- Fast semantic search
- Sync-with-disk to remove stale chunks

**LLM Integration**
- Groq, Google Gemini, and OpenRouter support
- Auto provider selection
- Configurable temperature and tokens

**REST API**
- FastAPI backend at http://localhost:8000
- `/api/chat`, `/api/reindex`, `/api/sync`, `/api/stats`, `/api/providers`

**Production Features**
- Comprehensive error handling
- Detailed logging
- Type hints throughout
- Configuration management

---

## RAG Pipeline

```
User Question
    ↓
Input Validation
    ↓
Query Embedding (SentenceTransformers)
    ↓
Vector Search (ChromaDB)
    ↓
Retrieve Top-K Documents
    ↓
Format Context
    ↓
Build Prompt (with instructions)
    ↓
Call LLM (Groq / Gemini / OpenRouter)
    ↓
Generate Response
    ↓
Validate Grounding
    ↓
Return Answer + Sources
```

---

## Configuration & Tuning

### Key Parameters

**Chunking**
- `CHUNK_SIZE` (default: 1000) — Characters per chunk
- `CHUNK_OVERLAP` (default: 200) — Character overlap

**Retrieval**
- `RETRIEVER_TOP_K` (default: 3) — Documents to retrieve

**LLM**
- `LLM_PROVIDER` — groq | gemini | openrouter | auto
- `LLM_TEMPERATURE` (default: 0.7) — Response creativity
- `LLM_MAX_TOKENS` (default: 1024) — Response length

Edit `.env` to change these values.

---

## Troubleshooting

### Installation Issues
Run: `python setup_validation.py`

### No Documents Found
1. Add files to `data/pdfs/` or `data/txt/`
2. Call `POST /api/reindex` or restart the server

### Slow Responses
1. Check internet connection
2. Reduce `RETRIEVER_TOP_K`
3. Reduce `LLM_MAX_TOKENS`

### API Errors
1. Verify your API key in .env
2. Check `LLM_PROVIDER` matches the key you set
3. Verify internet connection

See [QUICKSTART.md](QUICKSTART.md) for more troubleshooting.

---

## Project Checklist

### Setup
- [ ] Create virtual environment
- [ ] Install requirements.txt
- [ ] Set at least one API key in .env
- [ ] Add sample documents to data/

### Validation
- [ ] Run `python setup_validation.py`
- [ ] All checks pass

### First Run
- [ ] Start with `uvicorn server:app --reload --port 8000`
- [ ] Server accessible at http://localhost:8000
- [ ] Can submit a question via the web UI or API
- [ ] Get a grounded response

---

## Common Questions

**Q: Where do I add documents?**
A: Place PDF files in `data/pdfs/` and TXT files in `data/txt/`

**Q: Where do I set the API key?**
A: Edit `.env` and set `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`

**Q: How do I switch LLM providers?**
A: Set `LLM_PROVIDER=groq|gemini|openrouter|auto` in `.env`

**Q: How do I add new documents without restarting?**
A: Call `POST /api/reindex` while the server is running

**Q: How do I remove documents that were deleted from disk?**
A: Call `POST /api/sync` — it cleans up stale chunks automatically

---

## Support

- **Quick Issues**: Check [QUICKSTART.md](QUICKSTART.md) troubleshooting
- **Usage Questions**: See [README.md](README.md)
- **Technical Details**: Read [ACADEMIC_REPORT.md](ACADEMIC_REPORT.md)
- **Validation**: Run `python setup_validation.py`

---

**LoL Lore RAG Assistant v1.0** | Built with LangChain, ChromaDB, FastAPI, and multi-provider LLM support
