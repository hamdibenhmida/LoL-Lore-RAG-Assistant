# LoL Lore RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based exclusively on the League of Legends universe lore. Built for LoL fans and designed for accurate, grounded responses about champions, factions, history, and events in Runeterra.

**Live demo**: https://lol-lore-rag-assistant.onrender.com/

---

## Features

**Core Features**
- **Document Loading**: PDF file ingestion and chunking
- **Semantic Search**: ONNX-based vector search with BAAI/bge-small-en-v1.5
- **Multi-Provider LLM**: Groq, Google Gemini, and OpenRouter support
- **Persistent Vector Storage**: ChromaDB with Linux-native HNSW index
- **Web Interface**: FastAPI backend with HTML/JS frontend
- **Context Grounding**: Answers are based exclusively on indexed documents
- **Hallucination Prevention**: Explicit handling of missing information

**Advanced Features**
- Configurable chunk size and overlap
- Adjustable retrieval parameters (top-k)
- Retrieved document inspection
- Temperature control
- Re-index and sync endpoints
- System statistics and monitoring

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq / Google Gemini / OpenRouter |
| **Embeddings** | fastembed — BAAI/bge-small-en-v1.5 (ONNX, ~22 MB, no GPU) |
| **Vector DB** | ChromaDB 1.5.9 (Rust HNSW) |
| **Framework** | LangChain |
| **Backend** | FastAPI + uvicorn |
| **Deployment** | Docker on Render (Python 3.11) |

---

## Project Architecture

```
chat-ai/
├── server.py                       # FastAPI application
├── preindex.py                     # Builds Chroma index at docker build time
├── Dockerfile                      # Docker build (index baked in)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
│
├── data/
│   └── pdfs/                       # PDF source documents
│
├── frontend/                       # Static web frontend
│
├── ingestion/
│   ├── document_loader.py          # PDF loading
│   ├── text_splitter.py            # Document chunking
│   └── embeddings.py               # fastembed embedding generation
│
├── vector_store/
│   └── chroma_db.py                # ChromaDB management
│
├── retrieval/
│   └── retriever.py                # Similarity search
│
├── generation/
│   ├── llm.py                      # Groq LLM integration
│   ├── gemini_llm.py               # Google Gemini integration
│   ├── openrouter_llm.py           # OpenRouter integration
│   ├── llm_factory.py              # Provider selection
│   ├── prompt_template.py          # Prompt engineering
│   └── rag_chain.py                # Complete RAG pipeline
│
└── utils/
    └── config.py                   # Configuration management
```

---

## How the Index Works

Documents are never loaded at runtime. Instead:

1. **`docker build`** runs `python preindex.py`, which loads all PDFs from `data/pdfs/`, chunks them, embeds every chunk with fastembed (ONNX), and stores the HNSW index in `chroma_data/` inside the image.
2. **At startup** the app loads the pre-built index instantly — no PDF processing, no embedding delay.
3. **Layer caching**: the index build layer is only re-run when `data/` or embedding code changes, not on every code deploy.

---

## Adding New PDFs to the Knowledge Base

1. Copy your PDF into `data/pdfs/`
2. Commit and push:
   ```bash
   git add data/pdfs/your-file.pdf
   git commit -m "Add PDF: your-file.pdf"
   git push
   ```
3. Render auto-deploys — `docker build` re-runs `preindex.py` with all PDFs (old + new). Build takes ~15 min; subsequent startups are instant.

---

## Local Development

### Prerequisites

- Python 3.11
- At least one LLM API key (Groq, Gemini, or OpenRouter)

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Copy and edit environment file
cp .env.example .env
# Add your API key(s) to .env

# Build the vector index (first time, or after adding PDFs)
python preindex.py

# Start the server
uvicorn server:app --reload --port 8000
```

The app will be available at `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a question to the RAG system |
| GET | `/api/stats` | Knowledge base statistics |
| POST | `/api/reindex` | Re-index documents from `data/` |
| POST | `/api/sync` | Remove chunks for deleted files |
| GET | `/api/providers` | List available LLM providers |
| GET | `/api/suggestions` | Get suggested questions |
| GET | `/api/health` | Health check |

### Example Chat Request

```json
POST /api/chat
{
  "query": "Who is the Ruined King?",
  "top_k": 4,
  "temperature": 0.7
}
```

---

## Configuration

### Environment Variables (.env)

```bash
# LLM Provider Selection
LLM_PROVIDER=openrouter   # groq | gemini | openrouter | auto

# Groq API
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile

# Google Gemini
GEMINI_API_KEY=your_google_ai_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL_NAME=openai/gpt-4o-mini

# LLM settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_data
COLLECTION_NAME=rag_documents

# Retrieval
RETRIEVER_TOP_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Tuning Parameters

- **CHUNK_SIZE**: Larger chunks = more context per result; smaller = more precise match
- **CHUNK_OVERLAP**: Helps maintain continuity across chunk boundaries
- **RETRIEVER_TOP_K**: Number of chunks to retrieve (3–5 recommended)
- **LLM_TEMPERATURE**: Lower (0.3) = more precise; higher (0.9) = more creative
- **LLM_MAX_TOKENS**: Maximum response length

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
Prompt Template (Question + Context)
    ↓
LLM (Groq / Gemini / OpenRouter)
    ↓
Generate Grounded Response
    ↓
Return Answer + Sources
```

---

## Troubleshooting

### "No API key configured"
1. Check `.env` file exists in project root
2. Ensure at least one API key is set and valid
3. Verify `LLM_PROVIDER` matches the key you set

### "No documents loaded" (local)
1. Add PDF files to `data/pdfs/`
2. Run `python preindex.py` to rebuild the index
3. Restart the server

### Slow cold start on Render
Render's free tier sleeps after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake — this is normal. The index itself loads instantly once the container is running.

---

## License

This project is provided for educational purposes.

---

**LoL Lore RAG Assistant** | Built with LangChain, ChromaDB, fastembed, and FastAPI | Deployed on Render
