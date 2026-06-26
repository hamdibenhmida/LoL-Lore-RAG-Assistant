# RAG Chatbot - League of Legends Lore Assistant

A production-quality Retrieval-Augmented Generation (RAG) system that answers questions based exclusively on the League of Legends universe lore. Built for LoL fans and designed for accurate, grounded responses about champions, factions, history, and events in Runeterra.

## Features

**Core Features**
- **Document Loading**: Support for PDF and TXT file formats
- **Semantic Search**: Intelligent document retrieval using embeddings
- **Multi-Provider LLM**: Groq, Google Gemini, and OpenRouter support
- **Persistent Vector Storage**: ChromaDB with local persistence
- **Web Interface**: FastAPI backend with HTML/JS frontend
- **Context Grounding**: Ensures answers are based exclusively on documents
- **Hallucination Prevention**: Explicit handling of missing information

**Advanced Features**
- Configurable chunk size and overlap
- Adjustable retrieval parameters (top-k)
- Retrieved document inspection
- Temperature control
- Re-index and sync endpoints
- System statistics and monitoring

## Technical Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq / Google Gemini / OpenRouter |
| **Default Model** | llama-3.3-70b-versatile (Groq) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Framework** | LangChain |
| **Backend** | FastAPI |
| **Language** | Python 3.8+ |

## Project Architecture

```
chat-ai/
├── app.py                          # Launcher (starts uvicorn)
├── server.py                       # FastAPI application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (local)
├── .env.example                    # Environment template
├── start_server.bat                # Windows start script
│
├── data/
│   ├── pdfs/                       # PDF documents directory
│   └── txt/                        # Text documents directory
│
├── frontend/                       # Static web frontend
│
├── ingestion/
│   ├── document_loader.py          # PDF/TXT loading
│   ├── text_splitter.py            # Document chunking
│   └── embeddings.py               # Embedding generation
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
├── interface/
│   └── __init__.py
│
└── utils/
    └── config.py                   # Configuration management
```

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- At least one LLM API key (Groq, Gemini, or OpenRouter)
- Virtual environment (recommended)

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your API key(s)
```

### 5. Add Documents

```bash
# Place your documents in:
# - data/pdfs/   (PDF files)
# - data/txt/    (TXT files)
cp your_notes.pdf data/pdfs/
```

## Usage

### Running the Application

```bash
# Start the FastAPI server
uvicorn server:app --reload --port 8000

# Or on Windows, double-click:
start_server.bat
```

The application will be available at `http://localhost:8000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a question to the RAG system |
| GET | `/api/stats` | Knowledge base statistics |
| POST | `/api/reindex` | Re-index documents from data/ |
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
  "temperature": 0.7,
  "provider": "groq",
  "model": "llama-3.3-70b-versatile"
}
```

## Configuration

### Environment Variables (.env)

```bash
# LLM Provider Selection
LLM_PROVIDER=groq   # groq | gemini | openrouter | auto

# Groq API
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Google Gemini
GEMINI_API_KEY=your_google_ai_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL_NAME=openai/gpt-4o-mini

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_data
COLLECTION_NAME=rag_documents

# Retrieval
RETRIEVER_TOP_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging
LOG_LEVEL=INFO
```

### Tuning Parameters

- **CHUNK_SIZE**: Larger chunks = more context, smaller chunks = more precise
- **CHUNK_OVERLAP**: Helps maintain continuity between chunks
- **RETRIEVER_TOP_K**: Number of documents to retrieve (3-5 recommended)
- **LLM_TEMPERATURE**: Lower (0.3) = more precise, Higher (0.9) = more creative
- **LLM_MAX_TOKENS**: Maximum response length

## RAG Pipeline

```
User Question
    ↓
Retriever (Semantic Search)
    ↓
Retrieve Top-K Relevant Documents
    ↓
Format Context
    ↓
Prompt Template (Question + Context)
    ↓
LLM (Groq / Gemini / OpenRouter)
    ↓
Generate Grounded Response
    ↓
Validate Response
    ↓
Return Answer + Sources
```

## System Components

### 1. Document Ingestion (`ingestion/`)

**document_loader.py** — Loads PDF and TXT files with metadata

**text_splitter.py** — Recursive character splitting with configurable size/overlap

**embeddings.py** — Sentence Transformers embedding generation with caching

### 2. Vector Store (`vector_store/`)

**chroma_db.py** — Local persistent ChromaDB storage, similarity search, sync-with-disk

### 3. Retrieval (`retrieval/`)

**retriever.py** — Semantic similarity search, Top-K document ranking, context formatting

### 4. Generation (`generation/`)

**llm_factory.py** — Selects and instantiates the correct LLM provider

**llm.py** — Groq API integration

**gemini_llm.py** — Google Gemini integration

**openrouter_llm.py** — OpenRouter integration (200+ models via OpenAI-compatible API)

**prompt_template.py** — Prompt engineering, response validation, hallucination detection

**rag_chain.py** — Complete pipeline orchestration, error handling, statistics

### 5. Backend (`server.py`)

FastAPI application exposing the RAG pipeline as a REST API with static file serving.

## Troubleshooting

### "No API key configured"
1. Check `.env` file exists in project root
2. Ensure at least one API key is set and valid
3. Verify `LLM_PROVIDER` matches the key you've set

### "No documents loaded"
1. Add PDF or TXT files to `data/pdfs/` or `data/txt/`
2. Ensure files are not corrupted
3. Call `POST /api/reindex` to trigger re-indexing

### Slow response time
1. Reduce `RETRIEVER_TOP_K`
2. Reduce `LLM_MAX_TOKENS`
3. Check internet connection (LLM API calls)

### Irrelevant results
1. Adjust `CHUNK_SIZE` (try 800-1200)
2. Modify `CHUNK_OVERLAP` (try 100-300)
3. Increase `RETRIEVER_TOP_K`

## Performance Tips

1. **Optimal Chunk Size**: 800-1200 characters
2. **Retrieval Count**: 3-5 documents usually sufficient
3. **Temperature**: 0.7 for balanced responses
4. **Max Tokens**: 1024 for detailed answers

## Limitations

- Answers limited to document content (no external knowledge)
- Response quality depends on document quality
- Vector search is semantic, not keyword-based
- Subject to LLM provider API rate limits

## License

This project is provided for educational purposes.

---

**League of Legends Lore Assistant** | Built with LangChain, ChromaDB, and FastAPI
