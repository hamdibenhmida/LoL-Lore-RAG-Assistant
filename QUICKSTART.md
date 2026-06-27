# LoL Lore RAG Assistant — Quick Start

**Live demo**: https://lol-lore-rag-assistant.onrender.com/
_(Free tier — first request after inactivity may take ~30 s to wake up)_

---

## Run It Locally (5 minutes)

### Step 1: Get an API Key

Choose one provider:
- **Groq** (free): https://console.groq.com
- **Google Gemini** (free tier): https://aistudio.google.com/app/apikey
- **OpenRouter**: https://openrouter.ai/keys

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env and set your key, e.g. for OpenRouter:
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Build the Vector Index

```bash
python preindex.py
```

This loads the PDFs from `data/pdfs/`, chunks them, and builds a ChromaDB HNSW index using fastembed (ONNX — no GPU needed). Takes ~5–10 min depending on PDF size.

### Step 5: Run the Application

```bash
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Adding PDFs to the Knowledge Base

Drop new PDF files into `data/pdfs/`, then:

**Locally** — re-run `python preindex.py` and restart the server.

**On Render** — commit and push the new file:
```bash
git add data/pdfs/your-new-file.pdf
git commit -m "Add PDF: your-new-file.pdf"
git push
```
Render rebuilds the Docker image (~15 min), baking the updated index in. Subsequent cold starts are instant.

---

## LLM Provider Options

Set `LLM_PROVIDER` in `.env`:

| Provider | Key variable | Example models |
|----------|-------------|----------------|
| `openrouter` | `OPENROUTER_API_KEY` | openai/gpt-4o-mini |
| `groq` | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| `gemini` | `GEMINI_API_KEY` | gemini-1.5-flash |
| `auto` | (any of the above) | uses first available key |

---

## Configuration Tips

Edit `.env`:

```bash
# Faster, more focused responses
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVER_TOP_K=2

# More comprehensive responses
CHUNK_SIZE=1200
CHUNK_OVERLAP=300
RETRIEVER_TOP_K=5
```

---

## Troubleshooting

### "API key not configured"
Check `.env` exists and `LLM_PROVIDER` matches the key you set.

### "No documents loaded"
Run `python preindex.py` — it needs PDFs in `data/pdfs/` first.

### "ModuleNotFoundError"
```bash
pip install --upgrade -r requirements.txt
```

### Server won't start
```bash
# Try a different port
uvicorn server:app --reload --port 8080
```

---

## API Quick Reference

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the Ruined King?", "top_k": 3}'

# Get stats
curl http://localhost:8000/api/stats
```

---

## Documentation

| File | Purpose |
|------|---------|
| README.md | Complete feature and architecture documentation |
| QUICKSTART.md | This file — quick setup |
| ACADEMIC_REPORT.md | Technical deep dive |
