# RAG Chatbot - Quick Start Guide

## Quick Setup (5 minutes)

### Step 1: Get an API Key

Choose one provider:
- **Groq** (free): https://console.groq.com
- **Google Gemini** (free tier): https://aistudio.google.com/app/apikey
- **OpenRouter**: https://openrouter.ai/keys

### Step 2: Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env and set your key, e.g. for Groq:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Documents

Place PDF or TXT files in:
- `data/pdfs/` — PDF files
- `data/txt/` — text files

### Step 5: Run the Application

```bash
uvicorn server:app --reload --port 8000
```

The app will be available at: **http://localhost:8000**

On Windows you can also double-click **start_server.bat**.

---

## Complete Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- At least one LLM API key
- Internet connection

### Full Installation Steps

#### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2. Upgrade pip

```bash
pip install --upgrade pip
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure .env File

```bash
cp .env.example .env
# Open .env and fill in at least one API key
```

#### 5. Prepare Documents

```bash
# Place any PDF or TXT files in:
# - data/pdfs/
# - data/txt/
```

#### 6. Validate Setup

```bash
python setup_validation.py
```

#### 7. Run Application

```bash
uvicorn server:app --reload --port 8000
```

---

## LLM Provider Options

Set `LLM_PROVIDER` in `.env` to choose your provider:

| Provider | Key variable | Example models |
|----------|-------------|----------------|
| `groq` | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| `gemini` | `GEMINI_API_KEY` | gemini-1.5-flash |
| `openrouter` | `OPENROUTER_API_KEY` | openai/gpt-4o-mini |
| `auto` | (any of the above) | uses first available key |

---

## Configuration Tips

### Performance Tuning

Edit `.env` file:

```bash
# For faster, more focused responses
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVER_TOP_K=2

# For comprehensive, detailed responses
CHUNK_SIZE=1200
CHUNK_OVERLAP=300
RETRIEVER_TOP_K=5

# For faster LLM responses
LLM_MAX_TOKENS=512

# For more detailed responses
LLM_MAX_TOKENS=2048
```

### Document Optimization

- 1,000-10,000 words per file works best
- Clear section headings improve retrieval
- Minimal formatting artifacts in PDFs

---

## Troubleshooting

### "API key not configured"

```bash
# Check .env file exists and contains a valid key
# Verify LLM_PROVIDER matches the key you set
```

### "No documents loaded"

```bash
# Add files to data/pdfs/ or data/txt/
# Then call POST /api/reindex or restart the server
```

### "ModuleNotFoundError"

```bash
pip install --upgrade -r requirements.txt
```

### Server won't start

```bash
# Check port 8000 is free
# Try a different port:
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

# Re-index documents
curl -X POST http://localhost:8000/api/reindex

# Get stats
curl http://localhost:8000/api/stats
```

---

## Debugging Steps

1. **Check logs** — console output shows detailed INFO/ERROR messages

2. **Verify configuration**
   ```bash
   python utils/config.py
   ```

3. **Test components**
   ```bash
   python -m ingestion.document_loader
   python -m ingestion.embeddings
   ```

4. **Run validation**
   ```bash
   python setup_validation.py
   ```

---

## Documentation Files

| File | Purpose |
|------|---------|
| README.md | Complete feature documentation |
| QUICKSTART.md | This file — quick setup |
| ACADEMIC_REPORT.md | Technical analysis and architecture |
| .env.example | Configuration template |

---

**Enjoy your RAG Chatbot!**
