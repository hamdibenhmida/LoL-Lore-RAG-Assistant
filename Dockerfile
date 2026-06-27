FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy PDFs and indexing code first so the index layer is cached
# separately from application code changes.
COPY data/ data/
COPY ingestion/ ingestion/
COPY vector_store/ vector_store/
COPY utils/ utils/
COPY preindex.py .

# Build the Chroma HNSW index on Linux at image-build time.
# This layer is cached as long as the PDFs and embedding code don't change.
RUN python preindex.py

# Copy the rest of the application code
COPY . .

EXPOSE 8000

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
