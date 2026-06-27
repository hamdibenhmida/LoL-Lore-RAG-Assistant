"""
Build the Chroma vector index using fastembed (no API key needed).

Runs during `docker build` so the Linux-native HNSW index is baked into
the image. Render loads it at startup instead of re-indexing from PDFs.

Usage (local rebuild):
    python preindex.py
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from ingestion.document_loader import DocumentLoader
from ingestion.text_splitter import TextSplitter
from ingestion.embeddings import get_embeddings_generator
from vector_store.chroma_db import ChromaVectorStore

chroma_dir = Path(Config.CHROMA_PERSIST_DIRECTORY)
if chroma_dir.exists():
    shutil.rmtree(chroma_dir)
    print(f"Cleared old index at {chroma_dir}")

print("Loading documents from data/pdfs/ ...")
loader = DocumentLoader()
docs = loader.load_all_documents()
if not docs:
    print("ERROR: No documents found in data/pdfs/")
    sys.exit(1)
print(f"Loaded {len(docs)} pages")

print("Splitting into chunks...")
splitter = TextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

print("Loading fastembed model (BAAI/bge-small-en-v1.5)...")
get_embeddings_generator()
print("Model ready")

print(f"Building Chroma HNSW index ({len(chunks)} chunks) — takes ~10 min...")
vs = ChromaVectorStore()
vs.add_documents(chunks)

stats = vs.get_collection_stats()
count = stats.get("document_count", 0)
print(f"\nDone! {count} chunks indexed at {chroma_dir}")
