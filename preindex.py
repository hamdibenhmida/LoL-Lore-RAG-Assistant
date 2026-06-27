"""
Run once locally to pre-build the Chroma index with Gemini embeddings.
The resulting chroma_data/ folder is committed to git so Render never
has to re-index on cold starts.

Usage:
    python preindex.py
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.config import Config

if not Config.GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

chroma_dir = Path(Config.CHROMA_PERSIST_DIRECTORY)
if chroma_dir.exists():
    shutil.rmtree(chroma_dir)
    print(f"Cleared old index at {chroma_dir}")

print("Loading PDFs and building index with Gemini embeddings...")
print("(This runs once — subsequent startups will skip this entirely)\n")

Config.validate()
from generation.rag_chain import RAGChain
chain = RAGChain()

stats = chain.vector_store.get_collection_stats()
count = stats.get("document_count", 0)
print(f"\nDone! {count} chunks indexed at {chroma_dir}")
print("\nNext steps:")
print("  git add chroma_data/")
print("  git commit -m 'Add pre-built Chroma index'")
print("  git push")
