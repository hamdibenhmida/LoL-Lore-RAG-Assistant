"""
Embeddings generation module.

Uses Google Gemini Embedding API — no local model or ONNX runtime required.
"""

import logging
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import Config

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generate embeddings using the Google Gemini Embeddings API."""

    _embeddings = None

    def __init__(self, model_name: str = None):
        if EmbeddingsGenerator._embeddings is None:
            logger.info("Initialising Google Gemini embeddings")
            EmbeddingsGenerator._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=Config.GEMINI_API_KEY,
            )
            logger.info("Gemini embeddings ready")

    @property
    def model(self) -> GoogleGenerativeAIEmbeddings:
        return EmbeddingsGenerator._embeddings

    def embed_query(self, query: str) -> List[float]:
        if not query:
            return []
        return self.model.embed_query(query)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        return len(self.embed_query("test"))


_embeddings_generator = None


def get_embeddings_generator(model_name: str = None) -> EmbeddingsGenerator:
    global _embeddings_generator
    if _embeddings_generator is None:
        _embeddings_generator = EmbeddingsGenerator(model_name)
    return _embeddings_generator
