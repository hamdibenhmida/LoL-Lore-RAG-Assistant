"""
Embeddings generation module.

Generates embeddings using FastEmbed (ONNX-based, no torch dependency).
"""

import logging
from typing import List
from fastembed import TextEmbedding
from utils.config import Config

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingsGenerator:
    """Generate embeddings using FastEmbed."""

    _model = None

    def __init__(self, model_name: str = None):
        self.model_name = model_name or _DEFAULT_MODEL

        if EmbeddingsGenerator._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            EmbeddingsGenerator._model = TextEmbedding(model_name=self.model_name)
            logger.info("Embedding model loaded successfully")
        else:
            logger.info("Using cached embedding model")

    @property
    def model(self) -> TextEmbedding:
        return EmbeddingsGenerator._model

    def embed_query(self, query: str) -> List[float]:
        if not query:
            logger.warning("Empty query provided for embedding")
            return []
        embeddings = list(self.model.embed([query]))
        return embeddings[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = [emb.tolist() for emb in self.model.embed(texts)]
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def get_embedding_dimension(self) -> int:
        sample = list(self.model.embed(["test"]))
        return len(sample[0])


_embeddings_generator = None


def get_embeddings_generator(model_name: str = None) -> EmbeddingsGenerator:
    global _embeddings_generator
    if _embeddings_generator is None:
        _embeddings_generator = EmbeddingsGenerator(model_name)
    return _embeddings_generator
