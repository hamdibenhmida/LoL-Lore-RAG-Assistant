"""
Embeddings generation module.

Generates embeddings using SentenceTransformers model.
Provides reusable functions for embedding generation and batch processing.
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import Config

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generate embeddings using SentenceTransformers."""

    _model = None  # Class-level cache for the model

    def __init__(self, model_name: str = None):
        """
        Initialize embeddings generator.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Defaults to Config.EMBEDDING_MODEL_NAME
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME

        if EmbeddingsGenerator._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            EmbeddingsGenerator._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        else:
            logger.info("Using cached embedding model")

    @property
    def model(self) -> SentenceTransformer:
        """Get the cached model instance."""
        return EmbeddingsGenerator._model

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Text query to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not query:
            logger.warning("Empty query provided for embedding")
            return []

        embedding = self.model.encode(query, convert_to_tensor=False)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.model.encode(texts, convert_to_tensor=False)

        # Convert numpy arrays to lists
        embeddings_list = [emb.tolist() for emb in embeddings]

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings")
        return embeddings_list

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer representing the embedding dimension.
        """
        return self.model.get_sentence_embedding_dimension()


# Singleton instance for efficiency
_embeddings_generator = None


def get_embeddings_generator(model_name: str = None) -> EmbeddingsGenerator:
    """
    Get or create a singleton EmbeddingsGenerator instance.

    Args:
        model_name: Optional model name override.

    Returns:
        EmbeddingsGenerator instance.
    """
    global _embeddings_generator
    if _embeddings_generator is None:
        _embeddings_generator = EmbeddingsGenerator(model_name)
    return _embeddings_generator


if __name__ == "__main__":
    # Example usage
    embeddings = get_embeddings_generator()

    # Generate embedding for a query
    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)
    logger.info(f"Query embedding dimension: {len(query_embedding)}")

    # Generate embeddings for multiple texts
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "NLP processes human language"
    ]
    text_embeddings = embeddings.embed_texts(texts)
    logger.info(f"Generated embeddings for {len(text_embeddings)} texts")
    logger.info(f"Embedding dimension: {embeddings.get_embedding_dimension()}")
