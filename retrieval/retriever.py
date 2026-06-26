"""
Document retriever for semantic similarity search.

Retrieves relevant documents from the vector store based on query similarity.
Provides ranking and filtering capabilities for retrieved documents.
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from vector_store.chroma_db import ChromaVectorStore
from utils.config import Config

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve relevant documents from vector store."""

    def __init__(
        self,
        vector_store: ChromaVectorStore = None,
        top_k: int = None
    ):
        """
        Initialize document retriever.

        Args:
            vector_store: ChromaVectorStore instance to use for retrieval.
                         Creates new instance if not provided.
            top_k: Number of documents to retrieve.
                  Defaults to Config.RETRIEVER_TOP_K
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.top_k = top_k or Config.RETRIEVER_TOP_K

        logger.info(
            f"DocumentRetriever initialized with top_k={self.top_k}"
        )

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text to search for.
            k: Number of documents to retrieve.
              Defaults to self.top_k

        Returns:
            List of tuples (Document, relevance_score).
            Scores range from 0 to 1 (higher is more relevant).
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        k = k or self.top_k

        logger.info(f"Retrieving {k} documents for query: '{query[:100]}...'")

        try:
            # Search similar documents with scores
            results = self.vector_store.search_similar(query, k=k)

            logger.info(f"Retrieved {len(results)} documents")

            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def format_retrieved_docs(
        self,
        results: List[Tuple[Document, float]]
    ) -> str:
        """
        Format retrieved documents as a single context string.

        Args:
            results: List of (Document, score) tuples.

        Returns:
            Formatted context string with all retrieved documents.
        """
        if not results:
            logger.warning("No results to format")
            return ""

        context_parts = []

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()

            # Add document separator
            context_parts.append(f"--- Document {i} (Source: {source}, Score: {score:.3f}) ---")
            context_parts.append(content)

        return "\n\n".join(context_parts)

    def retrieve_and_format(self, query: str, k: int = None) -> Tuple[str, List[Document]]:
        """
        Retrieve documents and return both formatted context and raw documents.

        Args:
            query: Query text to search for.
            k: Number of documents to retrieve.

        Returns:
            Tuple of (formatted_context, list_of_documents).
        """
        results = self.retrieve(query, k=k)

        if not results:
            return "", []

        # Extract documents from results
        documents = [doc for doc, score in results]

        # Format as context
        context = self.format_retrieved_docs(results)

        return context, documents

    def get_retrieval_stats(self) -> dict:
        """
        Get statistics about the retriever.

        Returns:
            Dictionary with retriever statistics.
        """
        return {
            "top_k": self.top_k,
            "vector_store_stats": self.vector_store.get_collection_stats()
        }


if __name__ == "__main__":
    # Example usage
    from ingestion.document_loader import DocumentLoader
    from ingestion.text_splitter import TextSplitter

    logger.info("=== Document Retriever Example ===")

    # Load and prepare documents
    loader = DocumentLoader()
    documents = loader.load_all_documents()

    if documents:
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)

        # Create vector store and retriever
        vector_store = ChromaVectorStore()
        vector_store.get_or_create_store(split_docs)

        retriever = DocumentRetriever(vector_store, top_k=3)

        # Test retrieval
        query = "What is machine learning?"
        context, docs = retriever.retrieve_and_format(query)

        logger.info(f"Query: {query}")
        logger.info(f"Retrieved {len(docs)} documents")
        logger.info("Context:")
        logger.info(context[:500])  # Show first 500 chars
