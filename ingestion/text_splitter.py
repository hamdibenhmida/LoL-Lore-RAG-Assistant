"""
Text splitting and chunking module.

Handles document chunking using RecursiveCharacterTextSplitter to create
manageable chunks for embedding and retrieval, with configurable chunk size
and overlap parameters.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import Config

logger = logging.getLogger(__name__)


class TextSplitter:
    """Split documents into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Size of each chunk in characters.
                       Defaults to Config.CHUNK_SIZE
            chunk_overlap: Overlap between chunks in characters.
                          Defaults to Config.CHUNK_OVERLAP
            separators: List of separators to split on.
                       Defaults to standard separators
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        # Default separators for recursive splitting
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(
            f"TextSplitter initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of split document chunks.
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []

        logger.info(f"Splitting {len(documents)} documents into chunks...")

        split_docs = self.text_splitter.split_documents(documents)

        logger.info(f"Documents split into {len(split_docs)} chunks")
        if split_docs:
            avg_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
            logger.info(f"Average chunk size: {avg_size:.0f} characters")

        return split_docs

    def get_split_statistics(self, documents: List[Document]) -> dict:
        """
        Generate statistics about the split documents.

        Args:
            documents: List of split documents.

        Returns:
            Dictionary containing split statistics.
        """
        if not documents:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }

        chunk_sizes = [len(doc.page_content) for doc in documents]
        total_chars = sum(chunk_sizes)

        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars / len(documents),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }


if __name__ == "__main__":
    # Example usage
    from ingestion.document_loader import DocumentLoader

    loader = DocumentLoader()
    documents = loader.load_all_documents()

    splitter = TextSplitter()
    split_docs = splitter.split_documents(documents)

    stats = splitter.get_split_statistics(split_docs)
    logger.info("Split Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
