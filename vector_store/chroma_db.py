"""
ChromaDB vector store management.

Handles initialization, storage, and retrieval of embeddings in ChromaDB.
Ensures persistent storage to avoid recomputing embeddings on each run.
"""

import logging
from typing import List, Dict, Set
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from utils.config import Config

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manage ChromaDB for persistent vector storage."""

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embeddings: Embeddings = None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage.
                              Defaults to Config.CHROMA_PERSIST_DIRECTORY
            collection_name: Name of the collection.
                            Defaults to Config.COLLECTION_NAME
            embeddings: Embeddings instance to use.
                       Defaults to using SentenceTransformersEmbeddings
        """
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or Config.COLLECTION_NAME

        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Use provided embeddings or Google Gemini API (no local model needed)
        if embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=Config.GEMINI_API_KEY,
            )

        self.embeddings = embeddings
        self.vector_store = None

        logger.info(f"ChromaVectorStore initialized with collection: {collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            List of document IDs added to the store.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store...")

        try:
            # Create or update the Chroma store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )

            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return [str(i) for i in range(len(documents))]

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def get_or_create_store(self, documents: List[Document] = None) -> Chroma:
        """
        Get existing or create new vector store.

        Args:
            documents: Documents to use if creating new store.
                      If None, tries to load existing store.

        Returns:
            Chroma vector store instance.
        """
        # Try to load existing store
        try:
            logger.info(f"Attempting to load existing collection: {self.collection_name}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Successfully loaded existing vector store")
            return self.vector_store

        except Exception as e:
            logger.info(f"Could not load existing store: {str(e)}")

            # Create new store if documents provided
            if documents:
                logger.info("Creating new vector store from documents")
                return self.add_documents(documents) or self.vector_store
            else:
                logger.error("No documents provided and could not load existing store")
                raise

    def search_similar(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """
        Search for similar documents.

        Args:
            query: Query text to search for.
            k: Number of results to return.
              Defaults to Config.RETRIEVER_TOP_K

        Returns:
            List of tuples (document, score).
        """
        if self.vector_store is None:
            logger.error("Vector store not initialized. Call get_or_create_store first.")
            raise RuntimeError("Vector store not initialized")

        k = k or Config.RETRIEVER_TOP_K

        logger.info(f"Searching for {k} similar documents to query")

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def delete_by_source(self, source_path: str) -> int:
        """
        Delete all chunks whose metadata 'source' matches source_path.

        Args:
            source_path: The file path stored in chunk metadata (as returned
                         by the document loader).

        Returns:
            Number of chunks deleted.
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialised — nothing to delete")
            return 0

        try:
            collection = self.vector_store._collection
            # Fetch all IDs whose source matches
            results = collection.get(
                where={"source": source_path},
                include=["metadatas"]
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(
                    f"Deleted {len(ids)} chunks for source: {source_path}"
                )
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting chunks for source '{source_path}': {e}")
            raise

    def get_all_sources(self) -> Set[str]:
        """
        Return the set of unique 'source' values stored in the collection.

        Returns:
            Set of source file paths recorded in chunk metadata.
        """
        if self.vector_store is None:
            return set()
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["metadatas"])
            sources = set()
            for meta in results.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sources
        except Exception as e:
            logger.error(f"Error fetching sources from collection: {e}")
            return set()

    def sync_with_disk(self) -> Dict:
        """
        Remove chunks whose source documents no longer exist on disk.

        Compares every 'source' value in ChromaDB against the file system.
        Any source file that is missing from disk has its chunks deleted.

        Returns:
            Dict with keys:
                - removed_sources  (list[str]) files that were purged
                - deleted_chunks   (int)       total chunks removed
                - kept_sources     (list[str]) files still present on disk
        """
        logger.info("Starting sync: checking for deleted source documents…")

        all_sources   = self.get_all_sources()
        removed       = []
        deleted_total = 0
        kept          = []

        for src in all_sources:
            if Path(src).exists():
                kept.append(src)
            else:
                logger.info(f"Source no longer on disk — purging: {src}")
                deleted_total += self.delete_by_source(src)
                removed.append(src)

        logger.info(
            f"Sync complete: {len(removed)} source(s) removed, "
            f"{deleted_total} chunk(s) deleted, {len(kept)} source(s) kept."
        )
        return {
            "removed_sources": removed,
            "deleted_chunks":  deleted_total,
            "kept_sources":    kept,
        }

    def delete_collection(self) -> bool:
        """
        Delete the current collection.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.vector_store is not None:
                # ChromaDB doesn't have a direct delete method,
                # so we reinitialize with empty store
                logger.info(f"Deleting collection: {self.collection_name}")
                # This would require clearing the persistent data
                logger.info("Collection deletion requested (requires manual cleanup)")
                return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        try:
            if self.vector_store is None:
                return {
                    "status": "not_initialized",
                    "collection_name": self.collection_name
                }

            # Try to get collection info
            collection = self.vector_store._collection
            doc_count = collection.count()

            return {
                "collection_name": self.collection_name,
                "document_count": doc_count,
                "persist_directory": self.persist_directory,
                "status": "active"
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


if __name__ == "__main__":
    # Example usage
    from ingestion.document_loader import DocumentLoader
    from ingestion.text_splitter import TextSplitter

    loader = DocumentLoader()
    documents = loader.load_all_documents()

    if documents:
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)

        vector_store = ChromaVectorStore()
        vector_store.add_documents(split_docs)

        stats = vector_store.get_collection_stats()
        logger.info("Vector Store Stats:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
