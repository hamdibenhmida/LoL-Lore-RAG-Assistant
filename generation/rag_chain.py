"""
RAG chain implementation.

Orchestrates the complete RAG pipeline:
User Question → Retriever → Context → Prompt → LLM → Response
"""

import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

from ingestion.document_loader import DocumentLoader
from ingestion.text_splitter import TextSplitter
from ingestion.embeddings import get_embeddings_generator
from vector_store.chroma_db import ChromaVectorStore
from retrieval.retriever import DocumentRetriever
from generation.llm_factory import get_llm
from generation.prompt_template import PromptTemplate, ResponseEvaluator
from utils.config import Config

logger = logging.getLogger(__name__)


class RAGChain:
    """Complete RAG pipeline orchestrator."""

    def __init__(
        self,
        use_existing_vectorstore: bool = True,
        chunk_size: int = None,
        chunk_overlap: int = None,
        top_k: int = None,
        provider: str = None,
        llm_model: str = None,
    ):
        """
        Initialize RAG chain with all components.

        Args:
            use_existing_vectorstore: Load existing vector store if available.
            chunk_size: Size for document chunks.
            chunk_overlap: Overlap for chunks.
            top_k: Number of documents to retrieve.
            provider: LLM provider ("groq" or "gemini"). Defaults to Config.
            llm_model: Override the model name within the provider.
        """
        logger.info("Initializing RAG Chain...")

        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.top_k = top_k or Config.RETRIEVER_TOP_K

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embeddings = get_embeddings_generator()
        self.vector_store = ChromaVectorStore()
        self.retriever = DocumentRetriever(
            vector_store=self.vector_store,
            top_k=self.top_k
        )
        self.llm = get_llm(provider=provider, model_name=llm_model)
        self.prompt_template = PromptTemplate()
        self.response_evaluator = ResponseEvaluator()

        # Initialize vector store
        self._initialize_vectorstore(use_existing_vectorstore)

        logger.info("RAG Chain initialized successfully")

    def _initialize_vectorstore(self, use_existing: bool = True) -> None:
        """
        Initialize vector store with documents.

        Auto-indexes documents on startup if the collection is empty.

        Args:
            use_existing: Try to load existing store first.
        """
        logger.info("Initializing vector store...")

        try:
            if use_existing:
                # Try to load existing store
                try:
                    self.vector_store.get_or_create_store()
                    # Check if the store actually has documents
                    stats = self.vector_store.get_collection_stats()
                    doc_count = stats.get("document_count", 0)
                    if doc_count > 0:
                        logger.info(f"Loaded existing vector store with {doc_count} documents")
                        return
                    else:
                        logger.info("Vector store is empty — auto-indexing documents...")
                except Exception as e:
                    logger.info(f"Could not load existing store: {str(e)}")

            # Load documents
            logger.info("Loading documents...")
            documents = self.document_loader.load_all_documents()

            if not documents:
                logger.warning("No documents found in data/ folder")
                return

            # Split documents
            logger.info("Splitting documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)

            # Create/update vector store
            logger.info(f"Indexing {len(split_docs)} chunks into vector store...")
            self.vector_store.add_documents(split_docs)

            stats = self.vector_store.get_collection_stats()
            logger.info(f"Auto-indexing complete. Stats: {stats}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def retrieve_context(self, query: str, k: int = None) -> Tuple[str, List[Document]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query.
            k: Number of documents to retrieve.

        Returns:
            Tuple of (formatted_context, list_of_documents).
        """
        logger.info(f"Retrieving context for query: '{query[:100]}...'")

        context, documents = self.retriever.retrieve_and_format(query, k=k)

        logger.info(f"Retrieved {len(documents)} relevant documents")
        return context, documents

    def generate_answer(
        self,
        query: str,
        context: str,
        k: int = None
    ) -> Dict[str, Any]:
        """
        Generate answer to a question based on retrieved context.

        Args:
            query: User question.
            context: Retrieved context (optional, will retrieve if not provided).
            k: Number of documents to retrieve.

        Returns:
            Dictionary with answer, sources, and metadata.
        """
        logger.info(f"Generating answer for: '{query[:100]}...'")

        # If context not provided, retrieve it
        if not context or not context.strip():
            context, retrieved_docs = self.retrieve_context(query, k=k)
        else:
            retrieved_docs = []

        # Validate inputs
        if not self.prompt_template.validate_question(query):
            logger.warning("Invalid question format")
            return {
                "answer": "Please provide a valid question.",
                "query": query,
                "sources": [],
                "error": "Invalid question"
            }

        if not context or not context.strip():
            logger.warning("No context available for query")
            return {
                "answer": "I do not have enough information in the documents to answer this question.",
                "query": query,
                "sources": [],
                "error": "No context available"
            }

        # Format prompt
        try:
            prompt = self.prompt_template.format_prompt(context, query)
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return {
                "answer": "Error processing your question.",
                "query": query,
                "sources": [],
                "error": str(e)
            }

        # Generate response
        try:
            logger.info("Calling LLM for response generation...")
            response = self.llm.generate_response(prompt)

            # Clean response
            response = self.prompt_template.clean_response(response)

            # Evaluate response
            insufficient = self.response_evaluator.is_insufficient_answer(response)

            # Extract sources
            sources = []
            if retrieved_docs:
                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in retrieved_docs
                ]

            logger.info("Answer generated successfully")

            return {
                "answer": response,
                "query": query,
                "sources": sources,
                "insufficient_info": insufficient,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Error generating response. Please try again.",
                "query": query,
                "sources": [],
                "error": str(e)
            }

    def chat(
        self,
        query: str,
        k: int = None,
        provider: str = None,
        llm_model: str = None,
    ) -> Dict[str, Any]:
        """
        Main interface for the RAG system.

        Args:
            query: User question.
            k: Number of documents to retrieve.
            provider: Override LLM provider for this request.
            llm_model: Override model name for this request.

        Returns:
            Complete response dictionary.
        """
        logger.info(f"Processing query: '{query[:100]}...'")

        try:
            # Swap LLM on the fly if the caller requested a different provider
            if provider or llm_model:
                self.llm = get_llm(provider=provider, model_name=llm_model)

            # Retrieve context
            context, documents = self.retrieve_context(query, k=k)

            # Generate answer
            result = self.generate_answer(query, context, k=k)

            # Log stats
            logger.info(f"Query processed successfully. Answer length: {len(result['answer'])}")

            # Attach provider info to the result
            model_info = self.llm.get_model_info()
            result["provider"] = model_info.get("provider", "groq")
            result["model"] = model_info.get("model_name", "")

            return result

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": "An error occurred while processing your question.",
                "query": query,
                "sources": [],
                "error": str(e)
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.

        Returns:
            Dictionary with system statistics.
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "retriever": self.retriever.get_retrieval_stats(),
            "llm": self.llm.get_model_info(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k
        }


if __name__ == "__main__":
    # Example usage
    logger.info("=== RAG Chain Example ===")

    try:
        # Initialize RAG chain
        rag_chain = RAGChain()

        # Get system stats
        stats = rag_chain.get_system_stats()
        logger.info("System Stats:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Example query
        query = "What is machine learning?"
        logger.info(f"\nQuery: {query}")

        # Get response
        result = rag_chain.chat(query)
        logger.info(f"Answer: {result['answer'][:200]}...")

        if result['sources']:
            logger.info(f"Sources: {len(result['sources'])} documents")

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
