"""
Configuration management for the RAG system.

Loads all settings from environment variables and provides
centralized access to configuration throughout the application.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration for the RAG system."""

    # ===== LLM Provider Selection =====
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # "auto", "groq", "gemini", or "openrouter"

    # ===== Groq API Configuration =====
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # ===== Google Gemini Configuration =====
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    # ===== OpenRouter Configuration =====
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini")

    # ===== Embedding Configuration =====
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "models/text-embedding-004"
    )

    # ===== Vector Store Configuration =====
    CHROMA_PERSIST_DIRECTORY: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "./chroma_data"
    )
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    # ===== Retrieval Configuration =====
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ===== Logging Configuration =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ===== Paths =====
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PDF_DIR = DATA_DIR / "pdfs"
    TXT_DIR = DATA_DIR / "txt"

    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical configuration values.

        Returns:
            bool: True if all critical configs are present, False otherwise.
        """
        provider = cls.LLM_PROVIDER.lower()

        if provider == "gemini":
            if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your_google_ai_api_key_here":
                logger.error("GEMINI_API_KEY not configured. Please set it in .env file.")
                return False
        elif provider == "openrouter":
            if not cls.OPENROUTER_API_KEY or cls.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
                logger.error("OPENROUTER_API_KEY not configured. Please set it in .env file.")
                return False
        else:
            # Default: validate Groq
            if not cls.GROQ_API_KEY or cls.GROQ_API_KEY == "your_groq_api_key_here":
                logger.error("GROQ_API_KEY not configured. Please set it in .env file.")
                return False

        # Create directories if they don't exist
        if not cls.PDF_DIR.exists():
            cls.PDF_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created PDF directory: {cls.PDF_DIR}")

        if not cls.TXT_DIR.exists():
            cls.TXT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created TXT directory: {cls.TXT_DIR}")

        if not Path(cls.CHROMA_PERSIST_DIRECTORY).exists():
            Path(cls.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created ChromaDB directory: {cls.CHROMA_PERSIST_DIRECTORY}")

        logger.info("Configuration validated successfully")
        return True

    @classmethod
    def log_config(cls) -> None:
        """Log current configuration (excluding sensitive data)."""
        logger.info("=== RAG System Configuration ===")
        logger.info(f"LLM Provider: {cls.LLM_PROVIDER}")
        logger.info(f"LLM Model (Groq): {cls.LLM_MODEL_NAME}")
        logger.info(f"LLM Model (Gemini): {cls.GEMINI_MODEL_NAME}")
        logger.info(f"LLM Model (OpenRouter): {cls.OPENROUTER_MODEL_NAME}")
        logger.info(f"LLM Temperature: {cls.LLM_TEMPERATURE}")
        logger.info(f"Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"Chunk Size: {cls.CHUNK_SIZE}")
        logger.info(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        logger.info(f"Retriever Top-K: {cls.RETRIEVER_TOP_K}")
        logger.info(f"ChromaDB Directory: {cls.CHROMA_PERSIST_DIRECTORY}")
        logger.info("================================")


if __name__ == "__main__":
    Config.validate()
    Config.log_config()
