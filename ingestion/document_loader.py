"""
Document loader for ingesting PDFs and TXT files.

Supports loading documents from PDF and TXT formats using LangChain loaders.
Handles multiple files and normalizes metadata across different file types.
"""

import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils.config import Config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and manage documents from various formats."""

    def __init__(self, pdf_dir: Path = None, txt_dir: Path = None):
        """
        Initialize document loader.

        Args:
            pdf_dir: Directory containing PDF files. Defaults to Config.PDF_DIR
            txt_dir: Directory containing TXT files. Defaults to Config.TXT_DIR
        """
        self.pdf_dir = pdf_dir or Config.PDF_DIR
        self.txt_dir = txt_dir or Config.TXT_DIR

    def load_pdfs(self) -> List[Document]:
        """
        Load all PDF files from the PDF directory.

        Returns:
            List of LangChain Document objects extracted from PDFs.
        """
        documents = []

        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
            return documents

        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.info(f"No PDF files found in {self.pdf_dir}")
            return documents

        logger.info(f"Found {len(pdf_files)} PDF files to load")

        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading PDF: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                # Add source metadata to each document
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"

                documents.extend(pdf_docs)
                logger.info(
                    f"Successfully loaded {len(pdf_docs)} pages from {pdf_file.name}"
                )

            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
                continue

        logger.info(f"Total documents loaded from PDFs: {len(documents)}")
        return documents

    def load_txt_files(self) -> List[Document]:
        """
        Load all TXT files from the TXT directory.

        Returns:
            List of LangChain Document objects extracted from TXT files.
        """
        documents = []

        if not self.txt_dir.exists():
            logger.warning(f"TXT directory does not exist: {self.txt_dir}")
            return documents

        txt_files = list(self.txt_dir.glob("*.txt"))

        if not txt_files:
            logger.info(f"No TXT files found in {self.txt_dir}")
            return documents

        logger.info(f"Found {len(txt_files)} TXT files to load")

        for txt_file in txt_files:
            try:
                logger.info(f"Loading TXT: {txt_file.name}")
                loader = TextLoader(str(txt_file), encoding="utf-8")
                txt_docs = loader.load()

                # Add source metadata to each document
                for doc in txt_docs:
                    doc.metadata["source"] = txt_file.name
                    doc.metadata["file_type"] = "txt"

                documents.extend(txt_docs)
                logger.info(f"Successfully loaded {txt_file.name}")

            except Exception as e:
                logger.error(f"Error loading TXT {txt_file.name}: {str(e)}")
                continue

        logger.info(f"Total documents loaded from TXT files: {len(documents)}")
        return documents

    def load_all_documents(self) -> List[Document]:
        """
        Load all documents from both PDF and TXT directories.

        Returns:
            Combined list of all loaded documents.
        """
        logger.info("Starting to load all documents...")

        all_documents = []
        all_documents.extend(self.load_pdfs())
        all_documents.extend(self.load_txt_files())

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    @staticmethod
    def get_document_summary(documents: List[Document]) -> dict:
        """
        Generate a summary of loaded documents.

        Args:
            documents: List of loaded documents.

        Returns:
            Dictionary containing document statistics.
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_pages": 0,
                "sources": [],
                "file_types": {}
            }

        sources = set()
        file_types = {}
        total_chars = 0

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            file_type = doc.metadata.get("file_type", "unknown")
            sources.add(source)
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.page_content)

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "sources": sorted(list(sources)),
            "file_types": file_types
        }


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    summary = DocumentLoader.get_document_summary(docs)

    logger.info("Document Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
