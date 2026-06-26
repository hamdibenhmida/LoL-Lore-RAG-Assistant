"""
Setup validation script for RAG Chatbot.

Checks that all requirements are met and the system is properly configured.
Run with: python setup_validation.py
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def print_check(passed, message):
    """Print a check result."""
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {message}")
    return passed

def validate_python_version():
    """Check Python version."""
    print_header("Python Version Check")
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print_check(passed, f"Python 3.8+ required")
    
    return passed

def validate_project_structure():
    """Check project directory structure."""
    print_header("Project Structure")
    
    required_dirs = [
        "data",
        "data/pdfs",
        "data/txt",
        "ingestion",
        "vector_store",
        "retrieval",
        "generation",
        "interface",
        "utils"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).is_dir()
        all_exist = all_exist and print_check(exists, f"Directory: {dir_path}")
    
    return all_exist

def validate_core_files():
    """Check core project files."""
    print_header("Core Files")
    
    required_files = [
        "app.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "utils/config.py",
        "ingestion/document_loader.py",
        "ingestion/text_splitter.py",
        "ingestion/embeddings.py",
        "vector_store/chroma_db.py",
        "retrieval/retriever.py",
        "generation/llm.py",
        "generation/prompt_template.py",
        "generation/rag_chain.py",
        "server.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).is_file()
        all_exist = all_exist and print_check(exists, f"File: {file_path}")
    
    return all_exist

def validate_dependencies():
    """Check if key dependencies are installed."""
    print_header("Dependencies Check")
    
    dependencies = {
        "langchain": "LangChain",
        "langchain_groq": "LangChain Groq",
        "langchain_chroma": "LangChain ChromaDB",
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "langchain_community": "LangChain Community"
    }
    
    all_installed = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_check(True, f"{name} installed")
        except ImportError:
            print_check(False, f"{name} NOT installed - run: pip install -r requirements.txt")
            all_installed = False
    
    return all_installed

def validate_environment():
    """Check environment configuration."""
    print_header("Environment Configuration")
    
    # Check .env file
    env_exists = Path(".env").is_file()
    print_check(env_exists, ".env file exists")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Groq API key
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    has_key = groq_key and groq_key != "your_groq_api_key_here"
    print_check(has_key, "GROQ_API_KEY configured")
    
    if not has_key:
        print("   ⚠️  Need to set GROQ_API_KEY in .env file")
    
    # Check other config values
    print(f"   - LLM Model: {os.getenv('LLM_MODEL_NAME', 'llama-3.3-70b-versatile')}")
    print(f"   - Embedding Model: {os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')}")
    print(f"   - Chunk Size: {os.getenv('CHUNK_SIZE', '1000')}")
    print(f"   - Top-K: {os.getenv('RETRIEVER_TOP_K', '3')}")
    
    return env_exists and has_key

def validate_documents():
    """Check for sample documents."""
    print_header("Sample Documents")
    
    pdf_files = list(Path("data/pdfs").glob("*.pdf"))
    txt_files = list(Path("data/txt").glob("*.txt"))
    
    has_pdfs = len(pdf_files) > 0
    has_txts = len(txt_files) > 0
    
    print_check(has_pdfs or has_txts, f"Sample documents found")
    print(f"   - PDFs: {len(pdf_files)}")
    print(f"   - TXT files: {len(txt_files)}")
    
    if not (has_pdfs or has_txts):
        print("   ⚠️  Tip: Add PDF or TXT files to data/pdfs/ or data/txt/")
    
    return True  # Not critical for validation

def validate_config():
    """Validate configuration using Config class."""
    print_header("Configuration Validation")
    
    try:
        from utils.config import Config
        
        is_valid = Config.validate()
        print_check(is_valid, "Configuration is valid")
        
        # Log config
        Config.log_config()
        
        return is_valid
    except Exception as e:
        print_check(False, f"Configuration error: {str(e)}")
        return False

def main():
    """Run all validation checks."""
    print("\n" + "🔍 RAG Chatbot Setup Validation".center(50))
    print("This script verifies your installation is complete.".center(50))
    
    results = {
        "Python Version": validate_python_version(),
        "Project Structure": validate_project_structure(),
        "Core Files": validate_core_files(),
        "Dependencies": validate_dependencies(),
        "Environment": validate_environment(),
        "Configuration": validate_config(),
        "Sample Documents": validate_documents()
    }
    
    # Summary
    print_header("Summary")
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"Checks Passed: {passed_checks}/{total_checks}")
    print()
    
    for check_name, passed in results.items():
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {check_name}")
    
    print()
    
    if passed_checks == total_checks:
        print("✨ " + "="*45 + " ✨")
        print("✨  All checks passed! Ready to use!      ✨")
        print("✨ " + "="*45 + " ✨")
        print("\nNext step: uvicorn server:app --reload --port 8000")
        return 0
    else:
        print("⚠️  " + "="*46 + " ⚠️")
        print("⚠️  Some checks failed. Please review above.   ⚠️")
        print("⚠️  " + "="*46 + " ⚠️")
        print("\nRefer to QUICKSTART.md for troubleshooting.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
