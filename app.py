"""
Entry point for the RAG Chatbot.

Run the FastAPI server with:
    uvicorn server:app --reload --port 8000

Or use start_server.bat on Windows.
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "server:app", "--reload", "--port", "8000"],
        check=True,
    )
