@echo off
echo =========================================
echo   RAG Assistant - FastAPI Server
echo =========================================
echo.
echo  Server will be available at:
echo  http://localhost:8000
echo.
echo  Press Ctrl+C to stop the server.
echo.
cd /d "%~dp0"
call .venv\Scripts\activate 2>nul || echo (no venv found, using system Python)
uvicorn server:app --reload --port 8000 --host 0.0.0.0
pause
