@echo off
REM Start the RAG evaluation dashboard on http://localhost:8502 (independent of the chatbot on port 8000).
setlocal
cd /d "%~dp0.."
if not exist "evaluation\.venv\Scripts\python.exe" (
    echo Evaluation environment missing - run evaluation\setup_eval_env.bat first.
    exit /b 1
)
evaluation\.venv\Scripts\python -m streamlit run evaluation\dashboard.py --server.port 8502 --browser.gatherUsageStats false
