@echo off
REM One-time setup of the isolated evaluation environment (evaluation\.venv).
REM The chatbot's own Python environment is not changed.
setlocal
cd /d "%~dp0.."
if not exist "evaluation\.venv\Scripts\python.exe" (
    python -m venv --system-site-packages evaluation\.venv || goto :error
)
evaluation\.venv\Scripts\python -m pip install --no-deps ragas==0.4.3 || goto :error
evaluation\.venv\Scripts\python -m pip install "langchain<1" "langchain-core<1" "langchain-community<0.4" "langchain-openai<1" "datasets>=4.0.0" instructor tiktoken nest-asyncio appdirs "diskcache>=5.6.3" typer rich networkx "pillow>=10.4.0" || goto :error
evaluation\.venv\Scripts\python -c "import ragas; print('Evaluation environment ready - ragas', ragas.__version__)"
exit /b 0
:error
echo Setup failed.
exit /b 1
