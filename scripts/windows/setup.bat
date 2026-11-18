@echo off
REM ================================================================
REM  Sapphirus AI Assistant - one-time setup (Windows 10/11)
REM  Run from the project root:  scripts\windows\setup.bat
REM ================================================================
setlocal
cd /d "%~dp0\..\.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
echo [1/5] Creating virtual environment .venv ...
%PY% -m venv .venv || goto :error
call .venv\Scripts\activate.bat
echo [2/5] Installing dependencies ...
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist .env (
  echo [3/5] Creating .env from .env.example - edit it to add your OPENAI_API_KEY and tokens
  copy .env.example .env >nul
) else (
  echo [3/5] .env already exists - keeping it
)
echo [4/5] (sample data generation removed - using real LearningChain and company content)
echo [5/5] Indexing knowledge bases ...
python -m ingestion.cli sync || goto :error
echo.
echo Setup complete. Next: scripts\windows\run_all.bat
exit /b 0
:error
echo Setup failed - see the messages above.
exit /b 1
