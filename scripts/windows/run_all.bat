@echo off
REM Starts the assistant API and the Open WebUI front end in two windows, then opens the browser.
REM First start of Open WebUI downloads it (a few minutes); later starts are quick.
cd /d "%~dp0\..\.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
start "Sapphirus API" cmd /k python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
timeout /t 8 /nobreak >nul
start "Open WebUI" cmd /k python scripts\run_open_webui.py
echo Waiting for Open WebUI to start on http://127.0.0.1:3000 ...
:wait
timeout /t 5 /nobreak >nul
curl -s -o nul http://127.0.0.1:3000 || goto wait
start http://127.0.0.1:3000
