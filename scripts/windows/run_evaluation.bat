@echo off
cd /d "%~dp0\..\.."
call .venv\Scripts\activate.bat
python -m backend.evaluation.run_evaluation %*
echo Report: reports\evaluation_report.md
