# Deliverable 10 — Windows Installation and Execution Guide

Tested layout for Windows 10/11 (64-bit). Commands are for **PowerShell** unless noted.

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11, 3.12 or 3.13 (64-bit) | <https://www.python.org/downloads/> — tick **"Add python.exe to PATH"** |
| Git (optional) | any | to clone/version the project |
| Microsoft C++ Build Tools | only if a wheel fails to install | usually not needed — all core packages ship Windows wheels |
| Tesseract OCR (optional) | 5.x | only for scanned PDFs: install from the UB-Mannheim build, then set `OCR_ENABLED=true` and `TESSERACT_CMD` |
| PostgreSQL (optional) | 14+ | SQLite is used automatically when `DATABASE_URL` is empty |

## 2. Quick setup (one command)

```bat
cd C:\Projects\sapphirus_chatbot
scripts\windows\setup.bat
```

This creates `.venv`, installs `requirements.txt`, copies `.env.example` → `.env`, generates the sample datasets and indexes all three knowledge bases.

## 3. Manual setup (step by step)

```powershell
cd C:\Projects\sapphirus_chatbot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
# optional: on-prem embeddings + cross-encoder reranker, OCR, PostgreSQL driver, RAGAS
pip install -r requirements-optional.txt
copy .env.example .env
notepad .env
```

## 4. Configure `.env`

Minimum for the OpenAI setup:

```
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
EMBEDDING_BACKEND=openai
RERANKER_BACKEND=cross_encoder      # needs requirements-optional.txt; otherwise use: lexical
TOKEN_HASH_SECRET=<long random string>
BOOTSTRAP_ADMIN_TOKEN=<random token>
BOOTSTRAP_EMPLOYEE_TOKEN=<random token>
BOOTSTRAP_HR_TOKEN=<random token>
```

Generate random values with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

**No API key yet?** Use `LLM_PROVIDER=extractive`, `EMBEDDING_BACKEND=local_hash`, `RERANKER_BACKEND=lexical` — everything runs offline (answers are extracts from the sources).

## 5. Build the knowledge bases

```powershell
python -m scripts.generate_datasets      # sample data (skip when using real documents)
python -m ingestion.cli sync              # index all KBs
python -m ingestion.cli list-kbs          # check document/chunk counts
```

If you change `EMBEDDING_BACKEND` later: `python -m ingestion.cli reindex`.

## 6. Run

```bat
scripts\windows\run_all.bat
```

or in two terminals:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000     # API  -> http://localhost:8000/docs
python scriptsun_open_webui.py                          # UI   -> http://localhost:3000 (Open WebUI)
```

Or just double-click `Start_Sapphirus_Chatbot.bat` (starts both). Open WebUI runs via `uvx` on its own Python 3.11; the first start downloads it. Set `OPEN_WEBUI_API_KEY` in `.env` to an API token (`python -m backend.security.manage_users create --username openwebui --roles public`). The first account you create in Open WebUI becomes its admin.

Sign in by pasting a token (from `.env` or created with
`python -m backend.security.manage_users create --username priya --roles employee,hr`) into the sidebar.
No token = guest (`public` role: company + products only).

## 7. Test and evaluate

```bat
scripts\windows\run_tests.bat
scripts\windows\run_evaluation.bat
```

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Knowledge base 'x' was indexed with 'A' but the current embedder is 'B'` | `python -m ingestion.cli reindex --kb x` |
| Open WebUI shows no model / connection error | start the API on port 8000 and check `OPEN_WEBUI_API_KEY` in `.env` is a valid token |
| Answers say "Extracted from sources" | no LLM key — set `OPENAI_API_KEY` and restart the API |
| Cross-encoder warning in logs | install `requirements-optional.txt` or set `RERANKER_BACKEND=lexical` |
| Scanned PDF pages skipped | install Tesseract, `OCR_ENABLED=true`, `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`, then `sync` |
| Port already in use | change `--port` and `API_BASE_URL` |
| Corporate proxy blocks model downloads | set `HTTPS_PROXY`, or pre-download models and set `EMBEDDING_MODEL` to the local folder path |
