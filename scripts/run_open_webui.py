"""Start Open WebUI (the chat front end) connected to the Sapphirus assistant API.

    python scripts/run_open_webui.py            # http://localhost:3000

Open WebUI needs Python 3.11/3.12, so it runs through `uvx` (https://docs.astral.sh/uv/), which downloads its own
Python 3.11 in an isolated environment - the project's Python is not touched. The first start downloads Open WebUI
(a few minutes); later starts are fast.

Settings come from .env:
    OPEN_WEBUI_API_KEY   API token for Open WebUI (create: python -m backend.security.manage_users create
                         --username openwebui --roles public)
    OPEN_WEBUI_PORT      default 3000
    API_BASE_URL         the assistant API, default http://localhost:8000
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cfg = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = cfg.get("OPEN_WEBUI_API_KEY", "")
    if not key:
        print("OPEN_WEBUI_API_KEY is not set in .env. Create a token with:\n"
              "  python -m backend.security.manage_users create --username openwebui --roles public")
        return 1
    uvx = shutil.which("uvx")
    if not uvx:
        print("uvx not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return 1
    api = (cfg.get("API_BASE_URL") or "http://localhost:8000").rstrip("/")
    port = str(cfg.get("OPEN_WEBUI_PORT") or 3000)
    data_dir = ROOT / "data" / "open-webui"
    data_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "DATA_DIR": str(data_dir),
        "WEBUI_NAME": "Sapphirus AI Assistant",
        # the only model: the Sapphirus assistant served by our OpenAI-compatible API
        "ENABLE_OPENAI_API": "True",
        "OPENAI_API_BASE_URL": f"{api}/v1",
        "OPENAI_API_KEY": key,
        "ENABLE_OLLAMA_API": "False",
        "DEFAULT_MODELS": "sapphirus-assistant",
        # answers must come only from the knowledge base: no web search, no extra LLM side tasks
        "ENABLE_WEB_SEARCH": "False",
        "ENABLE_TITLE_GENERATION": "False",
        "ENABLE_TAGS_GENERATION": "False",
        "ENABLE_FOLLOW_UP_GENERATION": "False",
        "ENABLE_AUTOCOMPLETE_GENERATION": "False",
        "ENABLE_RETRIEVAL_QUERY_GENERATION": "False",
        "ENABLE_SEARCH_QUERY_GENERATION": "False",
        "ENABLE_IMAGE_GENERATION": "False",
        "ENABLE_CODE_INTERPRETER": "False",
        "ENABLE_EVALUATION_ARENA_MODELS": "False",
        "ENABLE_COMMUNITY_SHARING": "False",
        # send the end user's id / chat id so the API can rate-limit per user and keep one session per chat
        "ENABLE_FORWARD_USER_INFO_HEADERS": "True",
        "ANONYMIZED_TELEMETRY": "False",
        "SCARF_NO_ANALYTICS": "True",
        "DO_NOT_TRACK": "True",
    }
    cmd = [uvx, "--python", "3.11", "open-webui", "serve", "--host", "127.0.0.1", "--port", port]
    print(f"Starting Open WebUI on http://localhost:{port}  (API: {api}/v1)")
    return subprocess.call(cmd, env=env, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
