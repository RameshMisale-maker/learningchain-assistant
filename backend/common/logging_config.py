"""Logging setup: console + rotating file handler under logs/, text or JSON (LOG_FORMAT=json), with secret redaction.

Every record passes through RedactingFilter, which masks API keys (Anthropic/OpenAI), bearer tokens, the
assistant's own API tokens, HMAC signatures and e-mail addresses - so credentials and personal data never reach log
files or log shippers, even if a library or an exception message includes them.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from config.settings import get_settings

_CONFIGURED = False

_REDACTIONS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
    (re.compile(r"sap_[A-Za-z0-9_\-]{16,}"), "sap_***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1***"),
    (re.compile(r"(?i)(x-api-key|authorization|signature)([\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"), r"\1\2***"),
    (re.compile(r"sha256=[0-9a-f]{16,}"), "sha256=***"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<email>"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        clean = redact(message)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if clean != message or record.args:
            record.msg, record.args = clean, None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_text or record.exc_info:
            data["exception"] = record.exc_text or self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    import os

    use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
    fmt: logging.Formatter = JsonFormatter() if use_json else \
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    redactor = RedactingFilter()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redactor)
    root.addHandler(console)
    file_handler = RotatingFileHandler(settings.logs_dir / "sapphirus_assistant.log", maxBytes=5_000_000,
                                       backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)
    for noisy in ("httpx", "chromadb", "urllib3", "sentence_transformers", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
