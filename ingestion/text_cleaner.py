"""Text cleaning and normalisation applied to every extracted section before chunking.

Also contains the prompt-injection detector used at ingestion time (to flag suspicious
document content) and at query time (to flag suspicious user input).
"""
from __future__ import annotations

import re
from collections import Counter

from backend.common.text_utils import normalize_text
from ingestion.document_loader import LoadedDocument

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTISPACE_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_BULLET_RE = re.compile(r"^\s*[•●▪‣⁃◦]\s*", re.MULTILINE)
_SOFT_WRAP_RE = re.compile(r"(?<=[a-z0-9,])\n(?=[a-z(])")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)\s+(instructions|rules|prompts?)",
    r"forget\s+(all\s+|your\s+)?(previous\s+)?(instructions|rules)",
    r"(reveal|show|print|display|repeat)\s+(me\s+)?(your|the)\s+(system|hidden|initial)\s+(prompt|instructions)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"\b(developer|dan|jailbreak|god)\s+mode\b",
    r"act\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(unrestricted|unfiltered|evil)",
    r"override\s+(the\s+)?(system|safety|security)\s+(prompt|rules|instructions)",
    r"</?\s*(system|assistant|document|context)\s*>",
    r"bypass\s+(the\s+)?(access\s+control|rbac|permissions|restrictions)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _SOFT_WRAP_RE.sub(" ", text)
    text = _BULLET_RE.sub("- ", text)
    text = "\n".join(_MULTISPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(line for line in text.split("\n") if not _PAGE_NUM_RE.match(line) or not line.strip())
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_repeated_headers(doc: LoadedDocument, min_pages: int = 3, ratio: float = 0.6) -> None:
    """Drop short lines that repeat on most PDF pages (running headers/footers)."""
    pages = [s for s in doc.sections if s.page is not None and s.kind == "text"]
    if len({s.page for s in pages}) < min_pages:
        return
    counter: Counter[str] = Counter()
    for sec in pages:
        counter.update({ln.strip() for ln in sec.text.split("\n") if 0 < len(ln.strip()) <= 80})
    threshold = max(2, int(len(pages) * ratio))
    repeated = {line for line, count in counter.items() if count >= threshold}
    if not repeated:
        return
    for sec in pages:
        sec.text = "\n".join(ln for ln in sec.text.split("\n") if ln.strip() not in repeated)


def clean_document(doc: LoadedDocument) -> LoadedDocument:
    remove_repeated_headers(doc)
    for section in doc.sections:
        if section.kind == "table":
            section.text = normalize_text(_CONTROL_RE.sub("", section.text)).strip()
        else:
            section.text = clean_text(section.text)
    doc.sections = [s for s in doc.sections if s.text.strip()]
    return doc


def detect_injection(text: str) -> list[str]:
    """Return the suspicious phrases found in `text` (empty list when clean)."""
    return [m.group(0) for m in _INJECTION_RE.finditer(text or "")]


def neutralize_for_prompt(text: str) -> str:
    """Escape tag-like sequences so untrusted text (documents, web pages, chat history, questions) cannot close or
    open the <source>/<history> wrappers used in prompts, or impersonate system/assistant turns."""
    return re.sub(r"<\s*(/?)\s*(system|assistant|document|context|user|sources?|history|question|instructions?)\b",
                  r"&lt;\1\2", text, flags=re.IGNORECASE)
