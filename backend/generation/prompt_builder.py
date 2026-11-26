"""Builds LLM prompts from templates in prompts/ and the selected context chunks.

Each retrieved chunk is wrapped in a <source> block with an id (S1, S2, ...) and its
citation metadata. Text inside sources is neutralised so it cannot close the wrapper
or impersonate system/assistant turns (prompt-injection hardening).
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from backend.retrieval.hybrid import Candidate
from config.settings import get_settings
from ingestion.text_cleaner import neutralize_for_prompt


@dataclass
class SourceRef:
    sid: str
    candidate: Candidate
    is_golden: bool = False
    metadata: dict = field(default_factory=dict)
    context_text: str = ""     # parent section when parent-child chunking is used; else the chunk itself

    @property
    def text(self) -> str:
        return self.context_text or self.candidate.chunk.text

    @property
    def matched_text(self) -> str:
        """The (child) chunk that actually matched the question - used for citation snippets."""
        return self.candidate.chunk.text


@lru_cache(maxsize=None)
def _read_prompt(name: str) -> str:
    return (get_settings().prompts_dir / name).read_text(encoding="utf-8")


_PLACEHOLDER = re.compile(r"\[\[([a-z_]+)\]\]")


def load_prompt(name: str) -> str:
    """Prompt template with the current product's values filled into its [[placeholders]] (config/products/*.yaml).
    The remaining {fields} are filled by the caller with str.format()."""
    from backend.products import current_product

    values = current_product().prompt_vars()
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), _read_prompt(name))


load_prompt.cache_clear = _read_prompt.cache_clear  # type: ignore[attr-defined]  (tests reset prompt caches)


def describe_source(meta: dict) -> str:
    parts = [f'document="{meta.get("doc_name", "")}"', f'kb="{meta.get("kb", "")}"']
    page = meta.get("page")
    if isinstance(page, (int, float)) and page and page > 0:
        parts.append(f"page={int(page)}")
    if meta.get("section"):
        parts.append(f'section="{str(meta["section"])[:80]}"')
    if meta.get("source_url"):
        parts.append('url="' + str(meta["source_url"]).replace('"', "") + '"')
    if meta.get("version"):
        parts.append(f'version="{meta["version"]}"')
    if meta.get("uploaded_at"):
        parts.append(f'updated="{str(meta["uploaded_at"])[:10]}"')
    if meta.get("doc_type") == "golden_qa":
        parts.append(f'type="APPROVED ANSWER {meta.get("qid", "")}"')
    if meta.get("is_sample"):
        parts.append('status="SAMPLE data"')
    return " ".join(parts)


def select_context(candidates: list[Candidate], max_chars: int) -> list[SourceRef]:
    refs: list[SourceRef] = []
    used = 0
    seen: set[str] = set()
    for cand in candidates:
        # parent-child chunking: a matched child is widened to its parent section, and several children of the
        # same parent become one source (the best-ranked child represents it)
        key = str(cand.meta.get("parent_id") or cand.id)
        if key in seen:
            continue
        context = str(cand.meta.get("parent_text") or "")
        text_len = len(context or cand.chunk.text)
        if refs and used + text_len > max_chars:
            if not context:
                continue
            context, text_len = "", len(cand.chunk.text)   # parent does not fit: fall back to the child alone
            if used + text_len > max_chars:
                continue
        seen.add(key)
        used += text_len
        refs.append(SourceRef(sid=f"S{len(refs) + 1}", candidate=cand,
                              is_golden=cand.meta.get("doc_type") == "golden_qa", metadata=cand.meta,
                              context_text=context))
    return refs


def format_sources(refs: list[SourceRef]) -> str:
    blocks = []
    for ref in refs:
        body = neutralize_for_prompt(ref.text)
        if ref.is_golden:
            body = f"APPROVED QUESTION: {ref.metadata.get('question', '')}\nAPPROVED ANSWER: {ref.metadata.get('answer', '')}"
        blocks.append(f"<source id=\"{ref.sid}\" {describe_source(ref.metadata)}>\n{body}\n</source>")
    return "\n\n".join(blocks)


def format_history(history: list[dict], max_turns: int = 4, max_chars: int = 400, last_answer_chars: int | None = None) -> str:
    """Recent turns for a prompt. `last_answer_chars` keeps more of the latest assistant answer, so follow-ups such
    as "explain the second point" can be resolved against the full list that answer contained."""
    turns = history[-max_turns * 2:]
    if not turns:
        return "(none)"
    last_asst = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=-1)
    return "\n".join(
        f"{t['role'].upper()}: "
        f"{neutralize_for_prompt(t['content'][:(last_answer_chars or max_chars) if i == last_asst else max_chars])}"
        for i, t in enumerate(turns))


OLDER_MESSAGE_CHARS = 160      # an older message is kept as a one-line gist
RECENT_MESSAGE_CHARS = 400
LAST_ANSWER_CHARS = 1000


def format_chat_history(history: list[dict], last_answer_chars: int | None = None) -> str:
    """The whole chat for a prompt (conversation memory): the last CHAT_HISTORY_RECENT_TURNS exchanges in detail
    (the latest answer up to `last_answer_chars`), older messages as a one-line gist, everything within
    CHAT_HISTORY_MAX_CHARS - in a very long chat the oldest lines are dropped first and the omission is noted."""
    s = get_settings()
    turns = [t for t in (history or [])[-max(2, s.chat_history_messages):]
             if t.get("role") in ("user", "assistant") and str(t.get("content") or "").strip()]
    if not turns:
        return "(none)"
    recent_from = len(turns) - max(1, s.chat_history_recent_turns) * 2
    last_asst = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=-1)
    lines = []
    for i, t in enumerate(turns):
        text = str(t["content"])
        if i == last_asst:
            limit = last_answer_chars or LAST_ANSWER_CHARS
        elif i >= recent_from:
            limit = RECENT_MESSAGE_CHARS
        else:
            limit, text = OLDER_MESSAGE_CHARS, " ".join(text.split())
        if len(text) > limit:
            text = text[:limit].rstrip() + " ..."
        lines.append(f"{t['role'].upper()}: {neutralize_for_prompt(text)}")
    dropped = 0
    while len(lines) > 2 and sum(len(ln) + 1 for ln in lines) > max(500, s.chat_history_max_chars):
        lines.pop(0)
        dropped += 1
    if dropped:
        lines.insert(0, f"(earlier in this chat: {dropped} more messages, omitted for length)")
    return "\n".join(lines)


def build_answer_prompt(question: str, refs: list[SourceRef], history: list[dict], comparison: bool = False,
                        entities: list[str] | None = None, original: str | None = None,
                        language: str = "English") -> tuple[str, str]:
    """`question` is the normalized question used for retrieval; `original` is the user's own wording, passed only so
    the model can match the user's tone (replies are always in English)."""
    task = ""
    if comparison and entities:
        task = (f"The user wants a comparison of: {', '.join(entities)}. Present a Markdown table of the attributes "
                "found in the sources, then a 1-2 sentence summary.")
    elif any(r.is_golden for r in refs):
        task = "An APPROVED ANSWER is available; prefer its wording and meaning."
    user = load_prompt("answer_prompt.txt").format(history=format_chat_history(history), sources=format_sources(refs),
                                                    question=neutralize_for_prompt(question), task_instruction=task,
                                                    original=neutralize_for_prompt(original or question), language=language)
    system = load_prompt("system_prompt.txt").replace("{insufficient_text}", get_settings().insufficient_info_text)
    return system, user
