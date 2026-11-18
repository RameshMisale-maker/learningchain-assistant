"""OpenAI-compatible endpoints so Open WebUI (or any OpenAI client) can use the assistant as a chat "model".

    GET  /v1/models
    POST /v1/chat/completions      (supports "stream": true via server-sent events)

Open WebUI: Admin Settings -> Connections -> OpenAI API -> URL http://localhost:8000/v1, key = an API token
created with `python -m backend.security.manage_users create ...`. Every answer still goes through the full
pipeline (routing, approved answers, retrieval, grounding, access control, audit log).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from backend.api.deps import client_ip, current_user
from backend.security.auth import AuthenticatedUser
from backend.security.rate_limit import RateLimiter
from backend.services.chat_service import ChatResponse, get_chat_service
from config.settings import get_settings

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

MODEL_ID = "sapphirus-assistant"
MODEL_NAME = "Sapphirus LearningChain Assistant"
# Separate assistant experiences. allowed_kbs is enforced server-side, so a scoped assistant can never answer
# from (or leak) another knowledge base - whatever the client sends.
ASSISTANTS: dict[str, dict] = {
    # LearningChain-only deployment (company information is served by a separate chatbot)
    "sapphirus-assistant": {"name": "LearningChain Assistant", "assistant": "products", "allowed_kbs": ["products"]},
}
SOURCES_MARKER = "\n\n---\n**Sources**"
HISTORY_TURNS = 8

_limiter: RateLimiter | None = None
# Open WebUI chat id -> our session id, so one Open WebUI conversation maps to one stored session (bounded LRU).
_sessions: OrderedDict[tuple[str, str], str] = OrderedDict()
_MAX_SESSIONS = 5000


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = None


class ChatCompletionIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = MODEL_ID
    messages: list[MessageIn]
    stream: bool = False


def _text(content: Any) -> str:
    """OpenAI message content is a string or a list of parts ({"type": "text", "text": ...})."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _rate_limit(user: AuthenticatedUser, request: Request) -> None:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().rate_limit_per_minute)
    # Open WebUI calls with one shared key; it forwards the end user's id when ENABLE_FORWARD_USER_INFO_HEADERS=true
    end_user = request.headers.get("x-openwebui-user-id") or client_ip(request)
    if not _limiter.allow(f"{user.username}:{end_user}"):
        raise HTTPException(status_code=429, detail="Too many requests - please wait a moment")


def _task_reply(prompt: str) -> str | None:
    """Open WebUI background tasks (chat title, tags, follow-ups, search queries) are sent as normal completions.
    They are not questions for the knowledge base, so answer them directly with the JSON Open WebUI expects."""
    if not prompt.lstrip().startswith("### Task:"):
        return None
    low = prompt.lower()
    if "title" in low:
        first = re.search(r"USER:\s*(.+)", prompt)
        title = (first.group(1).strip() if first else "LearningChain help")[:60]
        return json.dumps({"title": title})
    if "tags" in low:
        return json.dumps({"tags": ["LearningChain"]})
    if "follow" in low:
        return json.dumps({"follow_ups": []})
    if "queries" in low or "search" in low:
        return json.dumps({"queries": []})
    return ""


def _format(resp: ChatResponse) -> str:
    """Answer text plus a compact, de-duplicated sources list with clickable links where a source URL exists."""
    text = resp.answer
    seen, lines = set(), []
    for c in resp.citations:
        label = c.doc_name + (f" - {c.section}" if c.section and c.section != c.doc_name else "")
        if c.qid:
            label = f"Approved answer: {c.question or c.doc_name}" if getattr(c, "question", "") else                 f"Approved answer {c.qid} ({c.doc_name})"
        key = c.url or label
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{label}]({c.url})" if c.url else f"- {label}")
    if lines:
        text += SOURCES_MARKER + "\n" + "\n".join(lines[:5])
    return text


def _history(messages: list[MessageIn]) -> list[dict]:
    turns = []
    for m in messages[:-1]:
        if m.role not in ("user", "assistant"):
            continue
        content = _text(m.content).split(SOURCES_MARKER)[0].strip()
        if content and not content.startswith("### Task:"):
            turns.append({"role": m.role, "content": content})
    return turns[-HISTORY_TURNS:]


def _completion(content: str, created: int, cid: str, model: str = MODEL_ID) -> dict:
    return {"id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _stream(content: str, created: int, cid: str, model: str = MODEL_ID):
    def chunk(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                      "model": model, "choices": [{"index": 0, "delta": delta,
                                                                      "finish_reason": finish}]}) + "\n\n"
    yield chunk({"role": "assistant", "content": ""})
    for piece in re.findall(r"\S+\s*|\s+", content):  # word by word, keeping whitespace / newlines
        yield chunk({"content": piece})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"


@router.get("/models")
def list_models(user: AuthenticatedUser = Depends(current_user)) -> dict:
    return {"object": "list", "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "sapphirus",
                                        "name": cfg["name"]} for mid, cfg in ASSISTANTS.items()]}


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionIn, request: Request,
                           user: AuthenticatedUser = Depends(current_user)):
    _rate_limit(user, request)
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    assistant = ASSISTANTS.get(body.model)
    if assistant is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{body.model}'. Available: {', '.join(ASSISTANTS)}")
    question = _text(body.messages[-1].content).strip()
    created, cid = int(time.time()), f"chatcmpl-{uuid.uuid4().hex}"
    model_id = body.model

    content = _task_reply(question)
    if content is None:
        chat_id = request.headers.get("x-openwebui-chat-id", "")
        key = (user.username, chat_id)
        session_id = _sessions.get(key) if chat_id else None
        resp = await run_in_threadpool(lambda: get_chat_service().answer(
            question, user, session_id, True, client_ip(request), _history(body.messages),
            allowed_kbs=assistant["allowed_kbs"], channel="openai", assistant=assistant["assistant"]))
        if chat_id:
            _sessions[key] = resp.session_id
            _sessions.move_to_end(key)
            while len(_sessions) > _MAX_SESSIONS:
                _sessions.popitem(last=False)
        content = _format(resp)

    if body.stream:
        return StreamingResponse(_stream(content, created, cid, model_id), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})
    return _completion(content, created, cid, model_id)
