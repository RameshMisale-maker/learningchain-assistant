"""Chat (JSON and streaming), history, feedback and suggestion endpoints (website widget, portal, Open WebUI)."""
from __future__ import annotations

import json
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from backend.api.deps import client_ip, current_user, rate_limited
from backend.api.schemas import ChatRequest, ChatResponseOut, FeedbackRequest
from backend.db import repository
from backend.security.auth import AuthenticatedUser
from backend.security.rbac import accessible_kbs, suggested_questions
from backend.services.chat_service import get_chat_service

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/me")
def me(user: AuthenticatedUser = Depends(current_user)) -> dict:
    return user.to_dict() | {"knowledge_bases": accessible_kbs(user)}


@router.get("/suggestions")
def suggestions(user: AuthenticatedUser = Depends(current_user)) -> dict:
    return {"questions": suggested_questions(user)}


# This deployment answers LearningChain questions only - every assistant value maps to the products KB.
ASSISTANT_KBS = {"all": ["products"], "company": ["products"], "products": ["products"]}


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: AuthenticatedUser = Depends(rate_limited)):
    """Server-sent events for the website widget.

    The answer is produced and verified (grounding / citation / URL checks) BEFORE anything is shown, then streamed
    word by word - users never see text that is later withdrawn. Events: meta, delta (text), sources, done, error.
    """
    received = time.perf_counter()
    service = get_chat_service()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def events():
        try:
            resp = await run_in_threadpool(lambda: service.answer(
                body.message, user, body.session_id, True, client_ip(request),
                allowed_kbs=ASSISTANT_KBS[body.assistant], channel="widget", assistant=body.assistant))
        except Exception:
            yield sse("error", {"message": "The assistant is temporarily unavailable. Please try again."})
            return
        yield sse("meta", {"session_id": resp.session_id, "answer_type": resp.answer_type,
                           "knowledge_bases": resp.knowledge_bases})
        for piece in re.findall(r"\S+\s*|\s+", resp.answer):
            yield sse("delta", {"text": piece})
        cites = [{"id": c.id, "title": c.question or c.doc_name, "section": c.section, "url": c.url,
                  "kb": c.kb} for c in resp.citations]
        yield sse("sources", {"citations": cites})
        yield sse("done", {"message_id": resp.message_id, "confidence": resp.confidence,
                           "latency_ms": resp.latency_ms,
                           "first_byte_ms": round((time.perf_counter() - received) * 1000, 1)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/chat", response_model=ChatResponseOut)
async def chat(body: ChatRequest, request: Request, user: AuthenticatedUser = Depends(rate_limited)) -> dict:
    service = get_chat_service()
    allowed = ASSISTANT_KBS[body.assistant]
    response = await run_in_threadpool(lambda: service.answer(
        body.message, user, body.session_id, True, client_ip(request), allowed_kbs=allowed, channel="api",
        assistant=body.assistant))
    data = response.to_dict()
    data.pop("retrieved", None)
    if not user.is_admin:  # keep internal diagnostics for administrators only
        data["route"] = {k: v for k, v in data["route"].items() if k in ("intent", "knowledge_bases", "conversation_intent")}
        data["golden_match"] = {"qid": data["golden_match"]["qid"]} if data.get("golden_match") else None
    return data


@router.get("/chat/sessions/{session_id}/messages")
def history(session_id: str, user: AuthenticatedUser = Depends(current_user)) -> dict:
    return {"session_id": session_id, "messages": repository.get_history(session_id, user.username, limit=100)}


@router.delete("/chat/sessions/{session_id}")
def clear(session_id: str, user: AuthenticatedUser = Depends(current_user)) -> dict:
    if not repository.clear_session(session_id, user.username):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"cleared": True}


@router.post("/feedback")
def feedback(body: FeedbackRequest, request: Request, user: AuthenticatedUser = Depends(current_user)) -> dict:
    owner = repository.message_owner(body.message_id)
    if owner is None or owner != user.username:
        raise HTTPException(status_code=404, detail="Message not found")
    if body.rating == "conversation" and body.stars is None:
        raise HTTPException(status_code=422, detail="stars (1-5) is required for a conversation rating")
    rating = f"stars_{body.stars}" if body.rating == "conversation" else body.rating
    fid = repository.add_feedback(body.message_id, user.username, rating, body.comment)
    repository.write_audit(user.username, "feedback", str(body.message_id), details={"rating": body.rating},
                           ip_address=client_ip(request))
    return {"feedback_id": fid}
