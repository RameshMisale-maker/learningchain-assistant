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
from backend.products import ProductConfig, ProductConfigError, get_product, resolve_product
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


def assistant_kbs(assistant: str | None, product: ProductConfig | None = None) -> list[str]:
    """Knowledge bases an assistant value may search, from the product config's `scopes` (else all namespaces)."""
    return (product or get_product()).allowed_kbs(assistant)


def request_scope(body: ChatRequest, request: Request) -> tuple[ProductConfig, list[str]]:
    """Product for this request (id / legacy assistant alias / domain / default) and the knowledge bases it may use."""
    try:
        product = resolve_product(product=body.product, assistant=body.assistant, host=request.headers.get("host"))
    except ProductConfigError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return product, product.allowed_kbs(body.assistant)


@router.get("/ui")
def ui_config(request: Request, product: str | None = None, assistant: str | None = None) -> dict:
    """Display texts for the chat page / widget (title, subtitle, suggestions, welcome) from the product config."""
    try:
        cfg = resolve_product(product=product, assistant=assistant, host=request.headers.get("host"))
    except ProductConfigError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    support = {"support_email": cfg.escalation.email, "support_phone": cfg.escalation.phone,
               "support_url": cfg.escalation.contact_url}
    return {"product_id": cfg.product_id, "bot_name": cfg.bot_name, "company": cfg.company, **support, **cfg.ui.model_dump(
        exclude={"api_model_id", "api_model_name"})}


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: AuthenticatedUser = Depends(rate_limited)):
    """Server-sent events for the website widget.

    The answer is produced and verified (grounding / citation / URL checks) BEFORE anything is shown, then streamed
    word by word - users never see text that is later withdrawn. Events: meta, delta (text), sources, done, error.
    """
    received = time.perf_counter()
    service = get_chat_service()
    product, allowed = request_scope(body, request)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def events():
        try:
            resp = await run_in_threadpool(lambda: service.answer(
                body.message, user, body.session_id, True, client_ip(request),
                allowed_kbs=allowed, channel="widget", assistant=body.assistant, product=product))
        except Exception:
            yield sse("error", {"message": "The assistant is temporarily unavailable. Please try again."})
            return
        yield sse("meta", {"session_id": resp.session_id, "answer_type": resp.answer_type,
                           "knowledge_bases": resp.knowledge_bases, "ui_action": resp.ui_action,
                           "conversation_state": resp.conversation_state, "versions": resp.versions,
                           "product_id": product.product_id})
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
    product, allowed = request_scope(body, request)
    response = await run_in_threadpool(lambda: service.answer(
        body.message, user, body.session_id, True, client_ip(request), allowed_kbs=allowed, channel="api",
        assistant=body.assistant, product=product))
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
    """Answer feedback or a 1-5 conversation rating (also served at POST /api/feedback)."""
    conversation = isinstance(body.rating, int) or body.rating == "conversation"
    stars = body.rating if isinstance(body.rating, int) else body.stars
    if conversation and (stars is None or not 1 <= stars <= 5):
        raise HTTPException(status_code=422, detail="a conversation rating needs 1-5 stars")
    message_id = body.message_id
    if message_id is None:
        if not conversation or not body.session_id:
            raise HTTPException(status_code=422,
                                detail="message_id (or session_id for a conversation rating) is required")
        if repository.session_owner(body.session_id) != user.username:
            raise HTTPException(status_code=404, detail="Session not found")
        message_id = repository.last_assistant_message_id(body.session_id)
        if message_id is None:
            raise HTTPException(status_code=404, detail="No answer in this session to rate")
    owner = repository.message_owner(message_id)
    if owner is None or owner != user.username:
        raise HTTPException(status_code=404, detail="Message not found")
    rating = f"stars_{stars}" if conversation else body.rating
    session_id = body.session_id or repository.message_session(message_id) or ""
    product_id = body.product_id or repository.session_product(session_id) or ""
    fid = repository.add_feedback(message_id, user.username, rating, body.comment, session_id=session_id,
                                  product_id=product_id, stars=stars if conversation else None,
                                  details={"versions": body.versions} if body.versions else {})
    from backend.services.feedback_hooks import on_feedback

    on_feedback(message_id, rating, body.comment, user.username, session_id=session_id, product_id=product_id,
                stars=stars if conversation else None, versions=body.versions)
    repository.write_audit(user.username, "feedback", str(message_id), details={"rating": rating},
                           ip_address=client_ip(request))
    return {"feedback_id": fid, "message_id": message_id, "rating": rating}
