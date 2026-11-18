"""Administrator endpoints: knowledge-base management, documents, feedback, audit logs, retention."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.api.deps import admin_user
from backend.api.schemas import SyncRequest
from backend.db import repository
from backend.security.auth import AuthenticatedUser
from backend.security.upload_validator import UploadValidationError
from backend.services.admin_service import AdminService
from config.settings import get_settings
from ingestion import automation, jobs, registry
from ingestion.web import UnsafeURLError

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
_admin: AdminService | None = None


def service() -> AdminService:
    global _admin
    if _admin is None:
        _admin = AdminService()
    return _admin


async def _read_limited(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {get_settings().max_upload_mb} MB")
    return content


@router.get("/knowledge-bases")
def knowledge_bases(_: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"knowledge_bases": service().stats()}


def _queued(job: dict, created: bool) -> JSONResponse:
    return JSONResponse(status_code=202, content={"status": "queued" if created else "already_queued", "job": job,
                                                  "status_url": f"/api/v1/admin/jobs/{job['id']}"})


@router.post("/knowledge-bases/{kb}/sync")
async def sync_kb(kb: str, body: SyncRequest, wait: bool = Query(False),
                  user: AuthenticatedUser = Depends(admin_user)):
    try:
        if not wait:
            service().pipeline.kb_cfg(kb)
            return _queued(*automation.enqueue_sync(user.username, kb, body.full_reindex))
        results = await run_in_threadpool(service().sync, user, kb, body.full_reindex)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"kb": kb, "results": results}


@router.get("/documents")
def documents(kb: str | None = None, include_deleted: bool = False, _: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"documents": registry.list_documents(kb, include_deleted)}


@router.post("/documents")
async def upload(kb: str = Form(...), file: UploadFile = File(...), roles: str = Form(""), category: str = Form(""),
                 is_golden: bool = Form(False), wait: bool = Form(False), user: AuthenticatedUser = Depends(admin_user)):
    """Upload a document. Default: validated, stored and queued (HTTP 202 + job id) - indexing runs in the
    background. `wait=true` processes synchronously (backward compatible)."""
    content = await _read_limited(file)
    try:
        if not wait:
            from backend.security.auth import normalize_roles
            role_list = sorted(normalize_roles(roles)) if roles else None
            job, created = await run_in_threadpool(automation.stage_upload, user.username, kb,
                                                   file.filename or "upload", content, role_list, category or None,
                                                   is_golden)
            return _queued(job, created)
        result = await run_in_threadpool(service().upload_document, user, kb, file.filename or "upload", content,
                                         roles or None, category or None, is_golden)
    except UploadValidationError as exc:
        repository.write_audit(user.username, "document_upload", f"{kb}/{file.filename}", outcome="rejected",
                               details={"reason": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if result.status == "failed":
        raise HTTPException(status_code=422, detail=result.message or "Ingestion failed")
    return result.to_dict()


@router.put("/documents/{doc_id}")
async def replace(doc_id: str, file: UploadFile = File(...), wait: bool = Query(False),
                  user: AuthenticatedUser = Depends(admin_user)):
    """Replace a document with a new version (same doc id; old chunks removed after the new ones are indexed)."""
    content = await _read_limited(file)
    try:
        if not wait:
            doc = registry.get_document(doc_id)
            if not doc or doc["status"] != "active":
                raise KeyError(doc_id)
            return _queued(*await run_in_threadpool(automation.stage_upload, user.username, doc["kb"],
                                                    file.filename or "upload", content, None, None, False, doc_id))
        result = await run_in_threadpool(service().replace_document, user, doc_id, file.filename or "upload", content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@router.delete("/documents/{doc_id}")
async def delete(doc_id: str, wait: bool = Query(False), user: AuthenticatedUser = Depends(admin_user)):
    try:
        if not wait:
            return _queued(*automation.enqueue_delete(user.username, doc_id))
        result = await run_in_threadpool(service().delete_document, user, doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return result.to_dict()


@router.get("/feedback")
def feedback(limit: int = 200, _: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"summary": repository.feedback_summary(), "items": repository.list_feedback(limit)}


@router.get("/audit-logs")
def audit_logs(limit: int = 200, action: str | None = None, _: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"items": repository.list_audit(limit, action)}


@router.get("/users")
def users(_: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"users": repository.list_users()}


@router.post("/retention/purge")
def purge(user: AuthenticatedUser = Depends(admin_user)) -> dict:
    days = get_settings().chat_retention_days
    deleted = repository.purge_old_history(days)
    repository.write_audit(user.username, "retention_purge", f"{days} days", details={"deleted_messages": deleted})
    return {"retention_days": days, "deleted_messages": deleted}


# ============================================================================ ingestion jobs
@router.get("/jobs")
def list_jobs(status: str = "", limit: int = Query(100, le=500), _: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"counts": jobs.counts(), "worker": automation.worker_status(), "jobs": jobs.list_jobs(status, limit)}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, _: AuthenticatedUser = Depends(admin_user)) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: int, user: AuthenticatedUser = Depends(admin_user)) -> dict:
    try:
        job = jobs.retry(job_id, user.username)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    repository.write_audit(user.username, "ingestion_job_retry", f"job:{job_id}")
    return job


# ============================================================================ monitored websites
class WebSourceIn(BaseModel):
    url: str = Field(..., max_length=1000)
    kb: str = Field(..., max_length=64)
    source_type: str = Field("page", pattern="^(page|sitemap|zendesk|faq)$")
    interval_minutes: int | None = Field(None, ge=5, le=10080)
    enabled: bool = True
    crawl_now: bool = True


class WebSourceUpdate(BaseModel):
    enabled: bool | None = None
    interval_minutes: int | None = Field(None, ge=5, le=10080)


@router.get("/web-sources")
def web_sources(_: AuthenticatedUser = Depends(admin_user)) -> dict:
    return {"sources": automation.list_sources(), "allowed_domains": get_settings().web_allowed_domains}


@router.post("/web-sources")
def add_web_source(body: WebSourceIn, user: AuthenticatedUser = Depends(admin_user)) -> dict:
    try:
        src = automation.add_source(body.url, body.kb, body.source_type, body.interval_minutes, user.username,
                                    body.enabled)
    except UnsafeURLError as exc:
        repository.write_audit(user.username, "web_source_rejected", body.url[:200], outcome="rejected",
                               details={"reason": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    repository.write_audit(user.username, "web_source_added", src["url"],
                           details={"kb": src["kb"], "type": src["source_type"]})
    job = None
    if body.crawl_now and src["enabled"]:
        job = automation.enqueue_crawl(src["id"], user.username, force=True)[0]
    return {"source": src, "job": job}


@router.patch("/web-sources/{source_id}")
def update_web_source(source_id: int, body: WebSourceUpdate, user: AuthenticatedUser = Depends(admin_user)) -> dict:
    try:
        return automation.update_source(source_id, **body.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc


@router.delete("/web-sources/{source_id}")
def delete_web_source(source_id: int, delete_content: bool = True,
                      user: AuthenticatedUser = Depends(admin_user)) -> dict:
    try:
        info = automation.remove_source(source_id, delete_content, user.username)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc
    repository.write_audit(user.username, "web_source_removed", info["url"])
    return {"removed": info}


@router.post("/web-sources/{source_id}/crawl")
def crawl_web_source(source_id: int, force: bool = False, user: AuthenticatedUser = Depends(admin_user)):
    try:
        return _queued(*automation.enqueue_crawl(source_id, user.username, force))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc


# ============================================================================ performance & evaluation dashboard data
@router.get("/metrics/summary")
def metrics_summary(start: str | None = None, end: str | None = None, channel: str = "", assistant: str = "",
                    _: AuthenticatedUser = Depends(admin_user)) -> dict:
    from backend.services import metrics_service
    try:
        return metrics_service.summary(start, end, channel, assistant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {exc}") from exc


@router.get("/metrics/timeseries")
def metrics_timeseries(start: str | None = None, end: str | None = None, bucket: str = Query("hour", pattern="^(hour|day)$"),
                       channel: str = "", assistant: str = "", _: AuthenticatedUser = Depends(admin_user)) -> dict:
    from backend.services import metrics_service
    return {"bucket": bucket, "series": metrics_service.timeseries(start, end, bucket, channel, assistant)}


@router.get("/evaluations")
def evaluations(run_type: str = "", limit: int = Query(50, le=200), _: AuthenticatedUser = Depends(admin_user)) -> dict:
    """Evaluation history (for trends / before-after) plus the latest generated reports."""
    import json as _json

    from backend.services import metrics_service
    reports = {}
    base = get_settings().base_dir / "reports"
    for name, rel in {"hallucination": "hallucination/judge_validation.json",
                      "hallucination_system": "hallucination/system_hallucination.json",
                      "bench_chunking": "benchmarks/chunking.json", "bench_embedding": "benchmarks/embedding.json",
                      "bench_reranker": "benchmarks/reranker.json", "bench_retrieval": "benchmarks/retrieval.json",
                      "load_test": "load_test/load_test.json", "latest_rag": "evaluation_summary.json"}.items():
        path = base / rel
        if path.exists():
            try:
                reports[name] = _json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                reports[name] = {"error": "unreadable"}
    return {"runs": metrics_service.evaluation_runs(run_type, limit), "reports": reports}
