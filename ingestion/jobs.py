"""Database-backed ingestion job queue.

Why a DB queue (not Celery/Redis): the project already has a relational DB (SQLite locally, PostgreSQL in production)
and ingestion volume is small (documents, not events). A table-backed queue gives durability, retries, idempotency
and an admin-visible status without another service to run and secure.

Guarantees
  * Idempotent enqueue: a job with the same `dedupe_key` that is still queued/running is reused, not duplicated.
  * Atomic claim: `UPDATE ... WHERE id=? AND status='queued'` - only one worker can claim a job (safe on SQLite and
    PostgreSQL, across threads and processes).
  * Retries: failures are retried with exponential backoff up to `max_attempts`, then marked `dead`.
  * Crash recovery: jobs stuck in `running` longer than the lease are re-queued.
  * Idempotent processing: handlers rely on content hashes, so re-running a job never creates duplicate records.
"""
from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update

from backend.db.models import IngestionJob
from backend.db.session import db_session
from config.settings import get_settings

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
ACTIVE = ("queued", "running")
LEASE_MINUTES = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:   # SQLite returns naive datetimes
        return dt.replace(tzinfo=timezone.utc)
    return dt


def job_to_dict(job: IngestionJob) -> dict[str, Any]:
    return {"id": job.id, "job_type": job.job_type, "kb": job.kb, "payload": job.payload, "status": job.status,
            "attempts": job.attempts, "max_attempts": job.max_attempts, "error": job.error, "result": job.result,
            "requested_by": job.requested_by, "dedupe_key": job.dedupe_key,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "next_run_at": job.next_run_at.isoformat() if job.next_run_at else None}


def enqueue(job_type: str, kb: str = "", payload: dict | None = None, dedupe_key: str = "",
            requested_by: str = "system", max_attempts: int | None = None) -> tuple[dict, bool]:
    """Add a job. Returns (job, created). An identical active job (same dedupe_key) is returned instead."""
    with db_session() as s:
        if dedupe_key:
            existing = s.scalar(select(IngestionJob).where(IngestionJob.dedupe_key == dedupe_key,
                                                           IngestionJob.status.in_(ACTIVE)))
            if existing is not None:
                return job_to_dict(existing), False
        job = IngestionJob(job_type=job_type, kb=kb, payload=payload or {}, dedupe_key=dedupe_key,
                           requested_by=requested_by,
                           max_attempts=max_attempts or get_settings().ingest_max_attempts, next_run_at=_now())
        s.add(job)
        s.flush()
        logger.info("Enqueued ingestion job %s (%s %s)", job.id, job_type, dedupe_key or kb)
        return job_to_dict(job), True


def claim_next(worker_id: str = WORKER_ID) -> dict | None:
    """Atomically claim the oldest due queued job."""
    with db_session() as s:
        candidates = s.scalars(select(IngestionJob.id).where(IngestionJob.status == "queued",
                                                             IngestionJob.next_run_at <= _now())
                               .order_by(IngestionJob.id).limit(5)).all()
        for job_id in candidates:
            claimed = s.execute(update(IngestionJob).where(IngestionJob.id == job_id, IngestionJob.status == "queued")
                                .values(status="running", locked_by=worker_id, locked_at=_now(),
                                        attempts=IngestionJob.attempts + 1, updated_at=_now()))
            if claimed.rowcount == 1:
                s.flush()
                return job_to_dict(s.get(IngestionJob, job_id))
    return None


def complete(job_id: int, result: dict) -> None:
    with db_session() as s:
        job = s.get(IngestionJob, job_id)
        job.status, job.result, job.error, job.finished_at, job.locked_by = "succeeded", result, "", _now(), ""


def fail(job_id: int, error: str) -> dict:
    """Record a failure: re-queue with exponential backoff, or mark dead after max_attempts."""
    base = get_settings().ingest_retry_base_seconds
    with db_session() as s:
        job = s.get(IngestionJob, job_id)
        job.error, job.locked_by = error[:4000], ""
        if job.attempts >= job.max_attempts:
            job.status, job.finished_at = "dead", _now()
        else:
            job.status = "queued"
            job.next_run_at = _now() + timedelta(seconds=base * (2 ** (job.attempts - 1)))
        return job_to_dict(job)


def retry(job_id: int, requested_by: str = "admin") -> dict:
    """Manually retry a dead/failed job (attempt counter reset). Processing is idempotent, so no duplicates."""
    with db_session() as s:
        job = s.get(IngestionJob, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in ACTIVE:
            return job_to_dict(job)
        job.status, job.attempts, job.next_run_at, job.error, job.finished_at = "queued", 0, _now(), "", None
        job.requested_by = requested_by or job.requested_by
        return job_to_dict(job)


def requeue_stale(lease_minutes: int = LEASE_MINUTES) -> int:
    """Jobs left `running` by a crashed worker go back to the queue."""
    cutoff = _now() - timedelta(minutes=lease_minutes)
    with db_session() as s:
        stale = [j for j in s.scalars(select(IngestionJob).where(IngestionJob.status == "running")).all()
                 if (_aware(j.locked_at) or cutoff) <= cutoff]
        for job in stale:
            job.status, job.locked_by, job.next_run_at = "queued", "", _now()
        if stale:
            logger.warning("Re-queued %d stale ingestion job(s)", len(stale))
        return len(stale)


def get_job(job_id: int) -> dict | None:
    with db_session() as s:
        job = s.get(IngestionJob, job_id)
        return job_to_dict(job) if job else None


def list_jobs(status: str = "", limit: int = 100) -> list[dict]:
    with db_session() as s:
        q = select(IngestionJob).order_by(IngestionJob.id.desc()).limit(limit)
        if status:
            q = q.where(IngestionJob.status == status)
        return [job_to_dict(j) for j in s.scalars(q).all()]


def counts() -> dict[str, int]:
    from sqlalchemy import func

    with db_session() as s:
        return {status: n for status, n in s.execute(
            select(IngestionJob.status, func.count()).group_by(IngestionJob.status)).all()}
