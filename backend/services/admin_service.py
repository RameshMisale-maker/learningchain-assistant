"""Administrative operations: upload / replace / delete documents, golden dataset replacement, KB sync."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.db import repository
from backend.retrieval.kb_index import get_index_manager
from backend.security.auth import AuthenticatedUser, normalize_roles
from backend.security.upload_validator import UploadValidationError, validate_upload
from config.settings import get_settings
from ingestion import registry
from ingestion.pipeline import IngestionPipeline, IngestResult

logger = logging.getLogger(__name__)


class AdminService:
    def __init__(self, pipeline: IngestionPipeline | None = None) -> None:
        self.pipeline = pipeline or IngestionPipeline()
        self.settings = get_settings()

    def _archive(self, path: Path, kb: str) -> None:
        if path.exists():
            archive = self.settings.data_dir / "archive" / kb
            archive.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            shutil.copy2(path, archive / f"{stamp}__{path.name}")

    def upload_document(self, user: AuthenticatedUser, kb: str, filename: str, content: bytes,
                        roles: str | list[str] | None = None, category: str | None = None,
                        is_golden: bool = False) -> IngestResult:
        kb_cfg = self.pipeline.kb_cfg(kb)
        valid = validate_upload(filename, content)
        role_list = sorted(normalize_roles(roles)) if roles else None
        if is_golden:
            golden_files = kb_cfg.get("golden_files") or []
            if not golden_files:
                raise UploadValidationError(f"Knowledge base '{kb}' has no golden_files configured in knowledge_bases.yaml")
            target = self.pipeline._abs(golden_files[0])
            if target.suffix.lower() != valid.extension:
                raise UploadValidationError(f"Golden dataset must be a {target.suffix} file (configured path: "
                                            f"{golden_files[0]}). Change golden_files in the config to use another format.")
        else:
            target = self.pipeline.primary_upload_dir(kb) / valid.filename
        self._archive(target, kb)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".uploading")
        tmp.write_bytes(content)
        tmp.replace(target)
        result = self.pipeline.ingest_file(kb, target, uploaded_by=user.username, roles=role_list,
                                           category=category or None, golden=is_golden)
        get_index_manager().invalidate(kb)
        if result.status in ("failed", "duplicate") and not is_golden:
            existing = registry.get_document(result.doc_id) if result.doc_id else None
            if not existing or existing["status"] != "active":
                target.unlink(missing_ok=True)   # do not leave orphan files that a later sync would pick up
        repository.write_audit(user.username, "document_upload", f"{kb}/{valid.filename}", outcome=result.status,
                               details={"doc_id": result.doc_id, "version": result.version, "chunks": result.chunks,
                                        "roles": role_list, "golden": is_golden, "message": result.message})
        return result

    def replace_document(self, user: AuthenticatedUser, doc_id: str, filename: str, content: bytes) -> IngestResult:
        doc = registry.get_document(doc_id)
        if not doc or doc["status"] != "active":
            raise KeyError(doc_id)
        valid = validate_upload(filename, content)
        target = self.pipeline._abs(doc["source_path"])
        if target.suffix.lower() != valid.extension:
            raise UploadValidationError(f"Replacement must have the same file type ({target.suffix})")
        self._archive(target, doc["kb"])
        target.write_bytes(content)
        result = self.pipeline.ingest_file(doc["kb"], target, uploaded_by=user.username, roles=doc["access_roles"],
                                           golden=doc["doc_type"] == "golden_qa", force=True)
        get_index_manager().invalidate(doc["kb"])
        repository.write_audit(user.username, "document_replace", doc["source_path"], outcome=result.status,
                               details={"doc_id": doc_id, "version": result.version})
        return result

    def delete_document(self, user: AuthenticatedUser, doc_id: str) -> IngestResult:
        doc = registry.get_document(doc_id)
        if not doc or doc["status"] != "active":
            raise KeyError(doc_id)
        golden_paths = {self.pipeline._rel(self.pipeline._abs(p)) for p in self.pipeline.kb_cfg(doc["kb"]).get("golden_files", [])}
        extra_paths = {self.pipeline._rel(self.pipeline._abs(p)) for p in self.pipeline.kb_cfg(doc["kb"]).get("extra_files", [])}
        archive = doc["source_path"] not in golden_paths | extra_paths
        result = self.pipeline.delete_document(doc_id, user.username, archive_file=archive)
        get_index_manager().invalidate(doc["kb"])
        repository.write_audit(user.username, "document_delete", doc["source_path"], outcome=result.status,
                               details={"doc_id": doc_id, "note": "configured file kept on disk; remove it from "
                                        "knowledge_bases.yaml to stop re-ingestion" if not archive else "archived"})
        return result

    def sync(self, user: AuthenticatedUser, kb: str, full_reindex: bool = False) -> list[dict[str, Any]]:
        results = self.pipeline.reindex_kb(kb, user.username) if full_reindex else self.pipeline.sync_kb(kb, user.username)
        get_index_manager().invalidate(kb)
        repository.write_audit(user.username, "kb_reindex" if full_reindex else "kb_sync", kb,
                               details={"results": [r.status for r in results]})
        return [r.to_dict() for r in results]

    def stats(self) -> list[dict[str, Any]]:
        return self.pipeline.stats()
