"""Configuration-driven ingestion pipeline.

    file -> type detection -> extraction (text/tables/records, OCR if needed) -> cleaning
         -> chunking (+ contextual headers) -> injection scan -> embeddings -> vector store
         -> document registry (hash, version, ACL) -> KB revision bump

Supports: add, update (new version replaces old chunks), unchanged detection, duplicate
detection (same content under another path), delete (archive), full KB sync and re-index.
Knowledge-base definitions come from config/knowledge_bases.yaml.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.db import repository
from config.settings import Settings, get_settings
from ingestion import registry
from ingestion.chunking import Chunk, chunk_document
from ingestion.document_loader import SUPPORTED_EXTENSIONS, load_document, read_tabular
from ingestion.embedding import Embedder, get_embedder
from ingestion.text_cleaner import clean_document, detect_injection
from ingestion.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

ACCESS_FILE = "_access.yaml"
# placeholder titles that office tools write into exported files - never useful as a document title
_GENERIC_TITLE = re.compile(r"^(powerpoint presentation|presentation\d*|slide \d+|untitled.*|document\d*|"
                            r"microsoft (word|powerpoint|excel) - .*)$", re.IGNORECASE)
INACTIVE_GOLDEN_STATUSES = {"retired", "inactive", "draft", "rejected", "archived"}


@dataclass
class IngestResult:
    kb: str
    source: str
    status: str                # added | updated | unchanged | duplicate | deleted | failed
    doc_id: str = ""
    version: int = 0
    chunks: int = 0
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SOURCE_URL_RE = re.compile(r"(?:Article|Source|More|URL|Link)\s*:\s*[^\n]*?(https?://[^\s)\]>\"']+)", re.IGNORECASE)


def attach_source_urls(chunks: list, metadatas: list[dict[str, Any]]) -> None:
    """Give every chunk a `source_url` so answers can link to the original page: the URL declared in its own
    section (e.g. "Article: https://..."), else the document-level "Source: https://..." URL. An explicit
    `source_url` already in the metadata (golden Q&A column, web page) is kept."""
    section_url: dict[str, str] = {}
    doc_url = ""
    for ch in chunks:
        match = _SOURCE_URL_RE.search(ch.text)
        if match:
            url = match.group(1).rstrip(".,;")
            section_url.setdefault(ch.section or "", url)
            doc_url = doc_url or url
    for ch, meta in zip(chunks, metadatas):
        if not meta.get("source_url"):
            meta["source_url"] = section_url.get(ch.section or "") or doc_url or ""


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in re.split(r"[;,|]", str(value)) if v.strip()]


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None,
                 embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.config = self.settings.kb_config()

    # ------------------------------------------------------------ config helpers
    def kb_names(self) -> list[str]:
        return [name for name, cfg in self.config["knowledge_bases"].items() if cfg.get("enabled", True)]

    def kb_cfg(self, kb: str) -> dict[str, Any]:
        try:
            return self.config["knowledge_bases"][kb]
        except KeyError as exc:
            raise ValueError(f"Unknown knowledge base '{kb}'. Known: {', '.join(self.kb_names())}") from exc

    def _abs(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.settings.base_dir / p).resolve()

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.settings.base_dir).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    def doc_id_for(self, kb: str, path: Path) -> str:
        return hashlib.sha1(f"{kb}|{self._rel(path)}".encode()).hexdigest()[:16]

    def primary_upload_dir(self, kb: str) -> Path:
        dirs = self.kb_cfg(kb).get("source_dirs") or [f"knowledge_bases/{kb}"]
        path = self._abs(dirs[0])
        path.mkdir(parents=True, exist_ok=True)
        return path

    def access_rules_for(self, kb: str, path: Path) -> dict[str, Any]:
        """Resolve ACL/category for a file: KB defaults < folder _access.yaml defaults < file pattern rule."""
        cfg = self.kb_cfg(kb)
        rules: dict[str, Any] = {"roles": list(cfg.get("default_document_roles", cfg.get("allowed_roles", []))),
                                 "category": cfg.get("default_category", kb), "confidentiality": "internal",
                                 "sample": False}
        for folder in [path.parent, *path.parent.parents]:
            access_file = folder / ACCESS_FILE
            if access_file.exists():
                data = yaml.safe_load(access_file.read_text(encoding="utf-8")) or {}
                defaults = data.get("defaults", {})
                rules.update({k: v for k, v in defaults.items() if v is not None})
                for rule in data.get("files", []):
                    if fnmatch.fnmatch(path.name, rule.get("pattern", "")):
                        rules.update({k: v for k, v in rule.items() if k != "pattern" and v is not None})
                break
            if folder == self.settings.base_dir:
                break
        allowed = set(cfg.get("allowed_roles", []))
        rules["roles"] = [r for r in _split_list(rules["roles"]) if r in allowed or r == "admin"] or ["admin"]
        return rules

    # ------------------------------------------------------------ discovery
    def discover(self, kb: str) -> tuple[list[Path], list[Path]]:
        cfg = self.kb_cfg(kb)
        golden = [self._abs(p) for p in cfg.get("golden_files", [])]
        reviewed = self.settings.data_dir / "golden" / f"{kb}_reviewed.csv"   # approved in the review queue
        if reviewed.exists():
            golden.append(reviewed)
        golden_set = {g.resolve() for g in golden}
        files: list[Path] = []
        for d in cfg.get("source_dirs", []):
            folder = self._abs(d)
            if not folder.exists():
                logger.warning("Source folder %s for KB %s does not exist", folder, kb)
                continue
            for p in sorted(folder.rglob("*")):
                if (p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(("_", "."))
                        and p.resolve() not in golden_set):
                    files.append(p)
        files += [self._abs(p) for p in cfg.get("extra_files", [])]
        return [f for f in files if f.exists()], [g for g in golden if g.exists()]

    # ------------------------------------------------------------ core ingestion
    def ingest_file(self, kb: str, path: str | Path, uploaded_by: str = "system", roles: list[str] | None = None,
                    category: str | None = None, golden: bool | None = None, force: bool = False) -> IngestResult:
        path = self._abs(path)
        cfg = self.kb_cfg(kb)
        if golden is None:
            golden = path.resolve() in {self._abs(g).resolve() for g in cfg.get("golden_files", [])}
        result = IngestResult(kb=kb, source=self._rel(path), status="failed")
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            content_hash = file_hash(path)
            doc_id = self.doc_id_for(kb, path)
            result.doc_id = doc_id
            existing = registry.get_document(doc_id)
            if existing and existing["status"] == "active" and existing["content_hash"] == content_hash and not force \
                    and (roles is None or set(roles) == set(existing["access_roles"])):
                result.status, result.version, result.chunks = "unchanged", existing["version"], existing["num_chunks"]
                return result
            duplicate = registry.find_by_hash(kb, content_hash)
            if duplicate and duplicate["doc_id"] != doc_id and not force:
                result.status = "duplicate"
                result.message = f"Identical content already ingested as '{duplicate['doc_name']}' ({duplicate['doc_id']})"
                return result

            rules = self.access_rules_for(kb, path)
            if roles:
                allowed = set(cfg.get("allowed_roles", [])) | {"admin"}
                rules["roles"] = [r for r in roles if r in allowed] or ["admin"]
            if category:
                rules["category"] = category
            if golden:
                chunks, doc_meta, warnings = self._golden_chunks(kb, path, doc_id, rules)
                doc_type = "golden_qa"
            else:
                chunks, doc_meta, warnings = self._document_chunks(kb, path, doc_id)
                doc_type = "document"
            result.warnings = warnings
            if not chunks:
                raise ValueError("No content could be extracted (" + "; ".join(warnings or ["empty document"]) + ")")

            version = existing["version"] + 1 if existing else 1
            now = datetime.now(timezone.utc).isoformat()
            metadatas: list[dict[str, Any]] = []
            for ch in chunks:
                meta = {
                    "chunk_id": ch.chunk_id, "doc_id": doc_id, "doc_name": doc_meta["title"], "file_name": path.name,
                    "kb": kb, "category": ch.metadata.pop("category", None) or rules["category"],
                    "page": ch.page if ch.page is not None else -1, "section": ch.section or "",
                    "source_file": self._rel(path), "version": version, "uploaded_at": now,
                    "uploaded_by": uploaded_by, "chunk_index": ch.chunk_index, "chunk_kind": ch.kind,
                    "doc_type": doc_type, "confidentiality": rules["confidentiality"],
                    "is_sample": bool(rules.get("sample", False)),
                    "access_roles": ch.metadata.pop("access_roles", None) or rules["roles"],
                    "embed_text": ch.embed_text,
                }
                flags = detect_injection(ch.text)
                meta["injection_flag"] = bool(flags)
                if flags:
                    result.warnings.append(f"Possible prompt-injection text in chunk {ch.chunk_index}: {flags[0]!r}")
                meta.update(ch.metadata)
                meta["content_hash"] = content_hash
                metadatas.append(meta)
            attach_source_urls(chunks, metadatas)

            embeddings = self.embedder.embed_documents([c.embed_text for c in chunks])
            # new version first, then remove only the stale chunks: the document is never missing from search,
            # and a failure before this point leaves the previous version fully intact
            new_ids = [c.chunk_id for c in chunks]
            self.store.upsert(kb, new_ids, embeddings, [c.text for c in chunks], metadatas, self.embedder.model_id)
            self.store.delete_document(kb, doc_id, keep_ids=set(new_ids))
            saved = registry.upsert_document(
                doc_id=doc_id, kb=kb, doc_name=doc_meta["title"], source_path=self._rel(path),
                file_type=path.suffix.lstrip(".").lower(), doc_type=doc_type, category=rules["category"],
                access_roles=",".join(rules["roles"]), confidentiality=rules["confidentiality"],
                content_hash=content_hash, num_chunks=len(chunks), is_sample=bool(rules.get("sample", False)),
                uploaded_by=uploaded_by, notes="; ".join(result.warnings)[:2000])
            repository.bump_kb_revision(kb, self.embedder.model_id)
            result.status = "updated" if existing and existing["status"] == "active" else "added"
            result.version, result.chunks = saved["version"], len(chunks)
            logger.info("Ingested %s into %s (%s, v%s, %d chunks)", path.name, kb, result.status, result.version, len(chunks))
        except Exception as exc:
            logger.exception("Failed to ingest %s into %s", path, kb)
            result.status, result.message = "failed", str(exc)
        return result

    def _document_chunks(self, kb: str, path: Path, doc_id: str) -> tuple[list[Chunk], dict[str, Any], list[str]]:
        cfg = self.kb_cfg(kb)
        doc = clean_document(load_document(path))
        title = self._title_for(doc, path)
        chunking = cfg.get("chunking", {})
        chunks = chunk_document(doc, doc_id, chunking.get("chunk_size", 900), chunking.get("chunk_overlap", 150),
                                strategy=chunking.get("strategy", "structure"),
                                child_size=chunking.get("child_size", 400),
                                child_overlap=chunking.get("child_overlap", 80),
                                title=title)
        entity_fields = cfg.get("entity_fields", [])
        for ch in chunks:
            record = ch.metadata.pop("record", None)
            if record:
                ch.metadata["record_json"] = json.dumps(record, ensure_ascii=False, default=str)
                lowered = {str(k).strip().lower(): v for k, v in record.items()}
                lowered.update({k.replace(" ", "_"): v for k, v in list(lowered.items())})
                names = []
                for ef in entity_fields:
                    value = lowered.get(ef.strip().lower()) or lowered.get(ef.strip().lower().replace(" ", "_"))
                    if value:
                        ch.metadata[ef] = str(value)
                        names.append(str(value))
                if names:   # canonical entity (first configured field) + all aliases, used by retrieval/comparison
                    ch.metadata["entity_name"] = names[0]
                    ch.metadata["entity_aliases"] = names
                if lowered.get("category"):
                    ch.metadata["category"] = str(lowered["category"])
                if lowered.get("access_roles"):
                    ch.metadata["access_roles"] = _split_list(lowered["access_roles"])
            ch.metadata.pop("row", None)
            ch.metadata.pop("ocr", None)
            ch.metadata.pop("sheet", None)
        return chunks, {"title": title}, doc.warnings

    @staticmethod
    def _title_for(doc, path: Path) -> str:
        title = (doc.metadata.get("title") or "").strip()
        if _GENERIC_TITLE.match(title):
            title = ""          # e.g. "PowerPoint Presentation" from slide decks exported to PDF
        if not title and path.suffix.lower() == ".pdf":
            # exported slides/manuals start with cover text ("www.<company-site>", "USER OPERATIONS"); the file name
            # is the most descriptive title, and every chunk header carries it
            return path.stem.replace("_", " ").strip()
        if not title:
            for sec in doc.sections[:3]:
                first = sec.text.strip().split("\n", 1)[0].lstrip("# ").strip()
                if sec.kind == "text" and 5 <= len(first) <= 90 and not first.endswith((".", ":", ";", ",")):
                    title = first
                    break
        return title or path.stem.replace("_", " ")

    def _golden_chunks(self, kb: str, path: Path, doc_id: str, rules: dict[str, Any]
                       ) -> tuple[list[Chunk], dict[str, Any], list[str]]:
        cols = {"id": "Question ID", "category": "Category", "question": "Question", "answer": "Approved Answer",
                "keywords": "Keywords", "alternates": "Alternate Questions", "roles": "Access Roles", "status": "Status",
                "source_url": "Source URL"}
        cols.update(self.kb_cfg(kb).get("golden_columns", {}))
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
        else:
            rows = [r for df in read_tabular(path).values() for r in df.to_dict(orient="records")]
        warnings: list[str] = []
        chunks: list[Chunk] = []
        seen: set[str] = set()
        allowed = set(self.kb_cfg(kb).get("allowed_roles", [])) | {"admin"}
        for idx, row in enumerate(rows, start=1):
            def col(name: str) -> str:
                value = row.get(cols[name], row.get(name, ""))
                if isinstance(value, list):
                    return " | ".join(str(v) for v in value)
                return "" if value is None else str(value).strip()

            qid, question, answer = col("id") or f"ROW{idx:04d}", col("question"), col("answer")
            if not question or not answer:
                warnings.append(f"Row {idx}: missing question or answer - skipped")
                continue
            if col("status").lower() in INACTIVE_GOLDEN_STATUSES:
                continue
            if qid in seen:
                warnings.append(f"Duplicate question id {qid} - later row skipped")
                continue
            seen.add(qid)
            alternates = [a.strip() for a in re.split(r"\|", col("alternates")) if a.strip()]
            keywords = [k for k in re.split(r"[,;]", col("keywords")) if k.strip()]
            row_roles = [r for r in _split_list(col("roles")) if r in allowed] or rules["roles"]
            category = col("category") or rules["category"]
            text = f"Q: {question}\nA: {answer}"
            embed = "\n".join([f"Approved FAQ ({category})", question, *alternates,
                               "Keywords: " + ", ".join(k.strip() for k in keywords), answer])
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{qid}", text=text, embed_text=embed, page=None, section=f"{qid} - {category}",
                kind="golden", chunk_index=len(chunks),
                metadata={"qid": qid, "question": question, "answer": answer, "alternates": " || ".join(alternates),
                          "keywords": ", ".join(k.strip() for k in keywords), "category": category,
                          "access_roles": row_roles, "golden_status": col("status"),
                          "source_url": col("source_url")}))
        return chunks, {"title": path.stem.replace("_", " ").title()}, warnings

    # ------------------------------------------------------------ lifecycle operations
    def delete_document(self, doc_id: str, deleted_by: str = "system", archive_file: bool = True) -> IngestResult:
        doc = registry.get_document(doc_id)
        if not doc:
            return IngestResult(kb="", source="", status="failed", doc_id=doc_id, message="Unknown document id")
        removed = self.store.delete_document(doc["kb"], doc_id)
        registry.mark_deleted(doc_id)
        repository.bump_kb_revision(doc["kb"])
        source = self._abs(doc["source_path"])
        if archive_file and source.exists():
            archive_dir = self.settings.data_dir / "archive" / doc["kb"]
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            shutil.move(str(source), archive_dir / f"{stamp}__{source.name}")
        logger.info("Deleted document %s (%s) from %s by %s", doc["doc_name"], doc_id, doc["kb"], deleted_by)
        return IngestResult(kb=doc["kb"], source=doc["source_path"], status="deleted", doc_id=doc_id,
                            version=doc["version"], chunks=removed)

    def sync_kb(self, kb: str, uploaded_by: str = "system") -> list[IngestResult]:
        """Bring a KB in line with its source folders: add new, update changed, remove deleted files."""
        files, golden = self.discover(kb)
        results = [self.ingest_file(kb, f, uploaded_by=uploaded_by, golden=False) for f in files]
        results += [self.ingest_file(kb, g, uploaded_by=uploaded_by, golden=True) for g in golden]
        present = {self._rel(p) for p in files + golden}
        for doc in registry.list_documents(kb):
            if doc["source_path"] not in present:
                results.append(self.delete_document(doc["doc_id"], uploaded_by, archive_file=False))
        return results

    def reindex_kb(self, kb: str, uploaded_by: str = "system") -> list[IngestResult]:
        """Drop the KB collection and registry entries, then re-ingest everything (e.g. after changing embedder).
        With INDEX_VERSIONING_ENABLED the rebuild goes to a new index version instead (eval gate, then promote) and
        users are served by the current version until the new one is live."""
        if self.settings.index_versioning_enabled:
            from backend.indexing.versions import build_version

            return build_version(kb, uploaded_by, pipeline=self)["ingest_results"]
        # Preflight: prove the embedder works (API key, model access, network) BEFORE dropping anything, so a
        # misconfigured embedding model can never leave the KB empty.
        try:
            self.embedder.embed_documents(["reindex preflight check"])
        except Exception as exc:
            raise RuntimeError(f"Embedding model '{self.embedder.model_id}' is not usable - the existing index of "
                               f"'{kb}' was left untouched. Cause: {exc}") from exc
        self.store.drop(kb)
        registry.purge_kb(kb)
        repository.bump_kb_revision(kb, self.embedder.model_id)
        return self.sync_kb(kb, uploaded_by)

    def purge_kb(self, kb: str) -> int:
        """Remove all indexed content of a KB (source files are left untouched)."""
        self.store.drop(kb)
        count = registry.purge_kb(kb)
        repository.bump_kb_revision(kb)
        return count

    def stats(self) -> list[dict[str, Any]]:
        out = []
        for kb in self.kb_names():
            cfg = self.kb_cfg(kb)
            docs = registry.list_documents(kb)
            out.append({"kb": kb, "display_name": cfg.get("display_name", kb), "documents": len(docs),
                        "chunks": self.store.count(kb), "allowed_roles": cfg.get("allowed_roles", []),
                        "embedding_model": self.store.model_id(kb), "revision": repository.get_kb_revision(kb),
                        "sample_documents": sum(1 for d in docs if d["is_sample"])})
        return out
