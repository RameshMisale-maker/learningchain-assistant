"""Vector store abstraction with ChromaDB (default) and a flat NumPy index backend.

One collection per knowledge base (`<VECTOR_COLLECTION_PREFIX><kb>`) keeps the knowledge bases
physically separated. Metadata values are flattened to scalars (Chroma requirement);
list values such as access roles are stored as ",role1,role2," strings.
"""
from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


def collection_name(kb: str) -> str:
    """Collection serving `kb`: the version being built (inside `building()`), else the live index version
    (INDEX_VERSIONING_ENABLED), else the single per-KB collection."""
    from backend.indexing.versions import resolve_collection

    resolved = resolve_collection(kb)
    if resolved:
        return resolved
    return base_collection_name(kb)


def base_collection_name(kb: str) -> str:
    from config.settings import get_settings
    return f"{get_settings().vector_collection_prefix}{kb}"


def flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            flat[key] = "," + ",".join(str(v) for v in value) + ","
        elif isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def roles_from_meta(meta: dict[str, Any]) -> set[str]:
    return {r for r in str(meta.get("access_roles", "")).split(",") if r}


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, kb: str, ids: list[str], embeddings: np.ndarray, texts: list[str],
               metadatas: list[dict[str, Any]], model_id: str) -> None: ...

    @abstractmethod
    def query(self, kb: str, embedding: np.ndarray, k: int) -> list[StoredChunk]: ...

    @abstractmethod
    def get_all(self, kb: str) -> list[StoredChunk]: ...

    @abstractmethod
    def delete_document(self, kb: str, doc_id: str, keep_ids: set[str] | None = None) -> int:
        """Delete a document's chunks, except `keep_ids` (used to swap versions without a gap)."""

    @abstractmethod
    def drop(self, kb: str) -> None: ...

    @abstractmethod
    def count(self, kb: str) -> int: ...

    @abstractmethod
    def model_id(self, kb: str) -> str | None: ...

    def ensure_model(self, kb: str, model_id: str) -> None:
        existing = self.model_id(kb)
        if existing and existing != model_id and self.count(kb) > 0:
            raise RuntimeError(
                f"Knowledge base '{kb}' was indexed with '{existing}' but the current embedder is '{model_id}'. "
                f"Run: python -m ingestion.cli reindex --kb {kb}")


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Path) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._lock = threading.Lock()

    def _collection(self, kb: str, model_id: str | None = None, create: bool = True):
        name = collection_name(kb)
        if create:
            meta = {"hnsw:space": "cosine", "kb": kb}
            if model_id:
                meta["embedding_model"] = model_id
            return self.client.get_or_create_collection(name=name, metadata=meta, embedding_function=None)
        try:
            return self.client.get_collection(name=name, embedding_function=None)
        except Exception:
            return None

    def upsert(self, kb, ids, embeddings, texts, metadatas, model_id):
        with self._lock:
            self.ensure_model(kb, model_id)
            col = self._collection(kb, model_id)
            for start in range(0, len(ids), 500):
                end = start + 500
                col.upsert(ids=ids[start:end], embeddings=np.asarray(embeddings[start:end]).tolist(),
                           documents=texts[start:end], metadatas=[flatten_metadata(m) for m in metadatas[start:end]])

    def query(self, kb, embedding, k):
        col = self._collection(kb, create=False)
        if col is None or col.count() == 0:
            return []
        res = col.query(query_embeddings=[np.asarray(embedding).tolist()], n_results=min(k, col.count()),
                        include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(StoredChunk(id=cid, text=doc or "", metadata=dict(meta or {}), score=1.0 - float(dist)))
        return out

    def get_all(self, kb):
        col = self._collection(kb, create=False)
        if col is None:
            return []
        res = col.get(include=["documents", "metadatas"])
        return [StoredChunk(id=i, text=d or "", metadata=dict(m or {}))
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_document(self, kb, doc_id, keep_ids=None):
        col = self._collection(kb, create=False)
        if col is None:
            return 0
        with self._lock:
            existing = col.get(where={"doc_id": doc_id}, include=[])
            stale = [i for i in existing["ids"] if not keep_ids or i not in keep_ids]
            if stale:
                col.delete(ids=stale)
            return len(stale)

    def drop(self, kb):
        with self._lock:
            try:
                self.client.delete_collection(collection_name(kb))
            except Exception as exc:  # collection does not exist yet - nothing to drop
                logger.debug("drop(%s): %s", kb, exc)

    def count(self, kb):
        col = self._collection(kb, create=False)
        return col.count() if col is not None else 0

    def model_id(self, kb):
        col = self._collection(kb, create=False)
        if col is None:
            return None
        return (col.metadata or {}).get("embedding_model")


class NumpyVectorStore(VectorStore):
    """Flat (exact) cosine index persisted as .npy + .json per KB. Good for small corpora / tests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict[str, Any], np.ndarray]] = {}

    def _files(self, kb: str) -> tuple[Path, Path]:
        return self.path / f"{collection_name(kb)}.json", self.path / f"{collection_name(kb)}.npy"

    def _load(self, kb: str) -> tuple[dict[str, Any], np.ndarray]:
        meta_file, vec_file = self._files(kb)
        if not meta_file.exists():
            return {"model_id": None, "ids": [], "texts": [], "metadatas": []}, np.zeros((0, 0), dtype=np.float32)
        mtime = meta_file.stat().st_mtime
        cached = self._cache.get(str(meta_file))
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        vecs = np.load(vec_file) if vec_file.exists() else np.zeros((0, 0), dtype=np.float32)
        self._cache[str(meta_file)] = (mtime, data, vecs)
        return data, vecs

    def _save(self, kb: str, data: dict[str, Any], vecs: np.ndarray) -> None:
        meta_file, vec_file = self._files(kb)
        np.save(vec_file, vecs.astype(np.float32))
        tmp = meta_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(meta_file)
        self._cache.pop(str(meta_file), None)

    def upsert(self, kb, ids, embeddings, texts, metadatas, model_id):
        with self._lock:
            self.ensure_model(kb, model_id)
            data, vecs = self._load(kb)
            data = {**data, "ids": list(data["ids"]), "texts": list(data["texts"]), "metadatas": list(data["metadatas"])}
            index = {cid: i for i, cid in enumerate(data["ids"])}
            emb = np.asarray(embeddings, dtype=np.float32)
            if vecs.size == 0:
                vecs = np.zeros((0, emb.shape[1]), dtype=np.float32)
            new_rows = []
            for cid, vec, text, meta in zip(ids, emb, texts, metadatas):
                if cid in index:
                    i = index[cid]
                    vecs[i] = vec
                    data["texts"][i] = text
                    data["metadatas"][i] = flatten_metadata(meta)
                else:
                    data["ids"].append(cid)
                    data["texts"].append(text)
                    data["metadatas"].append(flatten_metadata(meta))
                    new_rows.append(vec)
            if new_rows:
                vecs = np.vstack([vecs, np.asarray(new_rows, dtype=np.float32)])
            data["model_id"] = model_id
            self._save(kb, data, vecs)

    def query(self, kb, embedding, k):
        data, vecs = self._load(kb)
        if not data["ids"]:
            return []
        scores = vecs @ np.asarray(embedding, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [StoredChunk(id=data["ids"][i], text=data["texts"][i], metadata=dict(data["metadatas"][i]),
                            score=float(scores[i])) for i in top]

    def get_all(self, kb):
        data, _ = self._load(kb)
        return [StoredChunk(id=i, text=t, metadata=dict(m)) for i, t, m in zip(data["ids"], data["texts"], data["metadatas"])]

    def delete_document(self, kb, doc_id, keep_ids=None):
        with self._lock:
            data, vecs = self._load(kb)
            keep = [i for i, m in enumerate(data["metadatas"])
                    if m.get("doc_id") != doc_id or (keep_ids and data["ids"][i] in keep_ids)]
            removed = len(data["ids"]) - len(keep)
            if removed:
                new = {"model_id": data["model_id"], "ids": [data["ids"][i] for i in keep],
                       "texts": [data["texts"][i] for i in keep], "metadatas": [data["metadatas"][i] for i in keep]}
                self._save(kb, new, vecs[keep] if len(keep) else np.zeros((0, vecs.shape[1]), dtype=np.float32))
            return removed

    def drop(self, kb):
        with self._lock:
            for f in self._files(kb):
                if f.exists():
                    f.unlink()
                self._cache.pop(str(f), None)

    def count(self, kb):
        return len(self._load(kb)[0]["ids"])

    def model_id(self, kb):
        return self._load(kb)[0].get("model_id")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_backend == "chroma":
        return ChromaVectorStore(settings.data_dir / "chroma")
    if settings.vector_backend == "numpy":
        return NumpyVectorStore(settings.data_dir / "numpy_index")
    raise ValueError(f"Unknown VECTOR_BACKEND '{settings.vector_backend}'")
