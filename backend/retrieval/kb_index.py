"""In-memory per-knowledge-base indexes built from the vector store contents.

For every KB this keeps:
  * the chunk list (text + metadata) and a BM25 keyword index
  * document-frequency statistics (used by the query router)
  * an entity lexicon (e.g. product names/codes) from configured `entity_fields`
  * the golden Q&A entries

Indexes are rebuilt automatically when the KB revision in the database changes
(i.e. after any ingestion, update or delete - even from another process).
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from backend.common.text_utils import tokenize
from backend.db import repository
from config.settings import get_settings
from ingestion.vector_store import StoredChunk, VectorStore, collection_name, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class KBIndex:
    kb: str
    revision: int
    chunks: list[StoredChunk]
    tokens: list[list[str]]
    bm25: BM25Okapi | None
    doc_freq: Counter
    entity_aliases: dict[str, str] = field(default_factory=dict)   # alias(lower) -> canonical name
    golden: list[StoredChunk] = field(default_factory=list)
    id_to_pos: dict[str, int] = field(default_factory=dict)
    collection: str = ""            # vector collection the index was built from (index versioning)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        n = max(self.size, 1)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def detect_entities(self, text: str) -> list[str]:
        lowered = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
        found: list[tuple[int, str]] = []
        for alias, canonical in sorted(self.entity_aliases.items(), key=lambda kv: -len(kv[0])):
            pos = lowered.find(f" {alias} ")
            if pos >= 0 and canonical not in [c for _, c in found]:
                found.append((pos, canonical))
        return [c for _, c in sorted(found)]


class KBIndexManager:
    def __init__(self, store: VectorStore | None = None, refresh_seconds: float = 2.0) -> None:
        self.store = store or get_vector_store()
        self.refresh_seconds = refresh_seconds
        self._indexes: dict[str, KBIndex] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()

    def kb_names(self) -> list[str]:
        cfg = get_settings().kb_config()["knowledge_bases"]
        return [k for k, v in cfg.items() if v.get("enabled", True)]

    def get(self, kb: str) -> KBIndex:
        now = time.monotonic()
        idx = self._indexes.get(kb)
        if idx is not None and now - self._checked.get(kb, 0) < self.refresh_seconds:
            return idx
        revision = repository.get_kb_revision(kb)
        collection = collection_name(kb)
        self._checked[kb] = now
        if idx is not None and idx.revision == revision and idx.collection == collection:
            return idx
        with self._lock:
            idx = self._indexes.get(kb)
            if idx is None or idx.revision != revision or idx.collection != collection:
                idx = self._build(kb, revision)
                idx.collection = collection
                self._indexes[kb] = idx
        return idx

    def invalidate(self, kb: str | None = None) -> None:
        if kb:
            self._indexes.pop(kb, None)
        else:
            self._indexes.clear()

    def _build(self, kb: str, revision: int) -> KBIndex:
        started = time.perf_counter()
        chunks = self.store.get_all(kb)
        tokens = [tokenize(c.metadata.get("embed_text") or c.text) for c in chunks]
        doc_freq: Counter = Counter()
        for toks in tokens:
            doc_freq.update(set(toks))
        bm25 = BM25Okapi(tokens) if tokens and any(tokens) else None
        cfg = get_settings().kb_config()["knowledge_bases"].get(kb, {})
        entity_fields = cfg.get("entity_fields", [])
        aliases: dict[str, str] = {}
        for c in chunks:
            if not entity_fields or not c.metadata.get("entity_name"):
                continue
            canonical = str(c.metadata["entity_name"])  # first configured entity field (e.g. product_name)
            values = [v for v in str(c.metadata.get("entity_aliases", "")).split(",") if v.strip()] or [canonical]
            for v in values:
                alias = re.sub(r"[^a-z0-9 ]+", " ", v.lower()).strip()
                if len(alias) >= 3:
                    aliases.setdefault(alias, canonical)
        golden = [c for c in chunks if c.metadata.get("doc_type") == "golden_qa"]
        idx = KBIndex(kb=kb, revision=revision, chunks=chunks, tokens=tokens, bm25=bm25, doc_freq=doc_freq,
                      entity_aliases=aliases, golden=golden, id_to_pos={c.id: i for i, c in enumerate(chunks)})
        logger.info("Built index for KB '%s' rev %s: %d chunks, %d golden, %d entity aliases (%.0f ms)", kb, revision,
                    len(chunks), len(golden), len(aliases), (time.perf_counter() - started) * 1000)
        return idx


_manager: KBIndexManager | None = None


def get_index_manager() -> KBIndexManager:
    global _manager
    if _manager is None:
        _manager = KBIndexManager()
    return _manager
