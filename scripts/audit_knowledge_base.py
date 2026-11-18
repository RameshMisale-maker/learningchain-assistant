"""Audit the indexed knowledge base: extraction completeness, structure, duplicates, metadata and chunk sizes.

Usage: python scripts/audit_knowledge_base.py [--json reports/kb_audit.json]
Compares indexed chunks with the source files on disk, so it reports what the chatbot can actually retrieve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from backend.retrieval.kb_index import get_index_manager  # noqa: E402

REQUIRED_META = ["kb", "doc_id", "doc_name", "source_file", "access_roles", "content_hash", "version"]


def words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def audit() -> dict:
    mgr = get_index_manager()
    report: dict = {"knowledge_bases": {}}
    for kb in mgr.kb_names():
        idx = mgr.get(kb)
        chunks = list(idx.chunks)
        by_file: dict[str, list] = defaultdict(list)
        for c in chunks:
            by_file[str(c.metadata.get("source_file", "?"))].append(c)
        sizes = [len(c.text) for c in chunks if c.metadata.get("doc_type") != "golden_qa"]
        tokens = [round(len(c.text) / 4) for c in chunks if c.metadata.get("doc_type") != "golden_qa"]
        text_hash = Counter(hashlib.sha1(re.sub(r"\s+", " ", c.text).strip().lower().encode()).hexdigest()
                            for c in chunks)
        dup_chunks = sum(n - 1 for n in text_hash.values() if n > 1)
        missing_meta = Counter(k for c in chunks for k in REQUIRED_META if c.metadata.get(k) in (None, "", []))
        tiny = [c for c in chunks if c.metadata.get("doc_type") != "golden_qa" and words(c.text) < 25]

        files = []
        for src, cs in sorted(by_file.items()):
            path = ROOT / src
            entry = {"file": src, "chunks": len(cs), "version": cs[0].metadata.get("version")}
            if path.suffix == ".md" and path.exists():
                raw = path.read_text(encoding="utf-8")
                headings = re.findall(r"^#{1,6}\s+(.+)$", raw, flags=re.M)
                indexed = " ".join(c.text for c in cs)
                lost = [h for h in headings if h.strip()[:60] not in indexed]
                # coverage: share of distinct source sentences (>=6 words) found verbatim in indexed text
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw) if words(s) >= 6]
                norm_idx = re.sub(r"\s+", " ", indexed)
                found = sum(1 for s in sents if re.sub(r"\s+", " ", s)[:80] in norm_idx)
                entry.update({"headings": len(headings), "headings_missing": lost[:5],
                              "sentence_coverage": round(found / len(sents), 3) if sents else None})
            files.append(entry)

        sec_counts = Counter(str(c.metadata.get("section", "")) for c in chunks)
        report["knowledge_bases"][kb] = {
            "documents": len(by_file), "chunks": len(chunks),
            "golden_qa_chunks": sum(1 for c in chunks if c.metadata.get("doc_type") == "golden_qa"),
            "chunk_chars": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes),
                            "mean": round(statistics.mean(sizes))} if sizes else {},
            "chunk_tokens_approx": {"median": statistics.median(tokens), "max": max(tokens)} if tokens else {},
            "duplicate_chunks": dup_chunks,
            "tiny_chunks_lt_25_words": len(tiny),
            "tiny_examples": [c.text[:90] for c in tiny[:3]],
            "chunks_missing_metadata": dict(missing_meta),
            "chunks_without_section": sec_counts.get("", 0),
            "embedding_model": chunks[0].metadata.get("embedding_model") if chunks else None,
            "files": files,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports" / "kb_audit.json"))
    args = ap.parse_args()
    rep = audit()
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    for kb, r in rep["knowledge_bases"].items():
        print(f"\n== {kb}: {r['documents']} documents, {r['chunks']} chunks ({r['golden_qa_chunks']} approved Q&A)")
        print(f"   chunk chars {r['chunk_chars']}  ~tokens {r['chunk_tokens_approx']}")
        print(f"   duplicate chunks: {r['duplicate_chunks']}   tiny chunks (<25 words): {r['tiny_chunks_lt_25_words']}"
              f"   no section: {r['chunks_without_section']}   missing metadata: {r['chunks_missing_metadata'] or 'none'}")
        for t in r["tiny_examples"]:
            print(f"     tiny: {t!r}")
        for f in r["files"]:
            extra = (f"  headings {f['headings']}, missing {len(f['headings_missing'])}, sentence coverage "
                     f"{f['sentence_coverage']}") if "headings" in f else ""
            print(f"   - {f['file']}: {f['chunks']} chunks v{f['version']}{extra}")
            for h in f.get("headings_missing", []):
                print(f"       missing heading: {h}")
    print(f"\nFull report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
