"""Benchmark chunking, embedding models, rerankers and retrieval modes on the real knowledge base.

Every configuration is built in an isolated temporary data directory (the live index is never touched) and evaluated
on the in-scope questions of datasets/evaluation_dataset/evaluation_dataset.csv:

  retrieval metrics (approved-answer shortcut disabled, KB fixed to the expected one -> isolates chunking/embedding/
  reranking): Recall@5, MRR, NDCG@5, Precision@5, retrieval latency (p50/p95)
  end-to-end metrics (full production pipeline, no LLM): destination accuracy, answer correctness (key facts),
  refusal rate on in-scope questions, end-to-end latency
  index metrics: chunks, tokens per chunk (embedding-model tokenizer), indexing time

Usage
  python scripts/benchmark_rag.py --suite chunking          # chunk size / overlap / strategy
  python scripts/benchmark_rag.py --suite embedding         # embedding models (downloads models on first run)
  python scripts/benchmark_rag.py --suite reranker
  python scripts/benchmark_rag.py --suite retrieval         # dense / BM25 / hybrid / top-k
  python scripts/benchmark_rag.py --one '{"chunk_size": 900}'  # single config (JSON overrides)
Results: reports/benchmarks/<suite>.json and .md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = {"name": "current", "chunk_size": 900, "chunk_overlap": 150, "strategy": "structure",
            "embedding_backend": "sentence_transformers", "embedding_model": "BAAI/bge-small-en-v1.5",
            "reranker_backend": "cross_encoder", "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "vector_weight": 0.5, "top_k_vector": 20, "top_k_bm25": 20, "top_k_rerank": 5}

# 1 token ~ 4 characters for this English corpus (measured per run with the model tokenizer and reported)
SUITES: dict[str, list[dict]] = {
    "chunking": [
        {"name": "current 900c/150 (~225 tok)"},
        {"name": "300 tok / 15%", "chunk_size": 1200, "chunk_overlap": 180},
        {"name": "500 tok / 15%", "chunk_size": 2000, "chunk_overlap": 300},
        {"name": "800 tok / 15%", "chunk_size": 3200, "chunk_overlap": 480},
        {"name": "1000 tok / 15%", "chunk_size": 4000, "chunk_overlap": 600},
        {"name": "~225 tok / 0 overlap", "chunk_overlap": 0},
        {"name": "~225 tok / 30% overlap", "chunk_overlap": 270},
        {"name": "500 tok / 0 overlap", "chunk_size": 2000, "chunk_overlap": 0},
        {"name": "fixed 300 tok (no headings)", "strategy": "fixed", "chunk_size": 1200, "chunk_overlap": 180},
        {"name": "fixed 500 tok (no headings)", "strategy": "fixed", "chunk_size": 2000, "chunk_overlap": 300},
        {"name": "structure+recursive 900/150", "strategy": "structure_recursive", "chunk_size": 900,
         "chunk_overlap": 150},
        {"name": "structure+recursive 1200/180", "strategy": "structure_recursive", "chunk_size": 1200,
         "chunk_overlap": 180},
        {"name": "parent 1800 / child 400", "strategy": "parent_child", "chunk_size": 1800, "chunk_overlap": 0,
         "child_size": 400, "child_overlap": 80},
        {"name": "parent 2400 / child 600", "strategy": "parent_child", "chunk_size": 2400, "chunk_overlap": 0,
         "child_size": 600, "child_overlap": 100},
        {"name": "article-level (parent-size)", "chunk_size": 12000, "chunk_overlap": 0},
    ],
    "embedding": [
        {"name": "bge-small-en-v1.5 (current)"},
        {"name": "bge-base-en-v1.5", "embedding_model": "BAAI/bge-base-en-v1.5"},
        {"name": "bge-large-en-v1.5", "embedding_model": "BAAI/bge-large-en-v1.5"},
        {"name": "e5-large-v2", "embedding_model": "intfloat/e5-large-v2"},
        {"name": "bge-m3", "embedding_model": "BAAI/bge-m3"},
        {"name": "local_hash (no model)", "embedding_backend": "local_hash", "embedding_model": "local_hash"},
    ],
    "reranker": [
        {"name": "ms-marco-MiniLM-L-6 (current)"},
        {"name": "bge-reranker-base", "reranker_model": "BAAI/bge-reranker-base"},
        {"name": "bge-reranker-v2-m3", "reranker_model": "BAAI/bge-reranker-v2-m3"},
        {"name": "lexical (no model)", "reranker_backend": "lexical"},
        {"name": "no reranking", "reranker_backend": "none"},
    ],
    "retrieval": [
        {"name": "hybrid 50/50 + rerank (current)"},
        {"name": "dense only + rerank", "vector_weight": 1.0},
        {"name": "BM25 only + rerank", "vector_weight": 0.0},
        {"name": "hybrid 70/30 + rerank", "vector_weight": 0.7},
        {"name": "hybrid, no rerank", "reranker_backend": "none"},
        {"name": "dense only, no rerank", "vector_weight": 1.0, "reranker_backend": "none"},
        {"name": "hybrid top-10 candidates", "top_k_vector": 10, "top_k_bm25": 10},
        {"name": "hybrid top-40 candidates", "top_k_vector": 40, "top_k_bm25": 40},
    ],
}


# ============================================================================ single run (in a subprocess)
def _run_single(cfg: dict) -> dict:
    import logging
    import yaml

    logging.disable(logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_"))
    kb_cfg = yaml.safe_load((ROOT / "config/knowledge_bases.yaml").read_text(encoding="utf-8"))
    for kb in kb_cfg["knowledge_bases"].values():
        kb["chunking"] = {"chunk_size": cfg["chunk_size"], "chunk_overlap": cfg["chunk_overlap"],
                          "strategy": cfg["strategy"]}
        for key in ("child_size", "child_overlap"):     # parent_child strategy only
            if cfg.get(key) is not None:
                kb["chunking"][key] = cfg[key]
    if cfg.get("min_route_score") is not None:      # e.g. the original word-based setup was calibrated at 0.12
        kb_cfg.setdefault("routing", {})["min_route_score"] = cfg["min_route_score"]
    (tmp / "kb.yaml").write_text(yaml.safe_dump(kb_cfg, sort_keys=False), encoding="utf-8")
    os.environ.update({
        "DATA_DIR": str(tmp / "data"), "LOGS_DIR": str(tmp / "logs"), "KB_CONFIG_PATH": str(tmp / "kb.yaml"),
        "DATABASE_URL": "", "VECTOR_BACKEND": "numpy", "LLM_PROVIDER": "extractive", "ANTHROPIC_API_KEY": "",
        "EMBEDDING_BACKEND": cfg["embedding_backend"], "EMBEDDING_MODEL": cfg["embedding_model"],
        "RERANKER_BACKEND": cfg["reranker_backend"], "RERANKER_MODEL": cfg["reranker_model"],
        "HYBRID_VECTOR_WEIGHT": str(cfg["vector_weight"]), "TOP_K_VECTOR": str(cfg["top_k_vector"]),
        "TOP_K_BM25": str(cfg["top_k_bm25"]), "TOP_K_RERANK": str(cfg["top_k_rerank"]),
        "INGEST_WORKER_ENABLED": "false", "LOG_LEVEL": "WARNING"})
    sys.path.insert(0, str(ROOT))
    from backend.evaluation.metrics import retrieval_metrics
    from backend.routing.router import RouteDecision
    from backend.security.auth import AuthenticatedUser
    from backend.services.chat_service import ChatService
    from config.settings import get_settings
    from ingestion.embedding import get_embedder
    from ingestion.pipeline import IngestionPipeline
    from backend.evaluation.metrics import REFUSAL_TYPES, fact_coverage
    import csv

    get_settings().base_dir = ROOT
    t0 = time.perf_counter()
    embedder = get_embedder()
    load_s = time.perf_counter() - t0
    pipeline = IngestionPipeline()
    t0 = time.perf_counter()
    for kb in pipeline.kb_names():
        res = pipeline.sync_kb(kb)
        failed = [r for r in res if r.status == "failed"]
        if failed:
            raise RuntimeError(f"ingestion failed: {failed[0].message}")
    index_s = time.perf_counter() - t0

    from backend.retrieval.kb_index import get_index_manager
    chunks = [c for kb in pipeline.kb_names() for c in get_index_manager().get(kb).chunks
              if c.metadata.get("doc_type") != "golden_qa"]
    tok = getattr(getattr(embedder, "model", None), "tokenizer", None)
    lengths = [len(tok.encode(c.text)) if tok else len(c.text) // 4 for c in chunks]

    rows = list(csv.DictReader(open(ROOT / "datasets/evaluation_dataset/evaluation_dataset.csv", encoding="utf-8")))
    svc = ChatService(llm=None)
    user = AuthenticatedUser("bench", {"public"})
    ret, e2e, rlat, elat = [], [], [], []
    for row in rows:
        expect_either = row["expect_refusal"].lower() == "either"
        expect_refusal = row["expect_refusal"].lower() == "true"
        matchers = [m for m in row["relevant_sources"].split("|") if m.strip()]
        kbs = [k for k in row["expected_kb"].split("|") if k and k != "none"]
        if not expect_refusal and matchers and kbs:
            q = svc.router.expand(row["question"])
            t = time.perf_counter()
            qvec = svc.embedder.embed_query(q)
            cands = svc._retrieve(q, RouteDecision(intent="kb_query", knowledge_bases=kbs), {"public"}, qvec)
            ranked = svc.reranker.rerank(q, cands) if cands else []
            rlat.append((time.perf_counter() - t) * 1000)
            items = [{"source_file": c.meta.get("source_file"), "doc_name": c.meta.get("doc_name"),
                      "section": c.meta.get("section"), "qid": c.meta.get("qid", "")} for c in ranked[:10]]
            ret.append(retrieval_metrics(items, matchers, 5))
        t = time.perf_counter()
        resp = svc.answer(row["question"], user, persist=False)
        elat.append((time.perf_counter() - t) * 1000)
        refused = resp.answer_type in REFUSAL_TYPES
        dest = "refuse" if refused else (resp.knowledge_bases[0] if resp.knowledge_bases else "refuse")
        if expect_either:
            exp_dest = dest
        else:
            exp_dest = "refuse" if expect_refusal else (dest if dest in kbs else (kbs[0] if kbs else "refuse"))
        facts = [f for f in row["key_facts"].split("|") if f.strip()]
        if expect_either:
            correct = None
        else:
            correct = (1.0 if refused else 0.0) if expect_refusal else (
                fact_coverage(facts, resp.answer) if facts else None)
        e2e.append({"dest_ok": dest == exp_dest, "correct": correct,
                    "in_scope_refused": (not expect_refusal) and (not expect_either) and refused})

    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
    pct = lambda xs, q: round(sorted(xs)[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None  # noqa: E731
    return {
        "config": cfg, "questions_retrieval": len(ret), "questions_e2e": len(e2e),
        "recall_at_5": mean([r["recall_at_k"] for r in ret]), "mrr": mean([r["mrr"] for r in ret]),
        "ndcg_at_5": mean([r["ndcg_at_k"] for r in ret]), "precision_at_5": mean([r["precision_at_k"] for r in ret]),
        "destination_accuracy": mean([float(x["dest_ok"]) for x in e2e]),
        "answer_correctness": mean([x["correct"] for x in e2e if x["correct"] is not None]),
        "in_scope_refusal_rate": mean([float(x["in_scope_refused"]) for x in e2e]),
        "retrieval_p50_ms": pct(rlat, 0.5), "retrieval_p95_ms": pct(rlat, 0.95),
        "e2e_p50_ms": pct(elat, 0.5), "e2e_p95_ms": pct(elat, 0.95),
        "chunks": len(chunks), "tokens_per_chunk_median": statistics.median(lengths) if lengths else 0,
        "tokens_per_chunk_max": max(lengths) if lengths else 0, "index_seconds": round(index_s, 1),
        "model_load_seconds": round(load_s, 1),
        "embedding_dim": int(embedder.embed_query("x").shape[0]),
    }


# ============================================================================ orchestration
def run_config(overrides: dict) -> dict:
    cfg = {**BASELINE, **overrides}
    proc = subprocess.run([sys.executable, __file__, "--single", json.dumps(cfg)], capture_output=True, text=True,
                          cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=3600)
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"config": cfg, "error": (proc.stderr or proc.stdout)[-1500:]}


COLUMNS = [("recall_at_5", "Recall@5"), ("mrr", "MRR"), ("ndcg_at_5", "NDCG@5"), ("precision_at_5", "P@5"),
           ("destination_accuracy", "Dest. acc."), ("answer_correctness", "Answer correct."),
           ("in_scope_refusal_rate", "In-scope refusals"), ("retrieval_p50_ms", "Retr. p50 ms"),
           ("retrieval_p95_ms", "Retr. p95 ms"), ("e2e_p50_ms", "E2E p50 ms"), ("chunks", "Chunks"),
           ("tokens_per_chunk_median", "Tok/chunk med"), ("embedding_dim", "Dim"), ("index_seconds", "Index s")]


def to_markdown(suite: str, results: list[dict]) -> str:
    lines = [f"# Benchmark: {suite}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} on the live knowledge base "
             "(isolated copies). Retrieval metrics: approved-answer shortcut off, expected KB fixed, k=5. "
             "End-to-end: full pipeline without an LLM.", "",
             "| Configuration | " + " | ".join(c[1] for c in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['config']['name']} | ERROR: {r['error'][-160:].strip()} |")
            continue
        vals = []
        for key, _ in COLUMNS:
            v = r.get(key)
            vals.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) and v <= 1 and key not in
                                                  ("retrieval_p50_ms",) else str(v)))
        lines.append(f"| {r['config']['name']} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES))
    ap.add_argument("--one", help="JSON overrides for a single run")
    ap.add_argument("--single", help=argparse.SUPPRESS)
    ap.add_argument("--only", default="", help="comma-separated config names to run from the suite")
    args = ap.parse_args()
    if args.single:
        print("RESULT " + json.dumps(_run_single(json.loads(args.single)), default=str))
        return 0
    configs = SUITES[args.suite] if args.suite else [json.loads(args.one or "{}") | {"name": "custom"}]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        configs = [c for c in configs if c["name"] in wanted]
    out_dir = ROOT / "reports" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.suite or "custom"
    results = []
    existing_path = out_dir / f"{name}.json"
    if args.only and existing_path.exists():          # partial re-run: keep earlier configurations' results
        wanted_names = {c["name"] for c in configs}
        results = [r for r in json.loads(existing_path.read_text(encoding="utf-8"))
                   if r.get("config", {}).get("name") not in wanted_names]
    for cfg in configs:
        started = time.time()
        res = run_config(cfg)
        res["wall_seconds"] = round(time.time() - started, 1)
        results.append(res)
        msg = res.get("error", "")[-200:] if "error" in res else \
            f"R@5={res['recall_at_5']} MRR={res['mrr']} dest={res['destination_accuracy']} " \
            f"correct={res['answer_correctness']} p50={res['retrieval_p50_ms']}ms"
        print(f"[{name}] {cfg['name']}: {msg}", flush=True)
        (out_dir / f"{name}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        (out_dir / f"{name}.md").write_text(to_markdown(name, results), encoding="utf-8")
    print(f"\nWritten {out_dir / (name + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
