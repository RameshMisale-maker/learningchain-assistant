"""Command-line interface for knowledge-base management.

Examples
--------
  python -m ingestion.cli list-kbs
  python -m ingestion.cli sync                      # sync every enabled KB with its source folders
  python -m ingestion.cli sync --kb hr_policies
  python -m ingestion.cli reindex --kb products     # full rebuild (e.g. after changing EMBEDDING_BACKEND)
  python -m ingestion.cli add --kb products --file C:\\docs\\catalogue.pdf --roles public,employee
  python -m ingestion.cli delete --doc-id 1a2b3c4d5e6f7a8b
  python -m ingestion.cli purge --kb hr_policies    # remove indexed content (source files untouched)
  python -m ingestion.cli list-docs --kb company
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from backend.common.logging_config import setup_logging
from ingestion import registry
from ingestion.pipeline import IngestionPipeline


def _print_results(results) -> int:
    failures = 0
    for r in results:
        line = f"[{r.status.upper():9}] {r.kb:12} {r.source}  (v{r.version}, {r.chunks} chunks)"
        if r.message:
            line += f"  - {r.message}"
        print(line)
        for w in r.warnings[:5]:
            print(f"             warning: {w}")
        failures += r.status == "failed"
    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    print("Summary:", json.dumps(summary))
    return failures


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Sapphirus knowledge-base management")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-kbs")
    p = sub.add_parser("sync"); p.add_argument("--kb")
    p = sub.add_parser("reindex"); p.add_argument("--kb")
    p = sub.add_parser("purge"); p.add_argument("--kb", required=True); p.add_argument("--yes", action="store_true")
    p = sub.add_parser("add")
    p.add_argument("--kb", required=True); p.add_argument("--file", required=True)
    p.add_argument("--roles", default=""); p.add_argument("--category", default="")
    p.add_argument("--golden", action="store_true", help="treat the file as an approved Q&A dataset")
    p.add_argument("--no-copy", action="store_true", help="index in place instead of copying into the KB folder")
    p = sub.add_parser("delete"); p.add_argument("--doc-id", required=True)
    p = sub.add_parser("list-docs"); p.add_argument("--kb")
    args = parser.parse_args(argv)

    pipeline = IngestionPipeline()
    kbs = [args.kb] if getattr(args, "kb", None) else pipeline.kb_names()

    if args.command == "list-kbs":
        for s in pipeline.stats():
            print(json.dumps(s))
        return 0
    if args.command == "sync":
        return 1 if _print_results([r for kb in kbs for r in pipeline.sync_kb(kb)]) else 0
    if args.command == "reindex":
        return 1 if _print_results([r for kb in kbs for r in pipeline.reindex_kb(kb)]) else 0
    if args.command == "purge":
        if not args.yes and input(f"Purge all indexed content of '{args.kb}'? [y/N] ").lower() != "y":
            return 1
        print(f"Removed {pipeline.purge_kb(args.kb)} documents from registry/index of '{args.kb}'")
        return 0
    if args.command == "add":
        src = Path(args.file).resolve()
        target = src
        if not args.no_copy and not args.golden:
            target = pipeline.primary_upload_dir(args.kb) / src.name
            if target.resolve() != src:
                shutil.copy2(src, target)
        roles = [r.strip() for r in args.roles.split(",") if r.strip()] or None
        return 1 if _print_results([pipeline.ingest_file(args.kb, target, roles=roles, category=args.category or None,
                                                         golden=args.golden or None)]) else 0
    if args.command == "delete":
        return 1 if _print_results([pipeline.delete_document(args.doc_id)]) else 0
    if args.command == "list-docs":
        for d in registry.list_documents(args.kb):
            print(f"{d['doc_id']}  {d['kb']:12} v{d['version']:<3} {d['num_chunks']:>4} chunks  "
                  f"roles={','.join(d['access_roles'])}  {d['source_path']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
