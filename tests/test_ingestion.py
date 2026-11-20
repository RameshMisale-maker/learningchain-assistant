"""Ingestion lifecycle: sync, unchanged, duplicate, update (versioning), delete, ACL rules, golden rows."""
import shutil

from ingestion import registry


def test_initial_sync_indexed_all_kbs(pipeline):
    stats = {s["kb"]: s for s in pipeline.stats()}
    assert set(stats) == {"company", "products"}                      # HR knowledge base removed
    # >= : other test modules add documents to the shared workspace (uploads, crawled pages)
    assert stats["company"]["documents"] >= 4 and stats["company"]["chunks"] > 30     # 3 pages + approved Q&A
    assert stats["products"]["documents"] >= 9 and stats["products"]["chunks"] > 200  # help centre + FAQ + Q&A


def test_resync_is_idempotent(pipeline):
    results = pipeline.sync_kb("products")
    assert {r.status for r in results} == {"unchanged"}


def test_golden_rows_are_individual_chunks(pipeline):
    chunks = [c for c in pipeline.store.get_all("products") if c.metadata.get("doc_type") == "golden_qa"]
    import pandas as pd
    from config.settings import BASE_DIR
    served = sum(int((~pd.read_csv(BASE_DIR / f)["Status"].astype(str).str.lower().isin(
        {"retired", "inactive", "draft", "rejected", "archived"})).sum())
        for f in ("datasets/products_golden_dataset.csv", "datasets/learningchain_faq_master.csv"))
    assert len(chunks) == served                 # one chunk per served approved answer (older Q&A + FAQ_Master)
    f05 = next(c for c in chunks if c.metadata["qid"] == "F05")
    assert "Azure" in f05.metadata["answer"] and f05.metadata["source_url"] == "https://learningchain.in/faqs/"
    assert ",public," in f05.metadata["access_roles"]


def test_access_rules_from_sidecar(pipeline, workspace):
    guide = next(d for d in registry.list_documents("products") if d["source_path"].endswith("Admin_Guide.md"))
    assert set(guide["access_roles"]) == {"public", "employee", "hr", "admin"}   # defaults from products/_access.yaml
    assert guide["confidentiality"] == "public"


def test_duplicate_detection(pipeline, workspace):
    src = workspace / "knowledge_bases/products/LearningChain_06_Verifier_Guide.md"
    dup = workspace / "knowledge_bases/products/Verifier_copy.md"
    shutil.copy(src, dup)
    result = pipeline.ingest_file("products", dup)
    assert result.status == "duplicate"
    dup.unlink()


def test_update_bumps_version_and_replaces_chunks(pipeline, workspace):
    path = workspace / "knowledge_bases/company/Test_Notice.txt"
    path.write_text("Parking notice (sample): visitor parking is on level B2.", encoding="utf-8")
    first = pipeline.ingest_file("company", path)
    assert first.status == "added" and first.version == 1
    path.write_text("Parking notice (sample): visitor parking moved to level B3.", encoding="utf-8")
    second = pipeline.ingest_file("company", path)
    assert second.status == "updated" and second.version == 2
    texts = [c.text for c in pipeline.store.get_all("company") if c.metadata["doc_id"] == first.doc_id]
    assert len(texts) == 1 and "B3" in texts[0]
    deleted = pipeline.delete_document(first.doc_id, archive_file=True)
    assert deleted.status == "deleted" and not path.exists()
    assert not [c for c in pipeline.store.get_all("company") if c.metadata["doc_id"] == first.doc_id]
