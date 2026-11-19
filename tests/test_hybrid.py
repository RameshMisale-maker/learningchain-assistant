"""Hybrid search: dense + BM25 fusion (RRF and relative score fusion) and multi-query (RAG-Fusion) retrieval."""
from __future__ import annotations

PUBLIC = {"public"}


def test_hybrid_combines_dense_and_keyword_hits(chat):
    found = chat.retriever.retrieve("How do I revoke a certificate?", "products", PUBLIC, top_n=20)
    assert found and found == sorted(found, key=lambda c: c.fused_score, reverse=True)
    assert any(c.vector_rank for c in found) and any(c.bm25_rank for c in found)
    assert any(c.vector_rank and c.bm25_rank for c in found)            # agreement between both searches


def test_relative_score_fusion_scores_are_bounded_and_rank_the_same_topic(chat):
    s = chat.settings
    old = s.hybrid_fusion
    try:
        s.hybrid_fusion = "relative"
        found = chat.retriever.retrieve("How do I revoke a certificate?", "products", PUBLIC, top_n=20)
        assert found and all(0.0 <= c.fused_score <= 1.25 for c in found)  # weights sum to 1 (x1.25 entity boost)
        assert any("revok" in (c.meta.get("section", "") + c.chunk.text).lower() for c in found[:5])
    finally:
        s.hybrid_fusion = old


def test_multi_query_fusion_keeps_hits_from_every_phrasing(chat):
    r = chat.retriever
    a = r.retrieve("revoke certificate", "products", PUBLIC, top_n=30)
    b = r.retrieve("cancel an issued credential", "products", PUBLIC, top_n=30)
    fused = r.retrieve_multi(["revoke certificate", "cancel an issued credential"], "products", PUBLIC, top_n=30)
    ids = {c.id for c in fused}
    assert a[0].id in ids and b[0].id in ids                            # each phrasing's best hit survives
    assert max(c.debug.get("multi_query_hits", 0) for c in fused) == 2  # chunks found by both get the boost
    both = [c for c in fused if c.debug.get("multi_query_hits") == 2]
    once = [c for c in fused if c.debug.get("multi_query_hits") == 1]
    if both and once:
        assert max(c.fused_score for c in both) >= max(c.fused_score for c in once)


def test_misspelt_question_retrieves_the_same_passages_as_the_clean_one(chat, public_user):
    clean = chat.answer("How do I revoke a certificate?", public_user, persist=False, allowed_kbs=["products"])
    messy = chat.answer("hw to revok a certificat", public_user, persist=False, allowed_kbs=["products"])
    clean_ids = {r["id"] for r in clean.retrieved[:5]}
    messy_ids = {r["id"] for r in messy.retrieved[:5]}
    assert clean_ids & messy_ids


def test_multi_query_can_be_switched_off(chat, public_user):
    s = chat.settings
    s.multi_query_retrieval = False
    try:
        resp = chat.answer("How do I revoke a certificate?", public_user, persist=False, allowed_kbs=["products"])
        assert resp.answer_type not in ("refusal", "out_of_scope")
    finally:
        s.multi_query_retrieval = True


def test_truncated_typos_reach_the_right_section_after_reranking(chat, public_user):
    resp = chat.answer("hw to revok a certificat", public_user, persist=False, allowed_kbs=["products"])
    assert resp.answer_type not in ("refusal", "out_of_scope")
    assert "revoke" in (resp.retrieved[0].get("section") or "").lower() and resp.retrieved[0]["score"] > 0.5
