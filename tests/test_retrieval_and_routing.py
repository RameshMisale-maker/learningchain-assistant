"""Hybrid retrieval, reranking, approved-answer matching, routing, scope guard and document-level access control."""
from __future__ import annotations

import pytest


def test_hybrid_finds_relevant_help_article(chat, public_user):
    from backend.routing.router import RouteDecision

    q = "How do I reset a forgotten password?"
    cands = chat._retrieve(chat.router.expand(q), RouteDecision(intent="kb_query", knowledge_bases=["products"]),
                           public_user.roles, chat.embedder.embed_query(q))
    ranked = chat.reranker.rerank(q, cands)
    assert any("Forget Your Password" in c.meta.get("section", "") for c in ranked[:5])


def test_golden_matcher_exact_and_paraphrase(chat, public_user):
    route = chat.router.route("What is LearningChain?", public_user.roles)
    exact = chat._best_golden("What is LearningChain?", route, public_user.roles,
                              chat.embedder.embed_query("What is LearningChain?"))
    assert exact and exact.qid == "F01" and exact.score >= 0.9
    q = "Which file formats are supported?"
    para = chat._best_golden(q, chat.router.route(q, public_user.roles), public_user.roles, chat.embedder.embed_query(q))
    assert para and para.qid == "F11"


@pytest.mark.parametrize("question,kb", [
    ("How do I verify a student certificate?", "products"),
    ("What is the difference between LearningChain and DigiLocker?", "products"),
    ("Where is the Sapphirus head office?", "company"),
    ("Do you provide recruitment services?", "company"),
])
def test_router_examples(chat, public_user, question, kb):
    route = chat.router.route(chat.router.expand(question), public_user.roles)
    assert kb in route.knowledge_bases, route.to_dict()


@pytest.mark.parametrize("question", ["What is the work from home policy?", "What is my salary?",
                                      "How do I cook biryani?", "What is the price of a laptop?"])
def test_router_declines_out_of_scope(chat, public_user, question):
    assert chat.router.route(chat.router.expand(question), public_user.roles).intent == "out_of_scope"


def test_document_acl_hides_restricted_content(workspace, pipeline, public_user, employee):
    from backend.retrieval.kb_index import get_index_manager
    from backend.services.chat_service import ChatService

    path = workspace / "knowledge_bases/products/Internal_Roadmap_Note.md"
    path.write_text("# LearningChain internal roadmap\n\n## Next quarter\n\nLearningChain will add batch revocation "
                    "approvals for registrar teams in the next quarter (internal, not public).\n", encoding="utf-8")
    res = pipeline.ingest_file("products", path, roles=["employee"])
    assert res.status == "added"
    get_index_manager().invalidate("products")
    svc = ChatService(llm=None)
    q = "What will LearningChain add for registrar teams next quarter?"
    staff = svc.answer(q, employee, persist=False)
    guest = svc.answer(q, public_user, persist=False)
    assert "batch revocation" in staff.answer
    assert "batch revocation" not in guest.answer and not any("Roadmap" in c.doc_name for c in guest.citations)
    pipeline.delete_document(res.doc_id, archive_file=True)
    get_index_manager().invalidate("products")
