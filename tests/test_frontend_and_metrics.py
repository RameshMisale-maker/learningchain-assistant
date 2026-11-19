"""Website widget endpoints, page security headers, dashboard APIs and claim-level evaluation helpers."""
from __future__ import annotations

import json

ADMIN = {"Authorization": "Bearer admin-token-for-tests-0001"}
PUBLIC = {"Authorization": "Bearer public-token-for-tests-0001"}


def _client():
    from fastapi.testclient import TestClient

    from backend.main import app
    return TestClient(app)


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        ev = next(l[7:] for l in block.split("\n") if l.startswith("event: "))
        data = json.loads(next(l[6:] for l in block.split("\n") if l.startswith("data: ")))
        out.append((ev, data))
    return out


def test_stream_endpoint_emits_meta_deltas_sources_done(workspace):
    with _client() as c:
        r = c.post("/api/v1/chat/stream", json={"message": "How do I revoke a certificate?", "assistant": "products"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        evs = _events(r.text)
    kinds = [e for e, _ in evs]
    assert kinds[0] == "meta" and kinds[-2:] == ["sources", "done"] and kinds.count("delta") > 3
    text = "".join(d["text"] for e, d in evs if e == "delta")
    assert "Revoke" in text
    sources = dict(evs)["sources"]["citations"]
    assert sources and sources[0]["url"].startswith("https://")
    assert dict(evs)["done"]["message_id"]


def test_stream_scoped_assistant_refuses_other_kb(workspace):
    with _client() as c:
        r = c.post("/api/v1/chat/stream", json={"message": "Who is the CEO of Sapphirus?", "assistant": "products"})
    evs = _events(r.text)
    assert dict(evs)["meta"]["answer_type"] == "out_of_scope"
    assert dict(evs)["sources"]["citations"] == []


def test_widget_and_admin_pages_security_headers(workspace):
    with _client() as c:
        chat = c.get("/chat")
        assert chat.status_code == 200 and "/api/v1/ui" in chat.text     # texts come from the product config
        ui = c.get("/api/v1/ui").json()
        assert ui["title"] and ui["welcome"]
        csp = chat.headers["content-security-policy"]
        assert "frame-ancestors 'self' https://sapphirus.in" in csp and "connect-src 'self'" in csp
        assert "x-frame-options" not in {k.lower() for k in chat.headers}
        admin = c.get("/admin")
        assert admin.headers["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in admin.headers["content-security-policy"]
        js = c.get("/widget.js")
        assert js.status_code == 200 and "iframe" in js.text and "api_key" not in js.text.lower()
        assert "sk-ant" not in chat.text and "ANTHROPIC" not in chat.text


def test_metrics_summary_requires_admin_and_aggregates(workspace):
    with _client() as c:
        for q in ("What are the steps to revoke an issued certificate?", "What is the capital of France?"):
            c.post("/api/v1/chat", json={"message": q, "assistant": "all"}, headers=PUBLIC)
        assert c.get("/api/v1/admin/metrics/summary", headers=PUBLIC).status_code == 403
        r = c.get("/api/v1/admin/metrics/summary?channel=api", headers=ADMIN)
        assert r.status_code == 200
        o = r.json()["overall"]
        assert o["requests"] >= 2 and o["successful"] >= 2
        assert o["end_to_end_ms"]["p50"] > 0 and o["retrieval_ms"]["n"] >= 1
        assert "out_of_scope" in o["answer_types"]
        ts = c.get("/api/v1/admin/metrics/timeseries?bucket=day", headers=ADMIN).json()
        assert ts["series"] and ts["series"][-1]["requests"] >= 2


def test_evaluation_history_endpoint(workspace):
    from backend.services.metrics_service import record_evaluation_run

    rid = record_evaluation_run("unit test run", "rag", {"llm": "x"}, {"by_category": {"overall": {"correctness": 0.9}},
                                                                     "confusion": {}})
    with _client() as c:
        runs = c.get("/api/v1/admin/evaluations", headers=ADMIN).json()["runs"]
    assert any(r["id"] == rid for r in runs)


def test_claim_split_and_lexical_judge():
    from backend.evaluation.claims import LexicalJudge, evaluate_answer, split_claims

    answer = ("**How do I Revoke the Certificates** [S1]\n\n- Click on Reports Module and then Generated Certificates "
              "tab.\n- After the certificate is Revoked we cannot do the verification.\n\n"
              "Full guide: https://example.com/a · Video walkthrough: https://v.com/1")
    claims = split_claims(answer)
    assert len(claims) == 2
    ctx = ["To Revoke the certificate, Click on Reports Module and then click on Generated Certificates tab. "
           "After the certificate is Revoked we cannot do the verification."]
    rep = evaluate_answer(answer + "\n\nRevoked certificates are reissued after 30 days automatically.", ctx,
                          LexicalJudge(), allowed_urls={"https://example.com/a"}, valid_citation_ids={"S1"})
    assert len(rep.unsupported) == 1 and "30 days" in rep.unsupported[0].claim
    assert rep.fabricated_urls == ["https://v.com/1"]
