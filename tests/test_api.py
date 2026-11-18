"""REST API tests using FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

ADMIN = {"Authorization": "Bearer admin-token-for-tests-0001"}
EMP = {"Authorization": "Bearer employee-token-for-tests-01"}


@pytest.fixture(scope="module")
def client(workspace):
    from backend.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_me_and_suggestions(client):
    assert client.get("/api/v1/me", headers=EMP).json()["roles"] == ["employee"]
    guest = client.get("/api/v1/me").json()
    assert guest["roles"] == ["public"] and set(guest["knowledge_bases"]) == {"company", "products"}
    assert client.get("/api/v1/suggestions", headers=EMP).json()["questions"]


def test_chat_and_feedback_flow(client):
    r = client.post("/api/v1/chat", json={"message": "What are the steps to revoke an issued certificate?"},
                    headers=EMP).json()
    assert r["knowledge_bases"] == ["products"] and r["citations"] and r["citations"][0]["url"].startswith("https://")
    assert set(r["route"]) <= {"intent", "knowledge_bases", "conversation_intent"}   # diagnostics hidden from non-admins
    fb = client.post("/api/v1/feedback", json={"message_id": r["message_id"], "rating": "helpful"}, headers=EMP)
    assert fb.status_code == 200
    other = client.post("/api/v1/feedback", json={"message_id": r["message_id"], "rating": "helpful"}, headers=ADMIN)
    assert other.status_code == 404                                    # cannot rate someone else's message
    hist = client.get(f"/api/v1/chat/sessions/{r['session_id']}/messages", headers=EMP).json()
    assert len(hist["messages"]) == 2
    assert client.delete(f"/api/v1/chat/sessions/{r['session_id']}", headers=EMP).json()["cleared"]


def test_admin_endpoints_require_admin(client):
    assert client.get("/api/v1/admin/knowledge-bases", headers=EMP).status_code == 403
    assert client.get("/api/v1/admin/knowledge-bases").status_code == 403
    assert client.get("/api/v1/admin/knowledge-bases", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_upload_query_delete(client):
    content = b"Sample notice: the Sapphirus innovation lab is open to employees on Thursdays from 2 PM."
    up = client.post("/api/v1/admin/documents", headers=ADMIN, data={"kb": "company", "roles": "employee", "wait": "true"},
                     files={"file": ("Innovation_Lab.txt", content)})
    assert up.status_code == 200 and up.json()["status"] == "added"
    ans = client.post("/api/v1/chat", json={"message": "When is the innovation lab open?"}, headers=EMP).json()
    assert "Thursdays" in ans["answer"]
    guest = client.post("/api/v1/chat", json={"message": "When is the innovation lab open?"}).json()
    assert "Thursdays" not in guest["answer"]                          # document-level ACL (employee only)
    bad = client.post("/api/v1/admin/documents", headers=ADMIN, data={"kb": "company", "wait": "true"},
                      files={"file": ("evil.pdf", b"MZ\x00binary")})
    assert bad.status_code == 400
    deleted = client.delete(f"/api/v1/admin/documents/{up.json()['doc_id']}?wait=true", headers=ADMIN)
    assert deleted.json()["status"] == "deleted"
    after = client.post("/api/v1/chat", json={"message": "When is the innovation lab open?"}, headers=EMP).json()
    assert "Thursdays" not in after["answer"]
    actions = {a["action"] for a in client.get("/api/v1/admin/audit-logs", headers=ADMIN).json()["items"]}
    assert {"document_upload", "document_delete"} <= actions
