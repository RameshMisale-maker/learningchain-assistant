"""Conversation ending and rating: closing intent, per-session state, structured LLM reply, rating flow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.generation.llm_client import LLMClient, LLMResult
from backend.generation.structured import parse_reply

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = {"Authorization": "Bearer public-token-for-tests-0001"}


class CountingLLM(LLMClient):
    """Answers every prompt with a structured JSON reply and counts the calls."""
    provider = "openai"

    def __init__(self, settings, reply="To verify a certificate, scan its QR code or open the Verification URL [S1].",
                 ui_action="none"):
        super().__init__(settings)
        self.calls, self.reply, self.ui_action, self.model = 0, reply, ui_action, "counting"

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        self.calls += 1
        if "Classify" in user or "clean up user messages" in user or "Return ONLY this JSON" in user:
            return LLMResult(text="{}", model=self.model, provider="openai")
        return LLMResult(text=json.dumps({"reply": self.reply, "ui_action": self.ui_action,
                                          "conversation_state": "active"}),
                         prompt_tokens=10, completion_tokens=5, model=self.model, provider="openai")


@pytest.fixture()
def service(workspace):
    from backend.services.chat_service import ChatService
    from config.settings import get_settings
    llm = CountingLLM(get_settings())
    return ChatService(llm=llm), llm


def _messages(session_id):
    from sqlalchemy import select

    from backend.db.models import ChatMessage
    from backend.db.session import db_session
    with db_session() as s:
        return len(s.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all())


# ------------------------------------------------------------------------------------------------ the requested flow
def test_goodbye_rating_ack_then_new_question(service, public_user):
    from backend.db import repository

    chat, llm = service
    first = chat.answer("How do I change my password?", public_user)       # needs the LLM (no approved answer)
    sid = first.session_id

    bye = chat.answer("no thanks", public_user, session_id=sid)
    assert bye.answer_type == "closing" and bye.ui_action == "show_rating" and bye.conversation_state == "closing"
    assert "rate" not in bye.answer.lower() and bye.answer                       # a goodbye, no "rate below" text
    assert repository.get_conversation_state(sid) == "closing"

    calls, stored = llm.calls, _messages(sid)
    ack = chat.answer("ok", public_user, session_id=sid)
    assert llm.calls == calls                                                    # no LLM call
    assert _messages(sid) == stored and ack.message_id is None                   # no new message
    assert ack.answer_type == "closed" and ack.ui_action == "show_rating"        # rating stays visible
    assert repository.get_conversation_state(sid) == "closed"
    for text in ("Thanks!", "THANK YOU", "\U0001F44D", "done", "bye"):           # still closed, still no LLM / message
        assert chat.answer(text, public_user, session_id=sid).answer_type == "closed"
    assert llm.calls == calls and _messages(sid) == stored

    again = chat.answer("how do I change my password?", public_user, session_id=sid)
    assert again.conversation_state == "active" and again.answer_type not in ("closed", "closing")
    assert again.ui_action == "none" and again.answer and llm.calls > calls
    assert repository.get_conversation_state(sid) == "active"


def test_greeting_with_introduction_is_not_frustration(service, public_user):
    from backend.routing import intent as convo

    msg = convo.classify_message("okay hello chatbot, Ramesh here", [])
    assert msg.intent == convo.GREETING
    for text in ("hi, this is Priya", "hello bot I am Ravi", "Hello! Myself Anita"):
        assert convo.classify_message(text, []).intent == convo.GREETING, text
    chat, _ = service
    resp = chat.answer("okay hello chatbot, Ramesh here", public_user, persist=False)
    assert resp.answer_type == "greeting" and resp.route.get("conversation_intent") == convo.GREETING
    assert convo.classify_message("hello, how do I log in?", []).intent != convo.GREETING   # a question stays one


def test_closing_and_acknowledgement_detection():
    from backend.routing import intent as convo

    for t in ("no thanks", "No, thank you", "that's all", "nothing else", "bye", "I'm done"):
        assert convo.classify_message(t, []).intent == convo.CLOSING, t
    for t in ("ok", "OKAY", "okkk", "thanks", "Thank you!", "bye", "\U0001F44D", "\U0001F44D\U0001F3FD", "done",
              "ok thanks"):
        assert convo.is_acknowledgement(t), t
    for t in ("how do I verify a certificate?", "ok so how do I revoke one", "thanks, but what about fees?", ""):
        assert not convo.is_acknowledgement(t), t


def test_stateless_history_closing(service, public_user):
    """Open WebUI / evaluation send the history themselves: the goodbye in it closes the conversation."""
    chat, llm = service
    history = [{"role": "user", "content": "no thanks"},
               {"role": "assistant", "content": "Bye!", "answer_type": "closing"}]
    calls = llm.calls
    resp = chat.answer("ok", public_user, persist=False, history=history)
    assert resp.answer_type == "closed" and llm.calls == calls


# ------------------------------------------------------------------------------------------------ structured reply
def test_structured_reply_parsing_and_fallback():
    ok = parse_reply('{"reply": "Hi [S1]", "ui_action": "show_escalation", "conversation_state": "active"}')
    assert ok.parsed and ok.reply == "Hi [S1]" and ok.ui_action == "show_escalation"
    fenced = parse_reply('```json\n{"reply": "A", "ui_action": "show_login"}\n```')
    assert fenced.reply == "A" and fenced.ui_action == "show_login"
    bad_action = parse_reply('{"reply": "A", "ui_action": "explode", "conversation_state": "weird"}')
    assert bad_action.ui_action == "none" and bad_action.conversation_state == "active"
    plain = parse_reply("Just text with **markdown** [S1]")
    assert not plain.parsed and plain.reply.startswith("Just text") and plain.ui_action == "none"
    broken = parse_reply('{"reply": "Partial answer\\nline two", "ui_action": ')
    assert broken.reply == "Partial answer\nline two" and broken.ui_action == "none"
    assert parse_reply('{"answer": "no reply key"}').ui_action == "none"


def test_llm_cannot_force_the_rating(workspace, public_user):
    from backend.services.chat_service import ChatService
    from config.settings import get_settings
    llm = CountingLLM(get_settings(), ui_action="show_rating")
    resp = ChatService(llm=llm).answer("How do I verify a student certificate?", public_user, persist=False)
    assert resp.ui_action == "none"                                              # only the goodbye shows the rating


# ------------------------------------------------------------------------------------------------ rating API + UI
def test_session_rating_api_and_low_rating_review(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.services import review

    with TestClient(app) as c:
        r = c.post("/api/v1/chat", json={"message": "How do I revoke a certificate?"}, headers=PUBLIC).json()
        sid = r["session_id"]
        bye = c.post("/api/v1/chat", json={"message": "no thanks", "session_id": sid}, headers=PUBLIC).json()
        assert bye["ui_action"] == "show_rating" and bye["conversation_state"] == "closing" and bye["versions"]
        ok = c.post("/api/v1/chat", json={"message": "ok", "session_id": sid}, headers=PUBLIC).json()
        assert ok["answer_type"] == "closed" and ok["message_id"] is None

        good = c.post("/api/feedback", json={"session_id": sid, "rating": 5, "comment": "great",
                                             "product_id": "sapphirus", "versions": bye["versions"]}, headers=PUBLIC)
        assert good.status_code == 200 and good.json()["rating"] == "stars_5"
        low = c.post("/api/feedback", json={"session_id": sid, "rating": 2, "comment": "steps were wrong",
                                            "product_id": "sapphirus", "versions": bye["versions"]}, headers=PUBLIC)
        assert low.status_code == 200
        item = next(i for i in review.list_items(source="low_rating") if i["details"].get("session_id") == sid)
        assert item["details"]["stars"] == 2 and item["details"]["comment"] == "steps were wrong"
        assert item["question"] == "no thanks" or "revoke" in item["question"].lower()
        review.resolve(item["id"], "test", "dismiss")

        assert c.post("/api/feedback", json={"session_id": sid, "rating": 9}, headers=PUBLIC).status_code == 422
        assert c.post("/api/feedback", json={"session_id": "not-mine", "rating": 4}, headers=PUBLIC).status_code == 404
        assert c.post("/api/v1/feedback", json={"message_id": r["message_id"], "rating": "helpful"},
                      headers=PUBLIC).status_code == 200                        # old answer feedback still works

        stream = c.post("/api/v1/chat/stream", json={"message": "bye", "session_id": None}, headers=PUBLIC).text
        meta = json.loads(stream.split("event: meta\ndata: ")[1].split("\n")[0])
        assert meta["ui_action"] == "show_rating" and meta["product_id"]


def test_feedback_row_stores_stars_session_and_versions(workspace):
    from sqlalchemy import select

    from backend.db import repository
    from backend.db.models import Feedback
    from backend.db.session import db_session

    sid = repository.ensure_session(None, "fb_user")
    mid = repository.add_message(sid, "assistant", "hello")
    fid = repository.add_feedback(mid, "fb_user", "stars_4", "nice", session_id=sid, product_id="sapphirus", stars=4,
                                  details={"versions": {"prompt_version": "x"}})
    with db_session() as s:
        row = s.get(Feedback, fid)
        assert (row.stars, row.session_id, row.product_id, row.details["versions"]["prompt_version"]) == \
            (4, sid, "sapphirus", "x")


def test_no_rate_below_text_anywhere():
    for folder in ("prompts", "config/products", "backend/static"):
        for f in (ROOT / folder).rglob("*"):
            if f.is_file() and f.suffix in (".txt", ".yaml", ".html", ".js"):
                text = f.read_text(encoding="utf-8").lower()
                assert "rate this conversation below" not in text and "rate below" not in text, f


def test_migration_0008_round_trip(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from config.settings import BASE_DIR
    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "0008")
        assert "conversation_state" in {c["name"] for c in inspect(conn).get_columns("chat_sessions")}
        assert {"stars", "session_id", "details"} <= {c["name"] for c in inspect(conn).get_columns("feedback")}
        command.downgrade(cfg, "0007")
        assert "conversation_state" not in {c["name"] for c in inspect(conn).get_columns("chat_sessions")}


# ------------------------------------------------------------------------------------------------ evaluation findings
def test_pronoun_without_topic_asks_which_feature():
    from backend.routing import intent as convo

    for text in ("How do I configure it?", "What is this?", "Is it free?"):
        msg = convo.classify_message(text, [])
        assert msg.intent == convo.CLARIFICATION and "which feature" in msg.reply.lower(), text
    history = [{"role": "user", "content": "How do I add a certificate type?"},
               {"role": "assistant", "content": "Open Certificates > Add Certificate Type ..."}]
    assert convo.classify_message("How do I configure it?", history).intent == convo.FOLLOW_UP
    for text in ("How is it different from DigiLocker?", "How do I verify a certificate?"):
        assert convo.classify_message(text, []).intent != convo.CLARIFICATION, text


def test_account_reply_only_for_the_users_own_data():
    from backend.routing import intent as convo

    assert not convo.mentions_own_data("The verification result shows 'Invalid'.")
    for text in ("Has my payment gone through?", "our verification status is pending", "I paid twice"):
        assert convo.mentions_own_data(text), text


# ------------------------------------------------------------------------------------------------ greeting replies
def test_greeting_reply_reads_the_whole_message(chat, public_user):
    """'okay hello chatbot, Ramesh here' is a greeting - never the 'Glad that helps' acknowledgement."""
    history = [{"role": "user", "content": "How do I verify a certificate?"},
               {"role": "assistant", "content": "Click the verification URL ...", "answer_type": "golden_direct"}]
    resp = chat.answer("okay hello chatbot, Ramesh here", public_user, persist=False, history=history)
    assert resp.answer_type == "greeting" and "Ramesh" in resp.answer and "Glad that helps" not in resp.answer
    first = chat.answer("hello chatbot, Ramesh here", public_user, persist=False, history=[])
    assert first.answer.startswith("Hello Ramesh!")


def test_small_talk_replies_match_the_message(workspace):
    from backend.services.chat_service import small_talk_reply

    roles = {"public"}
    assert "Glad" in small_talk_reply("ok", roles, first=False)
    assert "welcome" in small_talk_reply("great, thanks", roles, first=False).lower()
    assert small_talk_reply("good morning, I am Ravi", roles, first=False).startswith("Good morning Ravi!")
    assert "Ravi" not in small_talk_reply("hi I'm back", roles, first=False)


def test_introduced_name():
    from backend.routing.intent import introduced_name

    assert [introduced_name(t) for t in ("Ramesh here", "this is Priya", "I am Ravi", "hi I'm back", "it's me",
                                         "okay hello chatbot")] == ["Ramesh", "Priya", "Ravi", "", "", ""]
