"""Conversation memory: the whole chat reaches the model - recent exchanges in detail, older ones as a gist, all
within CHAT_HISTORY_MAX_CHARS."""
from __future__ import annotations

from backend.generation.prompt_builder import (LAST_ANSWER_CHARS, OLDER_MESSAGE_CHARS, RECENT_MESSAGE_CHARS,
                                               format_chat_history)


def _chat(n_exchanges, answer_len=300):
    out = []
    for i in range(1, n_exchanges + 1):
        out += [{"role": "user", "content": f"question number {i} about topic{i}"},
                {"role": "assistant", "content": f"answer {i} " + "x" * answer_len}]
    return out


def test_defaults_cover_a_whole_chat(workspace):
    from config.settings import get_settings

    s = get_settings()
    assert s.chat_history_messages >= 50 and s.chat_history_recent_turns == 4 and s.chat_history_max_chars >= 6000


def test_the_first_question_of_a_long_chat_is_remembered(workspace):
    text = format_chat_history(_chat(15))
    assert "question number 1 about topic1" in text and "question number 15 about topic15" in text
    assert "omitted" not in text


def test_recent_exchanges_in_detail_older_as_a_gist(workspace):
    lines = format_chat_history(_chat(10, answer_len=1500)).splitlines()
    answers = [ln for ln in lines if ln.startswith("ASSISTANT:")]
    assert len(answers[0]) <= len("ASSISTANT: ") + OLDER_MESSAGE_CHARS + 4                     # oldest: a gist
    assert len(answers[-2]) <= len("ASSISTANT: ") + RECENT_MESSAGE_CHARS + 4                   # recent: in detail
    assert len(answers[-2]) > len(answers[0])
    assert len(answers[-1]) > len(answers[-2]) and len(answers[-1]) <= len("ASSISTANT: ") + LAST_ANSWER_CHARS + 4


def test_a_very_long_chat_keeps_within_the_budget(workspace, monkeypatch):
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "chat_history_max_chars", 2000)
    text = format_chat_history(_chat(40))
    assert len(text) <= 2000 + 100
    assert text.splitlines()[0].startswith("(earlier in this chat:") and "question number 40" in text
    assert "question number 1 about" not in text                                   # the oldest are dropped first


def test_empty_history():
    assert format_chat_history([]) == "(none)"


def test_streamlit_and_api_use_the_setting():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "chat_history_messages" in (root / "streamlit_cloud" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "chat_history_messages" in (root / "backend" / "api" / "routes_openai.py").read_text(encoding="utf-8")
    assert "chat_history_messages" in (root / "backend" / "services" / "chat_service.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ first-question speed
def test_golden_warm_up_embeds_every_approved_wording_once(workspace):
    from backend.retrieval.golden_matcher import GoldenMatcher
    from backend.retrieval.kb_index import get_index_manager

    if not any(get_index_manager().get(kb).golden for kb in get_index_manager().kb_names()):
        import pytest
        pytest.skip("no approved answers in the test workspace")
    m = GoldenMatcher()
    assert m.warm() > 0
    assert m.warm() == 0                         # cached: a second warm-up (or the first user question) embeds nothing


def test_warm_up_is_on_by_default_and_off_in_tests(workspace, monkeypatch):
    from config.settings import Settings, get_settings

    assert get_settings().golden_warmup is False              # tests/conftest.py: no background threads in tests
    monkeypatch.delenv("GOLDEN_WARMUP", raising=False)
    assert Settings().golden_warmup is True
