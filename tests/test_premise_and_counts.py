"""24 Nov evaluation fixes: wrong-assumption corrections are not declined, and "how many" answers may only state a
count the sources state (the model must not count list items itself)."""
from types import SimpleNamespace

import pytest

from backend.generation.grounding import uncounted_numbers
from backend.services.chat_service import ChatService


def _refs(text):
    return [SimpleNamespace(text=text, metadata={})]


CLIENTS = _refs("Our clients: YCMOU, COEP, DY Patil University. Updated 12 June 2025. Over 50,000 certificates issued.")


@pytest.mark.parametrize("answer", [
    "LearningChain is currently used by **12 universities** [S3].",
    "There are **3** universities that use LearningChain: YCMOU, COEP and DY Patil.",
])
def test_counted_list_is_caught(answer):
    assert uncounted_numbers("How many universities currently use LearningChain?", answer, CLIENTS)


def test_stated_count_passes():
    assert uncounted_numbers("How many certificates were issued?", "Over **50,000 certificates** were issued [S1].",
                             CLIENTS) == []


def test_only_count_questions_are_checked():
    assert uncounted_numbers("How do I verify a certificate?", "1. Open the URL\n2. Pay 500 rupees", CLIENTS) == []


def test_step_numbering_is_not_a_count():
    assert uncounted_numbers("How many steps does verification have?", "1. Enter details\n2. Upload\n3. Pay",
                             _refs("Complete the 4-step verification process.")) == []


@pytest.mark.parametrize("term,answer", [
    ("aws", "learningchain stores data on **microsoft azure**, not aws [s1]."),
    ("word", "only pdf files can be uploaded, not word documents [s2]."),
    ("bangalore", "sapphirus has offices in hyderabad, bhopal, new delhi, mohali, pune and atlanta, not in bangalore [s1]."),
    ("aws", "data is stored on microsoft azure rather than aws [s1]."),
])
def test_correction_of_a_wrong_assumption_is_allowed(term, answer):
    assert ChatService._only_corrected(term, answer)


@pytest.mark.parametrize("term,answer", [
    ("ios", "learningchain does not have an ios app [s1]."),          # an undocumented negative claim
    ("aws", "learningchain uses aws [s1]."),                           # an undocumented positive claim
    ("aws", "learningchain stores data on azure, not aws."),           # no source tag in the sentence
    ("aws", "azure, not aws [s1]. aws is also used for backups [s1]."),  # a second, asserting mention
])
def test_undocumented_claims_are_still_caught(term, answer):
    assert not ChatService._only_corrected(term, answer)


def test_prompt_tells_the_model_to_correct_wrong_assumptions():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "wrong assumption" in text and "not AWS" in text
