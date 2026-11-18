"""Multi-part question detection and rule-based splitting (backend/routing/decompose.py)."""
from backend.routing.decompose import decompose, heuristic_split, is_multi_part


def test_multi_part_questions_are_detected():
    assert is_multi_part("If I upload a certificate as a PDF, where is it stored and how will a verifier check it?")
    assert is_multi_part("Compared with DigiLocker, who controls the data in LearningChain and where does it live?")
    assert is_multi_part("How long is data valid on the blockchain, and does the institution have to pay again?")


def test_single_questions_are_not_split():
    assert not is_multi_part("How do I revoke a certificate?")
    assert not is_multi_part("What are the terms and conditions?")            # 'and' inside a noun phrase
    assert not is_multi_part("I can't remember my password, what do I do?")  # statement + one question
    assert not is_multi_part("Can we connect our system to LearningChain and push certificates?")


def test_heuristic_split_keeps_leading_condition():
    subs = heuristic_split("If I upload a certificate as a PDF, where is it stored and how will a verifier check it?")
    assert subs == ["If I upload a certificate as a PDF, where is it stored?",
                    "If I upload a certificate as a PDF, how will a verifier check it?"]


class _FakeLLM:
    def __init__(self, data):
        self.data = data

    def complete_json(self, system, user, max_tokens=400):
        assert "where is it stored" in user
        return self.data, None


def test_llm_split_is_used_and_falls_back():
    q = "If I upload a certificate as a PDF, where is it stored and how will a verifier check it?"
    llm = _FakeLLM({"sub_questions": ["Where is an uploaded certificate PDF stored?",
                                      "How does a verifier check an uploaded certificate?"]})
    assert decompose(q, llm) == ["Where is an uploaded certificate PDF stored?",
                                 "How does a verifier check an uploaded certificate?"]
    assert decompose(q, _FakeLLM({})) == heuristic_split(q)            # unusable LLM output -> rules
    assert decompose("How do I revoke a certificate?", llm) == []       # single question -> no LLM call needed
