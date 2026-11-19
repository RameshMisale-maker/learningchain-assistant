"""Conversational intelligence layer - understands WHAT KIND of message arrived before any retrieval runs.

Before this layer every input was treated as a knowledge question: "Ho", "Wht", "What is", "xyz123" or
"Tell me more" went through routing/retrieval and came back as "outside what I can help with" or "insufficient
information". Now each message is classified first:

    GREETING                  hi, hello, good morning (+ typos such as "helo")       -> direct reply, no retrieval
    SMALL_TALK                how are you, thanks, ok, bye                           -> direct reply, no retrieval
    TYPO_OR_INCOMPLETE_INPUT  ho, wht, hw, "what is", "how to", "tell me"             -> smart clarification
    CLARIFICATION_REQUIRED    what?, which one?, when?                               -> clarification (history-aware)
    UNKNOWN_OR_GIBBERISH      xyz123, @@@###, asdfghjkl                              -> ask to rephrase
    OUT_OF_SCOPE_QUERY        tell me a joke, weather ...                            -> polite scope reply
    FOLLOW_UP_QUERY           tell me more, explain the second point, its features   -> context resolution -> RAG
    KNOWLEDGE_QUERY           everything else                                        -> existing RAG pipeline

Design rules
* Rules first: greetings, typos, gibberish and obvious small talk never cost an LLM call.
* The LLM (if configured) is consulted only for short messages the rules cannot place (unknown words).
* The user's original text is never changed - suggestions are offered, not applied - and knowledge queries reach
  retrieval exactly as typed. Safety checks (injection, restricted topics) run BEFORE this layer, so nothing here
  can bypass them, and no reply in this module states product facts (no invented information).
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Callable

from backend.common.text_utils import STOPWORDS
from backend.routing.router import FOLLOW_UP_RE, GREETING_RE

logger = logging.getLogger(__name__)

GREETING = "GREETING"
SMALL_TALK = "SMALL_TALK"
TYPO_OR_INCOMPLETE = "TYPO_OR_INCOMPLETE_INPUT"
CLARIFICATION = "CLARIFICATION_REQUIRED"
KNOWLEDGE = "KNOWLEDGE_QUERY"
FOLLOW_UP = "FOLLOW_UP_QUERY"
OUT_OF_SCOPE = "OUT_OF_SCOPE_QUERY"
GIBBERISH = "UNKNOWN_OR_GIBBERISH"
ACCOUNT = "ACCOUNT_SPECIFIC"          # this user's own payment / receipt / verification status - no lookup tools
ESCALATION = "ESCALATION"            # complaint, refund dispute, legal or harassment issue -> human support
INTENTS = (GREETING, SMALL_TALK, TYPO_OR_INCOMPLETE, CLARIFICATION, KNOWLEDGE, FOLLOW_UP, OUT_OF_SCOPE, GIBBERISH, ACCOUNT,
           ESCALATION)

EXAMPLES = ('*"What is LearningChain?"*, *"How do I verify a certificate?"* or '
            '*"What are the key features of LearningChain?"*')
INCOMPLETE_REPLY = "Sure - what would you like to know? Please share your full question, for example: *{hint}*"
REPHRASE_TEXT = ("Sorry, I couldn't understand that. Could you rephrase your question? You can ask me about "
                 f"LearningChain - for example {EXAMPLES}")

GREETING_ONLY_RE = re.compile(
    r"^\s*((hi+|hello+|hey+|hiya|helo+|hlo+|hllo|hellow|helllo|hai|hy|hie|yo|greetings|namaste|namaskar)"
    r"(\s+(there|team|sapphirus|bot|all))?|(good|gud|gd)\s*(morning|afternoon|evening|day)|gm)\s*[!.?]*\s*$",
    re.IGNORECASE)
AFFIRM_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ya|of course|go ahead)\s*[!.?]*\s*$", re.IGNORECASE)

QUESTION_WORDS = ["what", "how", "who", "why", "when", "where", "which", "can", "could", "does", "do", "is", "are",
                  "tell", "explain", "show", "give", "list", "describe"]
# Frequent shorthand / misspellings of the words a question starts with. None = genuinely ambiguous.
TYPOS: dict[str, str | None] = {
    "ho": "how", "hw": "how", "hwo": "how", "howw": "how", "hiw": "how",
    "wht": "what", "wat": "what", "wut": "what", "whta": "what", "waht": "what", "wha": "what", "whatt": "what",
    "whn": "when", "wen": "when", "whr": "where", "wher": "where", "were": None, "wy": "why", "y": "why",
    "hu": "who", "whu": "who", "wich": "which", "whch": "which", "cn": "can", "cud": "could", "tel": "tell",
    "tll": "tell", "expln": "explain", "explan": "explain", "m": "me", "wh": None, "w": None,
}
AMBIGUOUS_STARTS = ["What", "Where", "When", "Who", "Why", "Which"]
INCOMPLETE_HINTS = {
    "what": '"What is LearningChain?"', "how": '"How do I verify a certificate?"', "who": '"Who can issue certificates?"',
    "why": '"Why use blockchain certificates?"', "when": '"When is a certificate added to the blockchain?"',
    "where": '"Where does LearningChain store my data?"', "which": '"Which file formats can I upload?"',
    "can": '"Can I revoke a certificate?"', "tell": '"Tell me about the features of LearningChain"',
    "explain": '"Explain how certificate verification works"',
}
CLARIFY_ONLY_RE = re.compile(
    r"^\s*(what|when|where|who|why|which|which\s+one|what\s+do\s+you\s+mean|meaning|huh|sorry|pardon|come\s+again|"
    r"i\s+don'?t\s+understand|not\s+clear)\s*\?+\s*$|^\s*(huh|pardon|come again)\s*[!.]*\s*$", re.IGNORECASE)
ELABORATE_RE = re.compile(
    r"^\s*(please\s+)?((can|could|would)\s+you\s+)?(please\s+)?"
    r"(tell\s+me\s+more|more(\s+(details?|info(rmation)?|on\s+(that|this|it)))?|explain(\s+(that|this|it|them|"
    r"further|more|in\s+detail))?|elaborate(\s+(on\s+)?(that|this|it))?|go\s+on|continue|keep\s+going|"
    r"and\s+(then|more)|expand\s+(on\s+)?(that|this|it)|give\s+(me\s+)?(more\s+)?(details|examples?)|"
    r"what\s+else|anything\s+else\s+about\s+(it|that|this)|"
    r"(explain|describe|tell\s+me\s+about|what\s+about|what\s+is|more\s+on)\s+the\s+"
    r"(first|second|third|fourth|fifth|last|next|previous|\d+(st|nd|rd|th))\s*(one|point|step|item|feature|option)?|"
    r"(the\s+)?(first|second|third|fourth|fifth|last)\s+(one|point|step|item|feature|option))"
    r"(\s+please)?\s*[?.!]*\s*$", re.IGNORECASE)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
            "5th": 5, "last": -1}
OUT_OF_SCOPE_RE = re.compile(
    r"\b(tell\s+(me\s+)?a\s+joke|jokes?\b|weather|temperature\s+(today|outside)|cricket|football|ipl|match\s+score|"
    r"recipe|cook(ing)?\b|movie|film|song|lyrics|poem|horoscope|astrology|news\s+today|latest\s+news|"
    r"write\s+(me\s+)?(a\s+)?(code|program|essay|story|poem)|play\s+a\s+game|translate\b|capital\s+of|"
    r"president\s+of|prime\s+minister)\b", re.IGNORECASE)
SUPPORT_CONTACT = "email **info@sapphirus.in** or call **+91 96531 57225**"
ACCOUNT_RE = re.compile(
    r"\b(has|have|did|is|was|were|are)\s+(my|our)\s+(payment|transaction|refund|order|subscription|verification|"
    r"certificate|application|receipt|invoice)s?\b|"
    r"\b(status|update)\s+of\s+(my|our)\s+(payment|transaction|refund|order|verification|application)\b|"
    r"\b(my|our)\s+(payment|transaction|refund|order|verification|application)\s+(status|is pending|is not|isn't|"
    r"failed|went through|gone through|deducted|stuck)\b|"
    r"\b(check|show|send|get|download)\s+(me\s+)?(my|our)\s+(receipt|invoice|payment status|verification status|"
    r"transaction history)\b|\bmoney (was |got )?deducted\b", re.IGNORECASE)
ESCALATION_RE = re.compile(
    r"\b(complain\w*|harass\w*|abusive|legal action|legal notice|lawyer|lawsuit|sue (you|sapphirus)|consumer (court|forum)|"
    r"(want|need|demand|give me)\s+(a\s+|my\s+)?(refund|money back)|refund\s+(is\s+)?(not|never|still)\b|"
    r"(worst|pathetic|useless|terrible)\s+(service|support|app|experience))", re.IGNORECASE)
ACCOUNT_REPLY = ("I can't see your personal account details (payments, receipts or verification status) in this chat. "
                 "Please log in to your account to check it, or contact our support team: " + SUPPORT_CONTACT + ".")
ESCALATION_REPLY = ("I'm sorry about this. Our support team will look into it personally - please " + SUPPORT_CONTACT +
                    " and share the details there.")
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890")
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class MessageIntent:
    intent: str
    reply: str = ""               # direct reply for intents that do not use retrieval
    suggestion: str = ""          # e.g. "How" for "Ho" - offered to the user, never applied to the query
    uses_history: bool = False    # conversation history was needed to understand / resolve the message
    method: str = "rules"         # rules | llm
    reason: str = ""

    @property
    def needs_rag(self) -> bool:
        return self.intent in (KNOWLEDGE, FOLLOW_UP)

    def to_dict(self) -> dict:
        return {"conversation_intent": self.intent, "intent_method": self.method, "uses_history": self.uses_history,
                "suggestion": self.suggestion, "intent_reason": self.reason}


# ------------------------------------------------------------------------------------------------ helpers
def is_small_talk(text: str) -> bool:
    """Greeting / thanks / bye / yes - handled by fixed replies, so no normalization or LLM call is needed."""
    return bool(GREETING_ONLY_RE.match(text or "") or GREETING_RE.match(text or "") or AFFIRM_RE.match(text or ""))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def last_assistant_answer(history: list[dict]) -> str:
    """The previous assistant answer that carried content (not a greeting or a clarification)."""
    for h in reversed(history or []):
        if h.get("role") == "assistant" and h.get("answer_type", "generated") not in ("greeting", "closing",
                                                                                     "clarification", "blocked"):
            return str(h.get("content") or "")
    return ""


def answer_items(answer: str, limit: int = 6) -> list[str]:
    """Points of a previous answer: bold headings, else bullet / numbered list items (citation markers removed)."""
    items = [m.strip(" :-") for m in re.findall(r"\*\*(.+?)\*\*", answer)]
    items = [i for i in items if 2 < len(i) < 80 and not i.lower().startswith(("note", "key features", "summary"))]
    if len(items) < 2:
        items = [re.sub(r"\*\*|\[S\d+\]", "", m).strip(" :-")
                 for m in re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", answer, flags=re.MULTILINE)]
    return [re.sub(r"\[S\d+\]", "", i).strip() for i in items if i][:limit]


def ordinal_item(message: str, answer: str) -> str:
    """'explain the second point' -> the second point of the previous answer ('' when not found)."""
    m = re.search(r"\b(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b", message.lower())
    items = answer_items(answer, limit=10)
    if not m or not items:
        return ""
    n = ORDINALS[m.group(1)]
    if n == -1:
        return items[-1]
    return items[n - 1] if n <= len(items) else ""


def _typo_target(word: str) -> tuple[str | None, bool]:
    """(intended question word or None, matched?) for a short misspelt question word."""
    if word in TYPOS:
        return TYPOS[word], True
    if word in QUESTION_WORDS or len(word) < 3 or len(word) > 6:
        return None, False
    close = difflib.get_close_matches(word, QUESTION_WORDS, n=2, cutoff=0.75)
    if len(close) == 1:
        return close[0], True
    return None, bool(close)


def _looks_gibberish(word: str, known: Callable[[str], bool]) -> bool:
    if known(word):
        return False
    letters = re.sub(r"[^a-z]", "", word)
    if not letters:
        return True                                           # digits / symbols only
    if letters != word and any(ch.isdigit() for ch in word):
        return True                                           # xyz123, a1b2c3
    if any(word[i:i + 4] in row for row in _KEYBOARD_ROWS for i in range(len(word) - 3)):
        return True                                           # asdf, qwerty, hjkl runs
    if len(word) >= 4 and not re.search(r"[aeiouy]", word):
        return True                                           # no vowels: "zxcvb", "bcdfg"
    if re.search(r"(.)\1{3,}", word):
        return True                                           # "aaaaa"
    return False


def _clarify_reply(history: list[dict], message: str) -> tuple[str, bool]:
    prev = last_assistant_answer(history)
    if prev:
        items = answer_items(prev, limit=5)
        if "which" in message.lower() and items:
            opts = ", ".join(f'"{i}"' for i in items[:-1]) + f' or "{items[-1]}"' if len(items) > 1 else f'"{items[0]}"'
            return (f"Are you referring to one of the points we discussed earlier - {opts}? "
                    "Please tell me which one and I'll explain it."), True
        return ("Could you clarify what you'd like to know about my previous answer? For example, you can say "
                "*\"Tell me more\"* or *\"Explain the second point\"*."), True
    return f"Could you clarify what you'd like to know? I'll be happy to help - for example {EXAMPLES}", False


# ------------------------------------------------------------------------------------------------ classifier
def classify_message(text: str, history: list[dict] | None = None, known_word: Callable[[str], bool] | None = None,
                     llm=None) -> MessageIntent:
    """Classify one (already validated and safety-screened) user message. Never raises."""
    history = history or []
    t = (text or "").strip()
    words = _words(t)
    known = known_word or (lambda w: w in STOPWORDS or w in QUESTION_WORDS)
    has_history = bool(last_assistant_answer(history))

    # 1. greetings / small talk (incl. typos such as "helo", "gud morning") - no retrieval, no LLM
    if GREETING_ONLY_RE.match(t):
        return MessageIntent(GREETING, reason="greeting")
    if GREETING_RE.match(t) or AFFIRM_RE.match(t):
        return MessageIntent(SMALL_TALK, reason="small talk")

    # 2. no letters at all ("@@@###", "???", "123")
    if not re.search(r"[^\W\d_]", t):
        return MessageIntent(GIBBERISH, reply=REPHRASE_TEXT, reason="no words")
    if not words:                       # written only in another script (e.g. Devanagari): a question to answer
        return MessageIntent(KNOWLEDGE, reason="non-Latin script")

    # 2b. this user's own account (no lookup tools) and complaints / disputes (human support) - fixed replies
    if ESCALATION_RE.search(t):
        return MessageIntent(ESCALATION, reply=ESCALATION_REPLY, reason="complaint / dispute -> support")
    if ACCOUNT_RE.search(t):
        return MessageIntent(ACCOUNT, reply=ACCOUNT_REPLY, reason="personal account data")

    # 3. requests to continue / elaborate - resolved from the conversation, or clarified when there is none
    if ELABORATE_RE.match(t):
        if has_history:
            return MessageIntent(FOLLOW_UP, uses_history=True, reason="elaboration request")
        return MessageIntent(CLARIFICATION, reason="elaboration without a previous topic",
                             reply="Sure - which topic would you like me to explain? You can ask me, for example, "
                                   f"{EXAMPLES}")

    # 4. a bare interrogative: "What?", "Which one?", "When?"
    if CLARIFY_ONLY_RE.match(t):
        reply, used = _clarify_reply(history, t)
        return MessageIntent(CLARIFICATION, reply=reply, uses_history=used, reason="bare interrogative")

    # 5. a lone (misspelt) question word: "Ho", "Wht", "Hw", "Wh" - the intent is genuinely unclear, so ask for the
    #    question. The user's spelling is never mentioned or corrected in replies: multi-word messages with typos
    #    are normalized and answered (backend/routing/normalize.py).
    if 1 <= len(words) <= 2 and not t.endswith("?") or len(words) == 1:
        first, matched = _typo_target(words[0]) if words else (None, False)
        rest_ok = all(w in STOPWORDS or w in TYPOS for w in words[1:])
        if matched and rest_ok and (first is None or first in QUESTION_WORDS):
            if first is None:
                opts = ", ".join(f'"{w}"' for w in AMBIGUOUS_STARTS[:-1]) + f' or "{AMBIGUOUS_STARTS[-1]}"'
                return MessageIntent(TYPO_OR_INCOMPLETE, reason=f"lone word: {words[0]}", reply=INCOMPLETE_REPLY.format(
                    hint=INCOMPLETE_HINTS["what"]))
            return MessageIntent(TYPO_OR_INCOMPLETE, suggestion=first.capitalize(), reason=f"lone word: {words[0]}",
                                 reply=INCOMPLETE_REPLY.format(hint=INCOMPLETE_HINTS.get(first, INCOMPLETE_HINTS["what"])))

    # 6. only function words / truncated words: "What is", "How to", "Tell me", "Can you", "How t"
    if words and len(words) <= 5 and all(w in STOPWORDS or w in QUESTION_WORDS or w in TYPOS
                                                  or len(w) == 1 for w in words):
        starter = next((TYPOS.get(w) or w for w in words if (TYPOS.get(w) or w) in INCOMPLETE_HINTS), "")
        hint = INCOMPLETE_HINTS.get(starter, INCOMPLETE_HINTS["what"])
        return MessageIntent(TYPO_OR_INCOMPLETE, reason="incomplete question", reply=INCOMPLETE_REPLY.format(hint=hint))

    # 7. random characters: "xyz123", "asdfghjkl"
    if len(words) <= 3 and all(_looks_gibberish(w, known) for w in words):
        return MessageIntent(GIBBERISH, reply=REPHRASE_TEXT, reason="gibberish")

    # 8. clearly unrelated requests (the router's own out-of-scope checks still apply to everything else)
    if OUT_OF_SCOPE_RE.search(t):
        return MessageIntent(OUT_OF_SCOPE, reason="unrelated request")

    # 9. short message referring back to the conversation: "What about its features?", "And them?"
    if has_history and len(words) <= 8 and FOLLOW_UP_RE.search(t):
        return MessageIntent(FOLLOW_UP, uses_history=True, reason="reference to the previous turn")

    # 10. short message with words the rules can't place -> ask the LLM (if any); otherwise treat as a question
    # (a question word followed by a subject - "What is LearningChain?" - is already a well-formed question)
    unknown = [w for w in words if not known(w)]
    question_shaped = words[0] in QUESTION_WORDS and len(words) >= 2
    if llm is not None and len(words) <= 3 and unknown and not question_shaped:
        judged = _llm_classify(llm, t, history)
        if judged is not None:
            return judged
    return MessageIntent(KNOWLEDGE, reason="knowledge question")


def _llm_classify(llm, text: str, history: list[dict]) -> MessageIntent | None:
    """LLM fallback for ambiguous short messages. Only the label (and a one-word suggestion) is used - the reply
    shown to the user is always one of this module's fixed templates, so the LLM cannot invent information."""
    from backend.generation.prompt_builder import format_history, load_prompt

    try:
        data, _ = llm.complete_json("You classify chatbot messages. Output JSON only.",
                                    load_prompt("intent_prompt.txt").format(history=format_history(history, 2),
                                                                            message=text), max_tokens=300)
    except Exception as exc:   # fail open: the message is handled as a normal question
        logger.warning("LLM intent classification failed: %s", exc)
        return None
    intent = str(data.get("intent", "")).strip().upper()
    if intent not in INTENTS:
        return None
    suggestion = str(data.get("suggestion") or "").strip()
    suggestion = suggestion if re.fullmatch(r"[A-Za-z][A-Za-z ]{0,30}", suggestion) else ""
    result = MessageIntent(intent, suggestion=suggestion, method="llm", reason="llm classification",
                           uses_history=intent == FOLLOW_UP)
    if intent == TYPO_OR_INCOMPLETE:
        if len(_words(text)) >= 2:          # a misspelt question is still a question: answer it, never ask to retype
            result.intent, result.reason = KNOWLEDGE, "llm: misspelt question treated as a question"
        else:
            result.reply = INCOMPLETE_REPLY.format(hint=INCOMPLETE_HINTS["what"])
    elif intent == CLARIFICATION:
        result.reply, result.uses_history = _clarify_reply(history, text)
    elif intent == GIBBERISH:
        result.reply = REPHRASE_TEXT
    elif intent == FOLLOW_UP and not last_assistant_answer(history):
        result.intent, result.uses_history = KNOWLEDGE, False
    return result
