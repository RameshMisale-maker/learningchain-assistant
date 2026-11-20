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
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable

from backend.common.text_utils import STOPWORDS
from backend.routing.router import FOLLOW_UP_RE, greeting_re, names_alternation

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
FRUSTRATION = "FRUSTRATION"          # anger, insults, swearing: stay calm, ask what went wrong
CLOSING = "CLOSING"                  # the user is ending the conversation ("no thanks", "that's all", "bye")
INTENTS = (GREETING, SMALL_TALK, TYPO_OR_INCOMPLETE, CLARIFICATION, KNOWLEDGE, FOLLOW_UP, OUT_OF_SCOPE, GIBBERISH, ACCOUNT,
           ESCALATION, FRUSTRATION, CLOSING)

# Ending the conversation: "no" after "anything else?", "that's all", "bye" -> goodbye + rating request.
CLOSING_RE = re.compile(r"^\s*(no|nope|nah|no\s*,?\s*thanks?|no\s*,?\s*thank\s*(you|u)|no\s+more(\s+questions?)?|"
                        r"nothing(\s+else)?|that'?s\s+(all|it)|that\s+is\s+(all|it)|i'?m\s+done|im\s+done|"
                        r"all\s+good|i'?m\s+good|bye+|goodbye|good\s*bye|see\s+(you|ya)|take\s+care)\s*[!.?]*\s*$",
                        re.IGNORECASE)
# Short acknowledgements after a goodbye: no LLM call, no new message, the rating stays visible.
ACK_WORDS = {"ok", "okay", "okk", "k", "kk", "thanks", "thank you", "thank u", "thanku", "thankyou", "thx", "ty",
             "bye", "bye bye", "goodbye", "done", "ok thanks", "okay thanks", "ok thank you", "okay thank you",
             "ok bye", "okay bye", "thanks bye", "cool", "great", "fine", "sure", "alright", "all right", "noted"}
_ACK_EMOJI = ("👍", "🙏", "😊", "🙂", "✅", "👌")   # thumbs up, folded hands, ...
# "okay hello chatbot, Ramesh here", "hi, this is Ramesh", "hello bot I am Priya" - a greeting with an introduction
_GREETING_INTRO = (
    r"^\s*(?:(?:ok(?:ay)?|so|well|hi+|hello+|hey+|helo|hai|namaste|good\s+(?:morning|afternoon|evening))[\s,!.]*)+"
    r"(?:(?:there|team|all|chatbot|chat\s*bot|bot|assistant|{names})[\s,!.]*)?"
    r"(?:(?:i\s*am|i'?m|im|this\s+is|it'?s|my\s+name\s+is|myself)\s+[a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*)?"
    r"|[a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*)?\s+(?:here|this\s+side))?[\s,!.]*$")
_GREETING_WORD = re.compile(r"\b(hi+|hello+|hey+|helo|hai|namaste|good\s+(morning|afternoon|evening))\b", re.IGNORECASE)


_OWN_DATA_RE = re.compile(r"\b(my|mine|our|ours|i|i'm|i've|me|we|us)\b", re.IGNORECASE)
_PRONOUN_RE = re.compile(r"\b(it|this|that|them|those|these|one)\b", re.IGNORECASE)
_PRONOUNS = {"it", "this", "that", "them", "those", "these", "one"}
VAGUE_PRONOUN_REPLY = ("Sure - which feature or item do you mean? Please mention it in your question, and I'll walk you "
                       "through it.")


_INTRO_NAME_RE = re.compile(
    r"\b(?:i\s*am|i'm|im|this\s+is|my\s+name\s+is|myself|it'?s)\s+([a-z][a-z'-]{1,30})\b|"
    r"\b([a-z][a-z'-]{1,30})\s+(?:here|this\s+side)\b", re.IGNORECASE)
_NOT_NAMES = {"back", "here", "new", "fine", "good", "ok", "okay", "looking", "trying", "a", "an", "the", "from",
              "with", "student", "admin", "user", "verifier", "chatbot", "bot", "assistant", "there", "again", "just",
              "also", "still", "so", "well", "hello", "hi", "hey", "not", "very", "really", "interested", "confused"}


def introduced_name(text: str) -> str:
    """The name a user gives in a greeting ("Ramesh here", "this is Priya", "I am Ravi") - "" when there is none."""
    for m in _INTRO_NAME_RE.finditer(text or ""):
        name = (m.group(1) or m.group(2) or "").strip("'-")
        if name and name.lower() not in _NOT_NAMES and name.lower() not in STOPWORDS:
            return name[:1].upper() + name[1:].lower()
    return ""


def mentions_own_data(text: str) -> bool:
    """The user refers to their own account / records ("my payment", "our verification status", "I paid")."""
    return bool(_OWN_DATA_RE.search(text or ""))


def is_vague_pronoun_question(text: str, history: list[dict] | None) -> bool:
    """A short question whose only subject is a pronoun with nothing to refer to ("How do I configure it?",
    "What is this?") - the topic is unknown, so ask instead of guessing."""
    if last_assistant_answer(history or []):
        return False                      # "it" can refer to the previous answer: a normal follow-up
    words = _words(text)
    if not words or len(words) > 7 or not _PRONOUN_RE.search(text or ""):
        return False
    content = [w for w in words if w not in STOPWORDS and w not in QUESTION_WORDS and w not in _PRONOUNS
               and w not in {"i", "do", "does", "can", "how", "please", "pls"}]
    return len(content) <= 1             # at most the verb: "configure", "set", "use", "change"


def is_closing(text: str) -> bool:
    """The user is ending the conversation (e.g. "no" after "anything else?", "that's all", "bye")."""
    return bool(CLOSING_RE.match(text or ""))


def is_acknowledgement(text: str) -> bool:
    """A short acknowledgement (ok, okay, thanks, thank you, bye, done, a thumbs-up), case-insensitive, after
    removing punctuation, emoji skin tones and repeated letters ("okkk", "thanksss")."""
    t = (text or "").strip().lower()
    if not t:
        return False
    for emoji in _ACK_EMOJI:
        t = t.replace(emoji, " ok ")
    t = re.sub(r"[🏻-🏿️]", "", t)           # skin tones / variation selectors
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)                              # "okkkk" -> "ok", "thanksss" -> "thanks"
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if not words or len(words) > 4:
        return False
    if t in ACK_WORDS or " ".join(dict.fromkeys(words)) in ACK_WORDS:   # "ok ok" -> "ok"
        return True
    return all(w in {"ok", "okay", "thanks", "thank", "you", "u", "bye", "done", "great", "cool", "so", "much",
                     "very", "a", "lot", "again"} for w in words) and words[0] in {"ok", "okay", "thanks", "thank",
                                                                             "bye", "done", "great", "cool"}


@lru_cache(maxsize=8)
def _compile_intro(names: str):
    return re.compile(_GREETING_INTRO.replace("{names}", names), re.IGNORECASE)


def is_greeting_with_intro(text: str) -> bool:
    """A greeting, optionally addressing the bot and introducing the user, with no question in it."""
    from backend.routing.router import names_alternation

    t = (text or "").strip()
    return bool(t) and "?" not in t and bool(_GREETING_WORD.search(t)) and         bool(_compile_intro(names_alternation()).match(t))

INCOMPLETE_REPLY = "Sure - what would you like to know? Please share your full question."


# Product-specific replies come from the product config (config/products/<id>.yaml -> replies).
def _replies():
    from backend.products import current_product
    return current_product().replies


def rephrase_text() -> str:
    return _replies().rephrase


def account_reply() -> str:
    return _replies().account


def escalation_reply() -> str:
    return _replies().escalation


def frustration_text() -> str:
    return _replies().frustration


def frustration_again_text() -> str:
    return _replies().frustration_again


def incomplete_hints() -> dict[str, str]:
    from backend.products import current_product
    return current_product().incomplete_hints or {"what": ""}


_GREETING_ONLY = (
    r"^\s*((hi+|hello+|hey+|hiya|helo+|hlo+|hllo|hellow|helllo|hai|hy|hie|yo|greetings|namaste|namaskar)"
    r"(\s+(there|team|{names}bot|all))?|(good|gud|gd)\s*(morning|afternoon|evening|day)|gm)\s*[!.?]*\s*$")
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
ACCOUNT_RE = re.compile(
    r"\b(has|have|did|is|was|were|are)\s+(my|our)\s+(payment|transaction|refund|order|subscription|verification|"
    r"certificate|application|receipt|invoice)s?\b(?!\s+(safe|secure|protected|tamper|fake|forged|stolen|hack))|"
    r"\b(status|update)\s+of\s+(my|our)\s+(payment|transaction|refund|order|verification|application)\b|"
    r"\b(my|our)\s+(payment|transaction|refund|order|verification|application)\s+(status|is pending|is not|isn't|"
    r"failed|went through|gone through|deducted|stuck)\b|"
    r"\b(check|show|send|get|download)\s+(me\s+)?(my|our)\s+(receipt|invoice|payment status|verification status|"
    r"transaction history)\b|\bmoney (was |got )?deducted\b", re.IGNORECASE)
_ESCALATION = (
    r"\b(complain\w*|harass\w*|abusive|legal action|legal notice|lawyer|lawsuit|sue (you|{names}us)|consumer (court|forum)|"
    r"(want|need|demand|give me)\s+(a\s+|my\s+)?(refund|money back)|refund\s+(is\s+)?(not|never|still)\b)")
FRUSTRATION_RE = re.compile(
    r"\b(go to hell|shut up|get lost|screw (you|this)|damn|dammit|wtf|wth|f+u+c+k\w*|f\*+\w*|sh[i1]t\w*|bullshit|crap|"
    r"idiot\w*|stupid|dumb|moron\w*|useless|pathetic|rubbish|nonsense|garbage|trash|worthless|hopeless|"
    r"waste of (my )?time|not (at all )?helpful|no help|you (are|r) (bad|terrible|horrible|the worst)|"
    r"(worst|terrible|horrible|awful|bad) (bot|app|service|support|experience|chatbot|assistant)|"
    r"(i hate|hate) (you|this)|so annoying|frustrat\w*|fed up|sick of (this|you))\b|"
    r"\b(this|it|you) (sucks?|is (so )?bad)\b", re.IGNORECASE)
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
@lru_cache(maxsize=32)
def _compile_names(pattern: str, names: str) -> re.Pattern:
    return re.compile(pattern.replace("{names}", names), re.IGNORECASE)


def greeting_only_re() -> re.Pattern:
    return _compile_names(_GREETING_ONLY, names_alternation())


def escalation_re() -> re.Pattern:
    return _compile_names(_ESCALATION, names_alternation())


def is_small_talk(text: str) -> bool:
    """Greeting / thanks / bye / yes - handled by fixed replies, so no normalization or LLM call is needed."""
    return bool(greeting_only_re().match(text or "") or greeting_re().match(text or "") or AFFIRM_RE.match(text or ""))


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


def _last_assistant(history: list[dict]) -> dict | None:
    return next((h for h in reversed(history or []) if h.get("role") == "assistant"), None)


def frustration_reply(history: list[dict]) -> tuple[str, bool]:
    """Calm, short reply to frustration. Second frustrated message in a row -> offer the support team. Otherwise
    offer to retry the user's last real question when there is one."""
    last = _last_assistant(history)
    if last and last.get("answer_type") == "frustration":
        return frustration_again_text(), True
    prev_q = next((h.get("content", "") for h in reversed(history or []) if h.get("role") == "user"
                   and not FRUSTRATION_RE.search(h.get("content", "")) and len(_words(h.get("content", ""))) >= 3), "")
    if prev_q:
        topic = prev_q.strip().rstrip("?.!")[:80]
        return (f"Sorry you're having a frustrating time. Was it my answer about \"{topic}\"? Tell me what went wrong "
                "and I'll try again, or I can connect you with our support team."), True
    return frustration_text(), False


def _clarify_reply(history: list[dict], message: str) -> tuple[str, bool]:
    prev = last_assistant_answer(history)
    if prev:
        items = answer_items(prev, limit=5)
        if "which" in message.lower() and items:
            opts = ", ".join(f'"{i}"' for i in items[:-1]) + f' or "{items[-1]}"' if len(items) > 1 else f'"{items[0]}"'
            return (f"Are you referring to one of the points we discussed earlier - {opts}? "
                    "Please tell me which one and I'll explain it."), True
        return "Could you clarify which part of my previous answer you'd like to know more about?", True
    return "Could you clarify what you'd like to know?", False


# ------------------------------------------------------------------------------------------------ classifier
def classify_message(text: str, history: list[dict] | None = None, known_word: Callable[[str], bool] | None = None,
                     llm=None) -> MessageIntent:
    """Classify one (already validated and safety-screened) user message. Never raises."""
    history = history or []
    t = (text or "").strip()
    words = _words(t)
    known = known_word or (lambda w: w in STOPWORDS or w in QUESTION_WORDS)
    has_history = bool(last_assistant_answer(history))

    # 0. ending the conversation: goodbye + rating request (backend/services/chat_service.py)
    if is_closing(t):
        return MessageIntent(CLOSING, reason="closing")

    # 1. greetings / small talk (incl. typos such as "helo", "gud morning") - no retrieval, no LLM
    if greeting_only_re().match(t) or is_greeting_with_intro(t):
        return MessageIntent(GREETING, reason="greeting")
    if greeting_re().match(t) or AFFIRM_RE.match(t) or is_acknowledgement(t):
        return MessageIntent(SMALL_TALK, reason="small talk")

    # 1b. frustration, anger, insults or swearing: calm acknowledgement, ask what went wrong (support if repeated)
    real_question = bool(words) and words[0] in QUESTION_WORDS and len(words) >= 4
    if FRUSTRATION_RE.search(t) and len(words) <= 10 and not real_question:
        reply, used = frustration_reply(history)
        return MessageIntent(FRUSTRATION, reply=reply, uses_history=used, reason="frustration")

    # 2. no letters at all ("@@@###", "???", "123")
    if not re.search(r"[^\W\d_]", t):
        return MessageIntent(GIBBERISH, reply=rephrase_text(), reason="no words")
    if not words:                       # written only in another script (e.g. Devanagari): a question to answer
        return MessageIntent(KNOWLEDGE, reason="non-Latin script")

    # 2b. this user's own account (no lookup tools) and complaints / disputes (human support) - fixed replies
    if escalation_re().search(t):
        return MessageIntent(ESCALATION, reply=escalation_reply(), reason="complaint / dispute -> support")
    if ACCOUNT_RE.search(t):
        return MessageIntent(ACCOUNT, reply=account_reply(), reason="personal account data")

    # 3. requests to continue / elaborate - resolved from the conversation, or clarified when there is none
    if ELABORATE_RE.match(t):
        if has_history:
            return MessageIntent(FOLLOW_UP, uses_history=True, reason="elaboration request")
        return MessageIntent(CLARIFICATION, reason="elaboration without a previous topic",
                             reply="Sure - which topic would you like me to explain?")

    # 4. a bare interrogative: "What?", "Which one?", "When?"
    if CLARIFY_ONLY_RE.match(t):
        reply, used = _clarify_reply(history, t)
        return MessageIntent(CLARIFICATION, reply=reply, uses_history=used, reason="bare interrogative")

    # 4b. "How do I configure it?" as the first message: nothing for "it" to refer to -> ask which feature
    if is_vague_pronoun_question(t, history):
        return MessageIntent(CLARIFICATION, reply=VAGUE_PRONOUN_REPLY, reason="pronoun without a topic")

    # 5. a lone (misspelt) question word: "Ho", "Wht", "Hw", "Wh" - the intent is genuinely unclear, so ask for the
    #    question. The user's spelling is never mentioned or corrected in replies: multi-word messages with typos
    #    are normalized and answered (backend/routing/normalize.py).
    if 1 <= len(words) <= 2 and not t.endswith("?") or len(words) == 1:
        first, matched = _typo_target(words[0]) if words else (None, False)
        rest_ok = all(w in STOPWORDS or w in TYPOS for w in words[1:])
        if matched and rest_ok and (first is None or first in QUESTION_WORDS):
            if first is None:
                opts = ", ".join(f'"{w}"' for w in AMBIGUOUS_STARTS[:-1]) + f' or "{AMBIGUOUS_STARTS[-1]}"'
                return MessageIntent(TYPO_OR_INCOMPLETE, reason=f"lone word: {words[0]}", reply=INCOMPLETE_REPLY)
            return MessageIntent(TYPO_OR_INCOMPLETE, suggestion=first.capitalize(), reason=f"lone word: {words[0]}",
                                 reply=INCOMPLETE_REPLY)

    # 6. only function words / truncated words: "What is", "How to", "Tell me", "Can you", "How t"
    if words and len(words) <= 5 and all(w in STOPWORDS or w in QUESTION_WORDS or w in TYPOS
                                                  or len(w) == 1 for w in words):
        return MessageIntent(TYPO_OR_INCOMPLETE, reason="incomplete question", reply=INCOMPLETE_REPLY)

    # 7. random characters: "xyz123", "asdfghjkl"
    if len(words) <= 3 and all(_looks_gibberish(w, known) for w in words):
        return MessageIntent(GIBBERISH, reply=rephrase_text(), reason="gibberish")

    # 8. clearly unrelated requests (the router's own out-of-scope checks still apply to everything else)
    if OUT_OF_SCOPE_RE.search(t):
        return MessageIntent(OUT_OF_SCOPE, reason="unrelated request")

    # 9. short message referring back to the conversation: "What about its features?", "And them?"
    if has_history and len(words) <= 8 and FOLLOW_UP_RE.search(t):
        return MessageIntent(FOLLOW_UP, uses_history=True, reason="reference to the previous turn")

    # 10. short message with words the rules can't place -> ask the LLM (if any); otherwise treat as a question
    # (a question word followed by a subject - "What is <Product>?" - is already a well-formed question)
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
    if intent not in INTENTS or intent in (FRUSTRATION, ACCOUNT, ESCALATION):
        return None      # these three are decided by the rules only (an LLM guess must never label a greeting angry)
    suggestion = str(data.get("suggestion") or "").strip()
    suggestion = suggestion if re.fullmatch(r"[A-Za-z][A-Za-z ]{0,30}", suggestion) else ""
    result = MessageIntent(intent, suggestion=suggestion, method="llm", reason="llm classification",
                           uses_history=intent == FOLLOW_UP)
    if intent == TYPO_OR_INCOMPLETE:
        if len(_words(text)) >= 2:          # a misspelt question is still a question: answer it, never ask to retype
            result.intent, result.reason = KNOWLEDGE, "llm: misspelt question treated as a question"
        else:
            result.reply = INCOMPLETE_REPLY
    elif intent == CLARIFICATION:
        result.reply, result.uses_history = _clarify_reply(history, text)
    elif intent == GIBBERISH:
        result.reply = rephrase_text()
    elif intent == FOLLOW_UP and not last_assistant_answer(history):
        result.intent, result.uses_history = KNOWLEDGE, False
    return result
