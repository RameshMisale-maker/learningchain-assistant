"""Input validation and prompt-injection screening for user queries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import get_settings
from ingestion.text_cleaner import detect_injection

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_PATTERNS = re.compile(r"(api[_\- ]?key|password|secret|token)\s*[:=]\s*\S{8,}", re.IGNORECASE)

def blocked_message() -> str:
    """Reply to prompt-injection attempts (config/products/<id>.yaml -> replies.blocked)."""
    from backend.products import current_product
    return current_product().replies.blocked

# ------------------------------------------------------------------------------------------------ sensitive data
# Card numbers, CVV, OTP, UPI PIN and passwords must never be requested, accepted or stored. A message containing one
# gets a warning instead of an answer, and only a redacted copy is saved in the chat history.
SENSITIVE_DATA_MESSAGE = ("For your safety, please don't share card numbers, CVV, OTP, UPI PIN or passwords here - and "
                          "please don't repeat it. I've not saved it. Our team will never ask for these details. How else "
                          "can I help you?")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SECRET_FIELD_RE = re.compile(
    r"\b(cvv|cvc|otp|one[ -]?time[ -]?password|upi[ -]?pin|m?pin|atm[ -]?pin|pass(word|code|wd)?|pwd)\b"
    r"(\s*(is|was|=|:|-|no\.?|number)?\s*)([^\s,.;]{3,})", re.IGNORECASE)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _card_numbers(text: str) -> list[re.Match]:
    return [m for m in _CARD_RE.finditer(text or "")
            if 13 <= len(digits := re.sub(r"\D", "", m.group(0))) <= 19 and _luhn_ok(digits)]


def _secret_values(text: str) -> list[re.Match]:
    # "password reset" / "forgot my password" are questions, not secrets: the value must contain a digit or symbol,
    # or follow an explicit "is" / ":" / "="
    out = []
    for m in _SECRET_FIELD_RE.finditer(text or ""):
        value, sep = m.group(5), (m.group(4) or "").strip().lower()
        if re.search(r"[\d@#$%^&*!]", value) or sep in ("is", "was", "=", ":"):
            if value.lower() not in ("reset", "change", "forgot", "recovery", "the", "my", "a", "for", "to", "and",
                                     "not", "required", "needed", "policy", "expired", "wrong", "incorrect"):
                out.append(m)
    return out


def sensitive_data(text: str) -> str | None:
    """'card' / 'secret' when the message contains a payment card number or a secret value (CVV, OTP, PIN, password)."""
    if _card_numbers(text):
        return "card"
    if _secret_values(text):
        return "secret"
    return None


def redact_sensitive(text: str) -> str:
    """Copy of the message safe to store: card numbers and secret values replaced with [REDACTED]."""
    out = text or ""
    for m in sorted(_card_numbers(out), key=lambda m: m.start(), reverse=True):
        out = out[:m.start()] + "[REDACTED]" + out[m.end():]
    for m in sorted(_secret_values(out), key=lambda m: m.start(5), reverse=True):
        out = out[:m.start(5)] + "[REDACTED]" + out[m.end(5):]
    return out


# ------------------------------------------------------------------------------------------------ restricted topics
# Guardrail: healthcare, banking/financial advice, sexual content, suicide/self-harm, violence and illegal activity are
# refused before any retrieval or LLM call. Patterns are phrase-level on purpose: product questions about
# "fake certificates", "hacking" blockchain records or verification "payments" must still be answered.
def restricted_topic_message() -> str:
    """Polite refusal for restricted topics, redirecting to the product (replies.restricted_topic)."""
    from backend.products import current_product
    return current_product().replies.restricted_topic


SELF_HARM_MESSAGE = ("I'm really sorry you're going through this, but I can't help with that here. You don't have to "
                     "face it alone - please reach out to someone you trust, or call the Tele-MANAS mental health "
                     "helpline on 14416 (India, free, 24x7). If you are in immediate danger, call 112.")
_RESTRICTED: list[tuple[str, re.Pattern]] = [
    ("self_harm", re.compile(
        r"\b(suicid\w*|kill (my ?self|myself)|end (my|his|her) (own )?life|take my (own )?life|self[- ]?harm\w*|"
        r"hurt (my ?self|myself)|cut (my ?self|myself)|want to die|lethal dose|overdos\w*)\b", re.I)),
    ("sexual", re.compile(
        r"\b(sex|sexual\w*|porn\w*|nude\w*|nudity|erotic\w*|explicit (story|content|image)s?|onlyfans|fetish\w*)\b",
        re.I)),
    ("healthcare", re.compile(
        r"\b(medical (advice|advise|help|treatment|opinion)|medicine|medication|dosage|prescription|symptoms?|diagnos\w+|disease|fever|headache|blood (test|pressure|"
        r"sugar)|diabetes|cancer|pregnan\w+|vaccin\w+|antibiotic\w*|therapy for|doctor (says|recommend)|treat (my|a) "
        r"(cold|cough|pain|infection|illness))\b", re.I)),
    ("banking_financial", re.compile(
        r"\b(home loan|personal loan|car loan|loan (interest|emi|eligibility)|mortgage|credit (card|score)|"
        r"interest rates?|fixed deposit|mutual funds?|stock market|share price|stocks? to buy|invest (my|in)|"
        r"investment (advice|strateg\w*|tips|plans?)|banking (advice|investments?)|(buy|sell|invest in|price of) (bitcoin|ethereum|crypto\w*)|trading tips|"
        r"tax (advice|saving)|which bank)\b", re.I)),
    ("violence", re.compile(
        r"\b(how to (kill|murder|hurt|attack|stab|shoot|poison) |make a (bomb|weapon|gun)|build a (bomb|weapon)|"
        r"(buy|get) a (gun|weapon) illegally)", re.I)),
    ("illegal", re.compile(
        r"\b(how (do|can|to) (i |we )?(hack|break) into|steal\w*|launder\w*|buy drugs|sell drugs|counterfeit money|"
        r"(make|create|forge|print) (a )?(fake|forged|counterfeit) (certificate|degree|marksheet|document)s?|"
        r"bypass (the )?(verification|blockchain|security|login)|pick a lock|piracy|pirated)\b", re.I)),
]


def restricted_topic(query: str) -> str | None:
    """Name of the restricted topic the query asks about, or None."""
    for name, pattern in _RESTRICTED:
        if pattern.search(query or ""):
            return name
    return None


def restricted_topic_reply(topic: str) -> str:
    return SELF_HARM_MESSAGE if topic == "self_harm" else restricted_topic_message()


@dataclass
class QueryCheck:
    cleaned: str
    allowed: bool = True
    reason: str = ""
    flags: list[str] = field(default_factory=list)


def check_query(query: str) -> QueryCheck:
    settings = get_settings()
    cleaned = _CONTROL_RE.sub("", query or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return QueryCheck(cleaned="", allowed=False, reason="empty")
    if len(cleaned) > settings.max_query_chars:
        return QueryCheck(cleaned=cleaned[: settings.max_query_chars], allowed=False, reason="too_long")
    injection = detect_injection(cleaned)
    flags = list(injection)
    if _SECRET_PATTERNS.search(cleaned):
        flags.append("possible secret in query")
    if injection and settings.block_prompt_injection:
        return QueryCheck(cleaned=cleaned, allowed=False, reason="prompt_injection", flags=flags)
    return QueryCheck(cleaned=cleaned, flags=flags)
