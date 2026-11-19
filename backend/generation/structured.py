"""Structured answer from the LLM: {"reply", "ui_action", "conversation_state"}.

The answer prompt asks the model for one JSON object. `parse_reply` validates it; anything that is not valid
(plain text, broken JSON, unknown values) falls back to the text as the reply with ui_action = "none", so a model that
ignores the format can never break an answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

UI_ACTIONS = ("none", "show_rating", "show_escalation", "show_login")
STATES = ("active", "closing", "closed")


@dataclass
class StructuredReply:
    reply: str
    ui_action: str = "none"
    conversation_state: str = "active"
    parsed: bool = False          # False = fallback (the model did not return valid JSON)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_reply(text: str) -> StructuredReply:
    raw = (text or "").strip()
    body = _FENCE.sub("", raw).strip()
    data = None
    if body.startswith("{"):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", body, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
    if not isinstance(data, dict) or not isinstance(data.get("reply"), str) or not data["reply"].strip():
        if body.startswith("{"):        # broken JSON: salvage the reply text instead of showing JSON to the user
            m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', body, re.DOTALL)
            if m:
                try:
                    return StructuredReply(json.loads(f'"{m.group(1)}"').strip())
                except json.JSONDecodeError:
                    return StructuredReply(m.group(1).replace("\\n", "\n").strip())
        return StructuredReply(raw)
    action = str(data.get("ui_action") or "none").strip().lower()
    state = str(data.get("conversation_state") or "active").strip().lower()
    return StructuredReply(reply=data["reply"].strip(), ui_action=action if action in UI_ACTIONS else "none",
                           conversation_state=state if state in STATES else "active", parsed=True)
