"""Chat orchestration - the complete RAG execution path for one user question.

  validate & screen query -> conversational intent layer (greeting / small talk / typo / clarification /
  gibberish answered directly; follow-ups resolved from history) -> follow-up rewrite + abbreviation expansion
  -> route (intent + KBs + ACL) -> golden fast path (approved answer verbatim) -> hybrid retrieval per KB
  (dense + BM25, multi-query fusion, per entity for comparisons) -> rerank -> context selection
  -> LLM generation (or extractive fallback)
  -> grounding / hallucination check -> citations -> persist history, metrics and audit
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from backend.common.timing import Timings, request_timings, stage
from backend.db import repository
from backend.generation.extractive import extractive_answer
from backend.generation.grounding import (REFUSAL_TEXT, GroundingReport, cited_ids, is_refusal, lexical_grounding,
                                          trim_unsupported,
                                          llm_grounding, normalize_citations, remove_invalid_citations,
                                          remove_unknown_urls, uncounted_numbers)
from backend.generation.failover import AllProvidersDown, reset_dense_flag
from backend.generation.structured import parse_reply
from backend.services import response_cache, token_budget
from backend.generation.llm_client import LLMClient, LLMRefusal, LLMResult, get_llm
from backend.generation.prompt_builder import SourceRef, build_answer_prompt, select_context
from backend.retrieval.golden_matcher import GoldenMatch, GoldenMatcher
from backend.retrieval.hybrid import Candidate, HybridRetriever
from backend.retrieval.kb_index import get_index_manager
from backend.retrieval.reranker import Reranker, get_reranker, weighted_coverage
from backend.common.english_vocab import is_common_word
from backend.common.text_utils import STOPWORDS, stem
from backend.routing import intent as convo
from backend.routing.decompose import decompose
from backend.routing.normalize import Understanding, llm_understand, reply_language, understand
from backend.routing.router import QueryRouter, RouteDecision
from backend.security.auth import AuthenticatedUser
from backend.products import current_product, use_product
from backend.security.prompt_guard import (SENSITIVE_DATA_MESSAGE, blocked_message, check_query, redact_sensitive,
                                           restricted_topic, restricted_topic_reply, sensitive_data)
from config.settings import Settings, get_settings
from ingestion.embedding import Embedder, get_embedder

logger = logging.getLogger(__name__)

# Product texts (greetings, fixed replies, contacts) come from config/products/<id>.yaml via current_product().
def _replies():
    return current_product().replies



is_closing = convo.is_closing          # closing phrases live with the intent classifier (backend/routing/intent.py)


def small_talk_reply(text: str, roles: set[str], first: bool = True, intent: str | None = None) -> str:
    """Short reply for greetings / small talk (one sentence). The assistant introduces itself only on the first
    message of a conversation. The WHOLE message decides the reply: "okay hello chatbot, Ramesh here" is a greeting
    (not an "ok"), so greetings are recognised before the acknowledgement / thanks replies."""
    r = _replies()
    t = text.strip().lower()
    if is_closing(t):
        return r.closing
    social = convo.social_message(t)       # greeting / name / how-are-you in any combination: answer every part
    if social is not None:
        if not (social.greeting or social.name):
            if social.courtesy and not social.wellbeing:
                if re.search(r"\b(there|available|online)\b", t):          # "are you there?"
                    return "Yes, I'm here! How can I help you today?"
                return "Nice to meet you too! How can I help you today?"
            return r.how_are_you           # just "how are you doing?"
        return _greeting(t, r, first, wellbeing=social.wellbeing, courtesy=social.courtesy)
    if intent == convo.GREETING or convo.is_greeting_with_intro(t) or convo.greeting_only_re().match(t):
        return _greeting(t, r, first)
    if re.match(r"^(yes|yeah|yep|yup|sure|ya|of course|go ahead)\b", t):
        return r.yes
    if re.search(r"\b(thanks|thank\s*(you|u)|thx|ty)\b", t):     # "thanks", "great, thanks", "ok thank you"
        return r.thanks
    if re.match(r"^(bye|goodbye|good\s*bye|see you|see ya|take care)\b", t):
        return r.bye
    if convo.is_acknowledgement(t):    # the whole message is "ok" / "great, thanks" / a thumbs-up
        return r.acknowledged
    if re.match(r"^how are you", t):
        return r.how_are_you
    if re.match(r"^(who are you|what are you|what is your name|are you)", t):
        return r.who_are_you
    if re.match(r"^(what can you do|how can you help|help)", t):
        return r.what_can_you_do
    return _greeting(t, r, first)


def _greeting(t: str, r, first: bool, wellbeing: bool = False, courtesy: bool = False) -> str:
    """Hello (time-of-day aware, with the user's name when they introduced themselves, answering "how are you" when
    asked); the assistant introduces itself only on the first message."""
    m = re.search(r"\b(good|gud|gd)\s*(morning|afternoon|evening|day)\b", t)
    name = convo.introduced_name(t)
    hello = (f"Good {m.group(2)}" if m else ("Hello" if first else "Hi")) + (f" {name}!" if name else "!")
    if wellbeing:
        hello += " I'm doing well, thanks for asking."
    elif courtesy:
        hello += " Nice to meet you too."
    if first:
        return r.greeting_first.format(hello=hello)
    if m or name or wellbeing or courtesy:
        return r.greeting_again_timed.format(hello=hello)
    return r.greeting_again
AMBIGUOUS_MIN_SCORE = 0.35
LLM_CONFIDENT_RERANK = 0.5        # with an LLM, skip the lexical pre-refusal when the top passage scores this high
BROAD_TOPIC_CONFIDENT = 0.8       # "tell me about <topic>": answer directly only when the best passage scores this high
# platform / service names users often type in lower case; always treated as specific terms by the undocumented guard
PLATFORM_TERMS = frozenset({"ios", "android", "iphone", "ipad", "macos", "windows", "linux", "whatsapp", "telegram",
                            "aadhaar", "upi", "paytm", "gpay", "sla", "uptime"})
GOLDEN_MAX_MISSING_INTENT = 0.35   # max IDF share of query terms absent from the approved question side
PREMISE_NOTE = ("\n\nNOTE: the question assumes \"{terms}\", which no SOURCE mentions. Apply rule 2a: if the SOURCES "
                "state the actual fact on that point (for example the list of locations, formats or providers), correct "
                "the assumption with it - \"<fact from the SOURCES>, not <the assumption>\" with its source tag, in one "
                "sentence. Only if the SOURCES say nothing about that point, reply with the sentence of rule 2.")
PREMISE_NOTE_GENERIC = ("\n\nNOTE: the question takes something for granted (\"since ...\", \"which of ... in ...\"). Apply "
                        "rule 2a: if the SOURCES state the actual fact on that point (for example the list of "
                        "locations, formats or providers), correct the assumption with it - \"<fact from the SOURCES>, "
                        "not <the assumption>\" with its source tag, in one sentence. Only if the SOURCES say nothing "
                        "about that point, reply with the sentence of rule 2.")
REFUSED_ANSWER_TYPES = ("refusal", "out_of_scope", "blocked", "access_denied", "error", "restricted", "budget_exceeded")
GOLDEN_COMPARISON_MIN = 0.92   # golden match score needed to answer a comparison question with an approved answer
GOLDEN_EXTRA_ASSISTS = 2       # approved answers (>= assist threshold) added as candidates besides the best match
SUB_QUESTION_TOP_N = 12        # candidates retrieved per sub-question of a multi-part question
# Paraphrase handling (see _semantic_golden_ok): dense similarity of question vs approved question
SEMANTIC_GOLDEN_STRONG = 0.85  # this similar on its own -> use the approved answer
SEMANTIC_GOLDEN_MIN = 0.68     # this similar AND the re-ranker's top source is that same approved Q&A
SEMANTIC_GOLDEN_RERANK = 0.80  # ... with at least this re-rank score
UNVERIFIED_NOTE = ("\n\n_Note: the generated answer could not be fully verified against the sources, so the most "
                   "relevant source text is shown instead._")


@dataclass
class Citation:
    id: str
    doc_name: str
    kb: str
    source_file: str
    page: int | None
    section: str
    doc_id: str
    version: int | None
    qid: str = ""
    snippet: str = ""
    score: float = 0.0
    is_sample: bool = False
    is_golden: bool = False
    url: str = ""          # link to the original page / article (from ingestion metadata)
    question: str = ""     # approved question text, for golden Q&A sources


@dataclass
class ChatResponse:
    answer: str
    session_id: str
    message_id: int | None
    answer_type: str                      # golden_direct | generated | extractive | refusal | clarification | ...
    knowledge_bases: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    route: dict[str, Any] = field(default_factory=dict)
    golden_match: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    contains_sample_data: bool = False
    retrieved: list[dict[str, Any]] = field(default_factory=list)   # debug/eval info (chunk ids + scores)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)     # per-stage latency (ms), see backend/common/timing
    tokens_per_second: float = 0.0
    ui_action: str = "none"                  # none | show_rating | show_escalation | show_login (front end)
    conversation_state: str = "active"       # active | closing | closed (stored per session)
    versions: dict[str, str] = field(default_factory=dict)      # index_version / prompt_version (sent with ratings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_allows(kb: str, roles: set[str]) -> bool:
    cfg = get_settings().kb_config()["knowledge_bases"].get(kb, {})
    return "admin" in roles or bool(roles & set(cfg.get("allowed_roles", [])))


class ChatService:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 retriever: HybridRetriever | None = None, reranker: Reranker | None = None,
                 golden: GoldenMatcher | None = None, router: QueryRouter | None = None,
                 llm: LLMClient | None | bool = True) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder()
        if self.settings.llm_failover_enabled:      # embedding outage -> BM25-only search instead of an error
            from backend.generation.failover import ResilientEmbedder
            self.embedder = ResilientEmbedder(self.embedder, self.settings)
        self.llm: LLMClient | None = get_llm() if llm is True else (llm or None)
        self.retriever = retriever or HybridRetriever(self.settings, embedder=self.embedder)
        self.reranker = reranker or get_reranker()
        self.golden = golden or GoldenMatcher(self.embedder)
        self.router = router or QueryRouter(self.settings, self.embedder, llm=self.llm)
        if self.settings.golden_warmup:
            # embed the approved questions in the background now, so the first user question does not wait for it
            import threading
            threading.Thread(target=self.golden.warm, name="golden-warmup", daemon=True).start()

    @property
    def llm(self) -> LLMClient | None:
        """The AI model client - None for a request that is over its token budget (token_budget.suspend_llm)."""
        return None if token_budget.llm_suspended() else self._llm

    @llm.setter
    def llm(self, value: LLMClient | None) -> None:
        self._llm = value

    # ================================================================== public API
    def answer(self, question: str, user: AuthenticatedUser, session_id: str | None = None,
               persist: bool = True, ip_address: str = "", history: list[dict] | None = None,
               allowed_kbs: list[str] | None = None, channel: str = "api", assistant: str = "all",
               record_metrics: bool | None = None, product=None) -> ChatResponse:
        """Answer one question.

        history      : prior turns [{"role", "content"}] supplied by the client (Open WebUI sends the whole
                       conversation); when omitted, history is loaded from the stored session.
        allowed_kbs  : restrict this assistant to these knowledge bases (e.g. the product-only chatbot); questions
                       routed elsewhere get the out-of-scope reply, so one assistant can never read another's KB.
        channel/assistant : labels for the performance metrics (dashboard filters).
        product            : the product (config/products/<id>.yaml) this request belongs to; its replies, glossary and
                             prompts are used for the whole request. Default: the default product.
        """
        with use_product(product):
            reset_dense_flag()
            response_cache.clear_pending()
            started = time.perf_counter()
            session_id = repository.ensure_session(session_id, user.username) if persist else (session_id or "eval")
            if history is None:
                history = repository.get_history(session_id, user.username, limit=self.settings.chat_history_messages) if persist else []
            history = [h | {"content": redact_sensitive(str(h.get("content", "")))} for h in history]
            state = self._conversation_state(session_id, history, persist)
            if state in ("closing", "closed") and convo.is_acknowledgement(question):
                # "ok" / "thanks" / a thumbs-up after the goodbye: no LLM call, no new message, rating stays visible
                if persist:
                    repository.set_conversation_state(session_id, "closed")
                response = ChatResponse(answer="", session_id=session_id, message_id=None, answer_type="closed",
                                        confidence=1.0, route={"intent": "closed", "conversation_intent": convo.CLOSING},
                                        ui_action="show_rating", conversation_state="closed",
                                        latency_ms=round((time.perf_counter() - started) * 1000, 1))
                response.versions = self._versions(response)
                return response
            error = ""
            budget = token_budget.check(user.username, current_product().product_id)
            with request_timings() as timings, token_budget.suspend_llm(not budget.allowed):
                try:
                    response = self._answer(question, user, session_id, history, allowed_kbs, timings)
                except Exception as exc:   # never leak internals; record the failure for the dashboard
                    logger.exception("Chat request failed")
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    response = ChatResponse(answer=self.settings.insufficient_info_text, session_id=session_id,
                                            message_id=None, answer_type="error", warnings=["internal_error"])
            if not budget.allowed:
                response.route["token_budget"] = budget.to_dict()
                if response.answer_type in ("extractive", "extractive_fallback"):
                    # this question needed the AI model: say the limit is reached instead of quoting raw passages
                    response.answer = token_budget.budget_reply(budget, current_product().support_contact)
                    response.answer_type, response.citations = token_budget.BUDGET_ANSWER_TYPE, []
                    response.warnings.append(f"token budget exceeded ({budget.scope}: {budget.used}/{budget.limit})")
                if self.settings.metrics_enabled:
                    try:
                        from backend.observability.metrics import record_budget_block

                        record_budget_block(current_product().product_id, budget.scope)
                    except Exception:
                        logger.exception("Could not update Prometheus metrics")
            if response.answer_type in ("refusal", "error") and not response.route.get("refusal_kind"):
                # in scope, but no verified evidence: the fixed "insufficient information" reply, no sources
                response.answer, response.citations = self.settings.insufficient_info_text, []
            elif response.answer_type == "out_of_scope" and not response.route.get("refusal_kind"):
                response.answer, response.citations = self.settings.out_of_scope_text, []
            # never send exactly the same reply twice in a row
            previous = next((h for h in reversed(history) if h.get("role") == "assistant"), None)
            if previous and str(previous.get("content", "")).strip() == response.answer.strip():
                alternative = _replies().repeat_alternatives.get(response.answer_type)
                if alternative and alternative.strip() != response.answer.strip():
                    response.answer = alternative
            # "Hi, Ramesh here, how do I verify my certificate?" -> greet first, then answer
            stripped = convo.strip_social(question.strip())
            if stripped != question.strip() and response.answer_type not in ("greeting", "closing", "closed"):
                hello = convo.greeting_line(question.strip()[: len(question.strip()) - len(stripped)])
                if hello:
                    response.answer = f"{hello}" + chr(10) + chr(10) + response.answer
                    response.route["greeting_prefix"] = hello
            response.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            response.timings = timings.as_dict() | {"end_to_end_ms": response.latency_ms}
            self._apply_ui_action(response)
            response.versions = self._versions(response)
            if persist:      # a real question after a goodbye re-opens the conversation
                repository.set_conversation_state(session_id, response.conversation_state)
            if persist:
                repository.add_message(session_id, "user", redact_sensitive(question)[: self.settings.max_query_chars])
                response.message_id = repository.add_message(
                    session_id, "assistant", response.answer, knowledge_bases=response.knowledge_bases,
                    citations=[asdict(c) for c in response.citations], route_info=response.route,
                    answer_type=response.answer_type, latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                    cost_usd=response.cost_usd)
                if response.answer_type in ("blocked", "access_denied"):
                    repository.write_audit(user.username, f"chat_{response.answer_type}", ",".join(response.route.get("denied_kbs", [])),
                                           outcome="denied", details={"question": redact_sensitive(question)[:300]},
                                           ip_address=ip_address)
            if self.settings.review_queue_enabled and persist:
                try:
                    from backend.services.review import auto_review

                    auto_review(response, redact_sensitive(question))
                except Exception:
                    logger.exception("Could not queue the answer for review")
            if self.settings.response_cache_enabled:
                try:
                    response_cache.store(response)
                except Exception:      # the cache must never break a user request
                    logger.exception("Could not store the response in the cache")
            if record_metrics if record_metrics is not None else persist:   # after persisting: logs the message id
                self._record_metrics(response, user, session_id, channel, assistant, error)
            return response

    # ================================================================== pipeline
    def _answer(self, question: str, user: AuthenticatedUser, session_id: str, history: list[dict],
                allowed_kbs: list[str] | None = None, timings: Timings | None = None) -> ChatResponse:
        s = self.settings
        timings = timings or Timings()
        check = check_query(question)
        if not check.allowed:
            text = blocked_message() if check.reason == "prompt_injection" else (
                "Please enter a question." if check.reason == "empty" else
                f"Your question is too long. Please keep it under {s.max_query_chars} characters.")
            return ChatResponse(answer=text, session_id=session_id, message_id=None,
                                answer_type="blocked" if check.reason == "prompt_injection" else "invalid",
                                route={"intent": "blocked", "flags": check.flags}, warnings=check.flags)
        query = original = check.cleaned
        # "Hi Ramesh here, how do I revoke a certificate?" -> the greeting is dropped so the question is searched on
        # its own (a purely social message is left unchanged and answered by the conversation layer)
        query = convo.strip_social(query)
        # card numbers, CVV, OTP, UPI PIN or passwords: warn, never process them further (no LLM, no retrieval)
        kind = sensitive_data(query)
        if kind:
            return ChatResponse(answer=SENSITIVE_DATA_MESSAGE, session_id=session_id, message_id=None,
                                answer_type="sensitive", confidence=1.0,
                                route={"intent": "sensitive", "kind": kind, "conversation_intent": "SENSITIVE_DATA"})
        # query normalization (backend/routing/normalize.py): typos, short forms and product-name variants are fixed
        # ("wt is <compny>?" -> "what is <Company>?"); the normalized text is used from here on, the original only
        # for logging. Greetings / small talk skip it (fixed replies, no LLM call).
        understanding: Understanding | None = None
        if s.query_normalization and not convo.is_small_talk(query):
            with stage("normalize_ms"):
                understanding = understand(query, known=self._known_word,
                                           llm=self.llm if s.normalization_llm else None, history=history)
            query = understanding.normalized
        # guardrail: restricted topics (healthcare, banking, sexual, self-harm, violence, illegal) are refused before
        # any retrieval or LLM call - safety rules take precedence over the request (checked on both spellings)
        topic = restricted_topic(query) or restricted_topic(original)
        if topic:
            return ChatResponse(answer=restricted_topic_reply(topic), session_id=session_id, message_id=None,
                                answer_type="restricted", confidence=1.0,
                                route={"intent": "restricted", "topic": topic,
                                       "conversation_intent": convo.OUT_OF_SCOPE})
        # conversational intelligence layer (backend/routing/intent.py): runs AFTER the safety checks above, so it
        # can never bypass them; greetings, typos, incomplete input and gibberish are answered without retrieval
        message = convo.MessageIntent(convo.KNOWLEDGE, reason="conversation layer disabled")
        if s.conversation_layer:
            with stage("intent_ms"):
                message = convo.classify_message(query, history, known_word=self._known_word,
                                                 llm=self.llm if s.conversation_llm_intent else None)
            direct = self._conversational_response(message, query, user, session_id, history)
            if direct is not None:
                if understanding:
                    direct.route.update(understanding.to_dict())
                return direct
        with stage("query_rewrite_ms"):
            standalone, extra_queries, tool_only = self.router.resolve_follow_up(
                query, history, force=message.intent == convo.FOLLOW_UP)
            expanded = self.router.expand(standalone)
            # "is it worth using?" / "why should I use it?": SEARCH for the documented features and benefits, so the
            # answer can present them (prompt rule 2b) instead of finding nothing for the word "worth"; the model
            # still gets the user's own question
            value_query = ""
            if convo.opinion_kind(original) == "value" or convo.opinion_kind(standalone) == "value":
                value_query = convo.value_search_query(original, standalone)
                expanded = self.router.expand(value_query)
        # the question needs this user's own account data (payment, receipt, status), not the help documentation:
        # there are no account tools, so explain where to find it instead of searching the documents
        # (only when the user talks about their OWN data - "my payment", "our verification status"; a general
        # troubleshooting question such as "the verification result shows Invalid" is answered from the documents)
        if (tool_only or (understanding and understanding.needs_account_tool and s.conversation_layer)) \
                and convo.mentions_own_account(query):
            account = convo.MessageIntent(convo.ACCOUNT, reply=convo.account_reply(),
                                          reason="needs account data (LLM)", method="llm")
            direct = self._conversational_response(account, query, user, session_id)
            direct.route.update(understanding.to_dict() if understanding else {})
            return direct
        with stage("query_embedding_ms"):
            qvec = self.embedder.embed_query(expanded)
        if s.response_cache_enabled:
            cached = self._cached_response(expanded, user, allowed_kbs, qvec, session_id)
            if cached is not None:
                return cached
        with stage("routing_ms"):
            route = self.router.route(expanded, user.roles, qvec)
        # lenient scope check: a keyword-based "out of scope" is only final when the LLM/semantic check agrees
        with stage("scope_check_ms"):
            scope_action = self._lenient_scope(route, understanding, expanded, user.roles, allowed_kbs, qvec)
        if allowed_kbs is not None and route.intent not in ("greeting", "out_of_scope", "clarify", "access_denied"):
            # assistant scoped to specific KBs: drop the others; nothing left -> out of scope for this assistant
            route.knowledge_bases = [k for k in route.knowledge_bases if k in allowed_kbs]
            if not route.knowledge_bases:
                route.intent, route.reason = "out_of_scope", f"not in this assistant's knowledge bases ({allowed_kbs})"
        info = message.to_dict() | (understanding.to_dict() if understanding else {}) | {"scope_action": scope_action}
        if route.intent == "out_of_scope":
            info["conversation_intent"] = convo.OUT_OF_SCOPE
        elif route.intent == "greeting":
            info["conversation_intent"] = convo.SMALL_TALK
        base = dict(session_id=session_id, message_id=None,
                    route=route.to_dict() | {"rewritten_query": standalone} | info
                    | ({"search_query": value_query} if value_query else {}))

        if route.intent == "greeting":
            first = not any(h.get("role") == "assistant" for h in history)
            return ChatResponse(answer=small_talk_reply(query, user.roles, first=first),
                                answer_type="closing" if is_closing(query) else "greeting", confidence=1.0,
                                **base)
        if route.intent == "out_of_scope":
            # "any alternative?" / "who are your competitors?": answer that point rather than the generic decline
            kind = convo.opinion_kind(original) or convo.opinion_kind(standalone)
            if kind and not route.reason.startswith("not in this assistant's knowledge bases"):
                base["route"]["refusal_kind"] = kind
                return ChatResponse(answer=convo.opinion_reply(kind), answer_type="out_of_scope", **base)
            return ChatResponse(answer=s.out_of_scope_text, answer_type="out_of_scope", **base)
        if route.intent == "clarify":
            return ChatResponse(answer=route.clarification, answer_type="clarification", **base)
        if route.intent == "access_denied":
            names = ", ".join(s.kb_config()["knowledge_bases"][k].get("display_name", k) for k in route.denied_kbs)
            if not user.authenticated:
                text = _replies().access_denied_guest.format(names=names)
            else:
                text = _replies().access_denied_user.format(names=names)
            base["route"]["guest"] = not user.authenticated      # guests are offered the sign-in (ui_action)
            return ChatResponse(answer=text, answer_type="access_denied", **base)

        # ---------------- golden fast path
        with stage("golden_match_ms"):
            golden_best = self._best_golden(expanded, route, user.roles, qvec)
            golden_query = expanded
            if expanded != standalone:  # synonym expansion can hide a word-for-word match with an approved question
                plain = self._best_golden(standalone, route, user.roles, self.embedder.embed_query(standalone))
                if plain and (golden_best is None or plain.score > golden_best.score):
                    golden_best, golden_query = plain, standalone
        if golden_best and allowed_kbs is not None and golden_best.kb not in allowed_kbs:
            golden_best = None   # an approved answer from another assistant's KB must never be used here
        golden_info = None
        if golden_best:
            golden_info = {"qid": golden_best.qid, "score": round(golden_best.score, 3), "kb": golden_best.kb,
                           "matched_question": golden_best.matched_question, "components": golden_best.components}
        # an approved answer wins - also for "difference between X and Y" questions when the approved question is a
        # near-exact match (e.g. an official FAQ "What is the difference between <Product> and <Competitor>?")
        if golden_best and GoldenMatcher.is_direct(golden_best) \
                and (route.intent != "comparison" or golden_best.score >= GOLDEN_COMPARISON_MIN) \
                and self._golden_covers(golden_query, golden_best):
            return self._golden_response(golden_best, golden_info, base)

        # ---------------- multi-part questions: each part is also retrieved on its own (backend/routing/decompose.py)
        sub_questions: list[str] = []
        planned = list(dict.fromkeys(extra_queries + (understanding.search_queries if understanding else [])))
        if route.intent != "comparison":
            if len(planned) > 1 or extra_queries:
                # several questions in one message: the normalizer / follow-up rewrite already split them (max 3),
                # so no separate decomposition call is needed
                sub_questions = [q for q in planned if q.strip().lower() != standalone.strip().lower()][:3]
                if extra_queries and standalone not in sub_questions:
                    sub_questions = [standalone] + sub_questions
            else:
                with stage("decompose_ms"):
                    sub_questions = decompose(standalone, self.llm)

        # ---------------- retrieval + rerank
        # phrasings for multi-query hybrid search: the plain question (synonym expansion can dilute it) and the LLM
        # clean-up's search queries when it ran (typos, Hindi / Hinglish, vague wording)
        variants = [standalone] if standalone != expanded else []
        if understanding and len(understanding.search_queries) <= 1:
            variants += [self.router.expand(q) for q in understanding.search_queries]
        with stage("retrieval_ms"):
            candidates = self._retrieve(expanded, route, user.roles, qvec, variants=variants)
            sub_pools = self._retrieve_sub_questions(sub_questions, route, user.roles, candidates)
        if golden_best and GoldenMatcher.is_assist(golden_best) and golden_best.chunk.id not in {c.id for c in candidates}:
            candidates.append(Candidate(chunk=golden_best.chunk, kb=golden_best.kb, fused_score=0.0))
        # further approved answers that clear the assist threshold compete in re-ranking too: their Q&A text is
        # usually the most precise source for paraphrased and broad questions
        with stage("golden_match_ms"):
            present = {c.id for c in candidates}
            for m in self._golden_assists(expanded, route, user.roles, qvec, exclude=golden_best):
                if m.chunk.id not in present:
                    candidates.append(Candidate(chunk=m.chunk, kb=m.kb, fused_score=0.0))
                    present.add(m.chunk.id)
        with stage("rerank_ms"):
            reranked = self.reranker.rerank(expanded, candidates) if candidates else []
        if route.entities and route.intent != "comparison":
            self._apply_entity_focus(reranked, route.entities)
        # paraphrased question: the approved answer means the same and the re-ranker independently ranks it first
        semantic_golden = self._semantic_golden_ok(golden_best, reranked[0] if reranked else None)
        if semantic_golden and self.llm is None:
            return self._golden_response(golden_best, golden_info, base, answer_type="golden_semantic")
        if semantic_golden and golden_best.chunk.id not in {c.id for c in reranked}:
            # LLM mode: the approved answer becomes top-priority context instead of a verbatim reply, so the model
            # answers the question actually asked (a paraphrase match can be about a neighbouring question)
            reranked.insert(0, Candidate(chunk=golden_best.chunk, kb=golden_best.kb, rerank_score=golden_best.score))
        if golden_best and GoldenMatcher.is_assist(golden_best):
            for c in reranked:
                if c.id == golden_best.chunk.id:
                    c.rerank_score = max(c.rerank_score, golden_best.score)
            reranked.sort(key=lambda c: c.rerank_score, reverse=True)
        k = s.top_k_rerank + (2 if route.intent == "comparison" else 0)
        top = [c for c in reranked if c.rerank_score >= s.min_rerank_score][:k]
        if route.intent == "comparison":
            top = self._ensure_entity_coverage(top, reranked, route.entities)
        if sub_pools:
            with stage("rerank_ms"):
                top = self._ensure_sub_question_coverage(top, reranked, sub_pools)
        retrieved_dbg = [{"id": c.id, "kb": c.kb, "score": round(c.rerank_score, 3), "doc_name": c.meta.get("doc_name"),
                          "source_file": c.meta.get("source_file"), "section": c.meta.get("section"),
                          "qid": c.meta.get("qid", ""), "product_name": c.meta.get("product_name", ""),
                          "entity_name": c.meta.get("entity_name", "")}
                         # reported in context order (what the model reads first), then the rest of the re-ranking
                         for c in (top + [r for r in reranked if r.id not in {t.id for t in top}])[:10]]
        kbs_used = sorted({c.kb for c in top}) or route.knowledge_bases
        clarify = self._broad_topic_clarification(convo.strip_social(original), history, top, reranked) \
            or self._broad_topic_clarification(query, history, top, reranked)
        if clarify:
            base["route"]["broad_topic"] = True
            return ChatResponse(answer=clarify, answer_type="clarification", knowledge_bases=kbs_used,
                                retrieved=retrieved_dbg, **base)
        if route.ambiguous and (not top or top[0].rerank_score < AMBIGUOUS_MIN_SCORE or self._insufficient(expanded, top)):
            return ChatResponse(answer=route.clarification, answer_type="clarification", knowledge_bases=kbs_used,
                                retrieved=retrieved_dbg, **base)
        if route.ambiguous and top:
            kbs_used = sorted({c.kb for c in top[:3]})
        if not top or self._insufficient(expanded, top):
            # an opinion / comparison / worry ("any negatives?", "any alternative", "I assume lots of problems"):
            # answer the point the user raised instead of the bare "no confirmed information" sentence
            kind = convo.opinion_kind(original) or convo.opinion_kind(standalone)
            if kind:
                base["route"]["refusal_kind"] = kind
                return ChatResponse(answer=convo.opinion_reply(kind), answer_type="refusal", knowledge_bases=kbs_used,
                                    golden_match=golden_info, retrieved=retrieved_dbg, **base)
            return ChatResponse(answer=REFUSAL_TEXT, answer_type="refusal", knowledge_bases=kbs_used,
                                golden_match=golden_info, retrieved=retrieved_dbg, **base)

        with stage("context_ms"):
            refs = select_context(top, s.max_context_chars)
        comparison = route.intent == "comparison"
        llm_result: LLMResult | None = None
        llm_ui_action = "none"
        report: GroundingReport | None = None
        warnings: list[str] = []
        answer_type = "extractive"
        answer = ""
        if self.llm is not None:
            try:
                with stage("context_ms"):
                    system, user_prompt = build_answer_prompt(
                        standalone, refs, history, comparison, route.entities, original=original,
                        language=reply_language(original, understanding.language if understanding else None))
                llm_result = self.llm.complete(system, user_prompt, json_mode=s.structured_reply)
                structured = parse_reply(llm_result.text) if s.structured_reply else None
                if structured is not None:
                    llm_result.text, llm_ui_action = structured.reply, structured.ui_action
                    base["route"]["structured_reply"] = structured.parsed
                timings.set("llm_ttft_ms", llm_result.ttft_ms)
                timings.set("llm_generation_ms", llm_result.generation_ms)
                timings.set("llm_total_ms", llm_result.latency_ms)
                base["route"]["provider"] = getattr(llm_result, "provider_name", "") or llm_result.provider
                answer = remove_invalid_citations(normalize_citations(llm_result.text), {r.sid for r in refs})
                answer = remove_unknown_urls(answer, {str(r.metadata.get("source_url") or "") for r in refs})
                answer_type = "generated"
                if is_refusal(answer) and convo.has_premise(original):
                    # the question presupposes something no source mentions ("which of your offices in <city>"):
                    # ask once more, naming the assumption, so the model corrects it from the facts the sources do
                    # state (rule 2a) instead of declining; still declines when the sources say nothing on that point
                    # (the presupposition itself is the trigger - plain yes/no questions never match it)
                    assumed = self._terms_missing_from(original, refs)
                    note = PREMISE_NOTE.format(terms=", ".join(assumed)) if assumed else PREMISE_NOTE_GENERIC
                    retried = self.llm.complete(system, user_prompt + note, json_mode=s.structured_reply)
                    parsed = parse_reply(retried.text) if s.structured_reply else None
                    text = parsed.reply if parsed is not None else retried.text
                    text = remove_invalid_citations(normalize_citations(text), {r.sid for r in refs})
                    if text.strip() and not is_refusal(text):
                        warnings.append("wrong assumption corrected on retry"
                                        + (": " + ", ".join(assumed) if assumed else ""))
                        answer = remove_unknown_urls(text, {str(r.metadata.get("source_url") or "") for r in refs})
                        llm_result.prompt_tokens += retried.prompt_tokens
                        llm_result.completion_tokens += retried.completion_tokens
                undocumented = self._undocumented_terms(f"{query} {standalone}", answer)
                if undocumented:
                    # the question asks about something no indexed document mentions (e.g. "iOS", "Aadhaar") and the
                    # answer repeats it: that claim cannot come from the sources -> "no confirmed information"
                    warnings.append("answer asserts undocumented term(s): " + ", ".join(undocumented))
                    answer, answer_type = REFUSAL_TEXT, "refusal"
                uncounted = uncounted_numbers(f"{query} {standalone}", answer, refs)
                if uncounted:
                    # "How many universities use it?": the model counted a list itself - no stated total exists
                    warnings.append("answer states a count the sources do not: " + ", ".join(uncounted))
                    answer, answer_type = REFUSAL_TEXT, "refusal"
                with stage("grounding_ms"):
                    report = lexical_grounding(answer, refs, standalone, s.grounding_min_support)
                    if s.grounding_mode == "llm" and not report.is_refusal and report.supported:
                        report = llm_grounding(self.llm, answer, refs)
                if s.grounding_mode != "off" and not report.supported and report.unsupported_sentences                         and not report.unsupported_numbers:
                    # drop only the sentences the sources do not back; keep the answer if the rest passes the check
                    trimmed = trim_unsupported(answer, report)
                    trimmed_report = lexical_grounding(trimmed, refs, standalone, s.grounding_min_support)                         if trimmed else None
                    if trimmed_report is not None and trimmed_report.supported and not trimmed_report.is_refusal:
                        warnings.append(f"removed {len(report.unsupported_sentences)} unsupported sentence(s)")
                        answer, report = trimmed, trimmed_report
                if s.grounding_mode != "off" and not report.supported:
                    warnings.append("Generated answer failed grounding check; extractive fallback used")
                    answer = extractive_answer(expanded, refs, comparison, route.entities) + UNVERIFIED_NOTE
                    answer_type = "extractive_fallback"
            except LLMRefusal as exc:
                logger.warning("LLM declined to answer (%s)", exc)
                warnings.append("llm_refusal")
                answer, answer_type = REFUSAL_TEXT, "refusal"
            except AllProvidersDown as exc:     # failover chain exhausted: static reply + the best sources
                logger.error("All LLM providers down (%s) - static fallback reply", exc)
                warnings.append("llm_unavailable")
                product = current_product()
                answer = product.replies.service_unavailable.format(support=product.support_contact,
                                                                    faqs=product.examples_markdown)
                answer_type, report = "service_unavailable", None
            except Exception as exc:
                logger.exception("LLM generation failed; using extractive answer")
                warnings.append(f"LLM unavailable: {type(exc).__name__}")
                answer = ""
        undocumented_ask = re.match(r"^\s*(does|do|is|are|can|could|will|would|has|have)\b", query or "", re.IGNORECASE) \
            and self._undocumented_terms(query, query)
        if not answer and undocumented_ask:
            # yes / no question about something no document mentions ("Does it have an iOS app?"): quoting a loosely
            # related passage would read like a "yes" - say there is no confirmed information instead
            warnings.append("question asks about undocumented term(s): " + ", ".join(undocumented_ask))
            answer, answer_type = REFUSAL_TEXT, "refusal"
        if not answer:
            answer = extractive_answer(expanded, refs, comparison, route.entities)
            answer_type = "extractive"
        if report is None or answer_type != "generated":
            report = lexical_grounding(answer, refs, standalone, s.grounding_min_support)
        if is_refusal(answer):
            answer_type = "refusal"
            # the model (or the undocumented-term guard) declined an opinion / comparison / worry question:
            # reply to its point, as when nothing was retrieved
            kind = convo.opinion_kind(original) or convo.opinion_kind(standalone)
            if kind:
                answer = convo.opinion_reply(kind)
                base["route"]["refusal_kind"] = kind
        used =cited_ids(answer) or ([refs[0].sid] if answer_type != "refusal" else [])
        citations = self._citations(refs, used)
        confidence = round(min(1.0, top[0].rerank_score * (0.5 + 0.5 * report.faithfulness)), 3)
        resp = ChatResponse(answer=answer, answer_type=answer_type, knowledge_bases=kbs_used, citations=citations,
                            ui_action=llm_ui_action if answer_type == "generated" else "none",
                            golden_match=golden_info, grounding=report.to_dict(), confidence=confidence,
                            contains_sample_data=any(c.is_sample for c in citations), retrieved=retrieved_dbg,
                            warnings=warnings, model=llm_result.model if llm_result else "extractive", **base)
        if llm_result:
            resp.prompt_tokens, resp.completion_tokens = llm_result.prompt_tokens, llm_result.completion_tokens
            resp.cost_usd = round(llm_result.cost_usd(s), 6)
            resp.tokens_per_second = llm_result.tokens_per_second
        return resp

    # ================================================================== helpers
    def _record_metrics(self, resp: "ChatResponse", user: AuthenticatedUser, session_id: str, channel: str,
                        assistant: str, error: str) -> None:
        s, t = self.settings, resp.timings
        embed_tokens = 0
        if s.embedding_price_per_1m:   # only API embedders cost money; local models are free per request
            embed_tokens = max(1, len(str(resp.route.get("rewritten_query", "")).split()) * 2)
        embedding_cost = embed_tokens * s.embedding_price_per_1m / 1_000_000
        rerank_cost = (len(resp.retrieved) / 1000) * s.rerank_price_per_1k if s.rerank_price_per_1k else 0.0
        retrieval = t.get("retrieval_ms", 0.0)
        try:
            repository.write_request_metric(
                channel=channel, assistant=assistant, username=user.username, session_id=session_id or "",
                answer_type=resp.answer_type, knowledge_bases=",".join(resp.knowledge_bases),
                success=not error, error=error, model=resp.model or ("extractive" if not resp.prompt_tokens else ""),
                end_to_end_ms=resp.latency_ms, ttft_ms=t.get("response_ttft_ms", resp.latency_ms),
                llm_ttft_ms=t.get("llm_ttft_ms", 0.0), llm_generation_ms=t.get("llm_generation_ms", 0.0),
                llm_total_ms=t.get("llm_total_ms", 0.0), retrieval_ms=retrieval,
                vector_search_ms=t.get("vector_search_ms", 0.0), bm25_search_ms=t.get("bm25_search_ms", 0.0),
                rerank_ms=t.get("rerank_ms", 0.0), context_ms=t.get("context_ms", 0.0),
                tokens_per_second=resp.tokens_per_second, input_tokens=resp.prompt_tokens,
                output_tokens=resp.completion_tokens, llm_cost_usd=resp.cost_usd,
                embedding_cost_usd=round(embedding_cost, 8), rerank_cost_usd=round(rerank_cost, 8),
                total_cost_usd=round(resp.cost_usd + embedding_cost + rerank_cost, 8), timings=t,
                **self._request_log_fields(resp))
        except Exception:   # metrics must never break a user request
            logger.exception("Could not record request metrics")
        if s.metrics_enabled:
            try:
                from backend.observability.metrics import record_request

                record_request(resp, channel, current_product().product_id, error)
            except Exception:
                logger.exception("Could not update Prometheus metrics")

    @staticmethod
    def _conversation_state(session_id: str, history: list[dict], persist: bool) -> str:
        """Stored state of a persisted session; for client-supplied history (Open WebUI, evaluation): "closing" when
        the last assistant turn was the goodbye."""
        if persist:
            return repository.get_conversation_state(session_id)
        last = next((h for h in reversed(history) if h.get("role") == "assistant"), None)
        return "closing" if last and last.get("answer_type") in ("closing", "closed") else "active"

    @staticmethod
    def _apply_ui_action(resp: "ChatResponse") -> None:
        """Front-end action + conversation state from the answer type; an LLM-chosen action is kept for answers."""
        if resp.answer_type == "closing":
            resp.ui_action, resp.conversation_state = "show_rating", "closing"
            return
        resp.conversation_state = "active"
        if resp.answer_type == "escalation":
            resp.ui_action = "show_escalation"
        elif resp.answer_type == "access_denied" and resp.route.get("guest"):
            resp.ui_action = "show_login"
        elif resp.answer_type not in ("generated", "extractive_fallback"):
            resp.ui_action = "none"
        if resp.ui_action == "show_rating":          # only the goodbye shows the rating
            resp.ui_action = "none"

    @staticmethod
    def _versions(resp: "ChatResponse") -> dict[str, str]:
        try:
            from backend.indexing.versions import index_version_label, prompt_version_label

            return {"index_version": index_version_label(list(resp.knowledge_bases)),
                    "prompt_version": prompt_version_label()}
        except Exception:
            return {}

    def _cached_response(self, query: str, user: AuthenticatedUser, allowed_kbs: list[str] | None, qvec,
                         session_id: str) -> "ChatResponse | None":
        """Exact / semantic cache hit for the standalone question (RESPONSE_CACHE_ENABLED), else None."""
        try:
            with stage("cache_lookup_ms"):
                kbs = list(allowed_kbs) if allowed_kbs is not None else get_index_manager().kb_names()
                hit = response_cache.lookup(response_cache.begin(query, kbs, user.roles, qvec))
        except Exception:
            logger.exception("Response cache lookup failed - answering normally")
            response_cache.clear_pending()
            return None
        if hit is None:
            return None
        return ChatResponse(answer=hit["answer"], session_id=session_id, message_id=None,
                            answer_type=hit["answer_type"], knowledge_bases=hit["knowledge_bases"],
                            citations=[Citation(**c) for c in hit["citations"]], confidence=hit["confidence"],
                            golden_match=hit.get("golden_match"), grounding=hit.get("grounding"),
                            model=hit.get("model", ""), contains_sample_data=hit.get("contains_sample_data", False),
                            route=hit.get("route", {}) | {"cache_hit": True, "cache": hit["_cache"]})

    def _request_log_fields(self, resp: "ChatResponse") -> dict:
        """Full request logging: which index / prompt / provider produced the answer and what was retrieved."""
        from backend.indexing.versions import index_version_label, prompt_version_label

        route = resp.route or {}
        return {
            "index_version": index_version_label(list(resp.knowledge_bases)),
            "prompt_version": prompt_version_label(),
            "provider": str(route.get("provider") or (self.settings.effective_llm_provider if resp.prompt_tokens else ""))[:64],
            "intent": str(route.get("intent", ""))[:32],
            "cache_hit": bool(route.get("cache_hit")),
            "golden_hit": resp.answer_type.startswith("golden"),
            "refused": resp.answer_type in REFUSED_ANSWER_TYPES,
            "retrieved": [{"id": r.get("id"), "kb": r.get("kb"), "score": r.get("score")} for r in resp.retrieved[:10]],
            "message_id": resp.message_id,
        }

    # ------------------------------------------------------------------ conversational layer + scope
    def _undocumented_terms(self, question: str, answer: str) -> list[str]:
        """Specific terms of the question (a product, platform, acronym) that occur in NO indexed document but do
        appear in the generated answer.

        A word counts as specific when the user wrote it like a name or acronym ("iOS", "SLA", "Android" mid-sentence)
        or it is not an everyday English word ("uptime", "aadhaar"). Everyday words missing from the documents
        ("ordinary", "forgot") are paraphrases, not new facts - the grounding check handles those."""
        low = (answer or "").lower()
        terms: dict[str, None] = {}
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9]{2,}", question or ""):
            raw, w = m.group(0), m.group(0).lower()
            if w in terms or self._known_word(w):
                continue
            sentence_start = not (question or "")[: m.start()].strip() or \
                (question or "")[: m.start()].rstrip()[-1:] in ".?!:"
            name_like = any(c.isupper() for c in raw[1:]) or (raw[0].isupper() and not sentence_start) \
                or w in PLATFORM_TERMS
            if name_like or not is_common_word(w):
                terms[w] = None
        return [w for w in terms if re.search(rf"\b{re.escape(w)}\b", low) and not self._only_corrected(w, low)]

    def _terms_missing_from(self, question: str, refs) -> list[str]:
        """Name-like words of the question ("Bangalore", "AWS") that the retrieved SOURCES do not contain - what a
        presupposing question takes for granted that the model cannot find in front of it."""
        corpus = " ".join(str(getattr(r, "text", "") or "") for r in refs).lower()
        out: list[str] = []
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9]{2,}", question or ""):
            raw, w = m.group(0), m.group(0).lower()
            before = (question or "")[: m.start()].rstrip()
            name_like = (raw[0].isupper() and before and before[-1:] not in ".?!:") or any(c.isupper() for c in raw[1:])
            if name_like and w not in out and not re.search(rf"\b{re.escape(w)}", corpus):
                out.append(w)
        return out

    @staticmethod
    def _only_corrected(term: str, answer_low: str) -> bool:
        """True when every mention of `term` in the answer corrects a wrong assumption of the question rather than
        asserting something about it: "the data is stored on Microsoft Azure, not AWS [S1]", "only PDF files,
        not Word documents", "offices in Hyderabad ..., not in Bangalore". The documented fact must be in the same
        sentence (a source tag) - "it does not have an iOS app" still counts as an undocumented claim."""
        hits = 0
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer_low):
            if not re.search(rf"\b{re.escape(term)}\b", sentence):
                continue
            hits += 1
            t = re.escape(term)
            corrected = re.search(rf"(,|;|\(|—|-)\s*(and\s+)?not\s+(in\s+|on\s+|at\s+|an?\s+|the\s+)?{t}\b"
                                  rf"|\b(instead of|rather than)\s+(an?\s+|the\s+)?{t}\b"
                                  # "Bangalore is not among them / not listed": points to a list the sources give.
                                  # ("there is no X" / "X is not supported" stay unaccepted - they would let a made-up
                                  # negative answer a plain yes/no question that must be declined)
                                  rf"|\b{t}\b\s+(is|are)\s+not\s+(among|listed|one of|in the list|part of)", sentence)
            if not (corrected and re.search(r"\[s\d+\]", sentence)):
                return False
        return hits > 0

    @staticmethod
    def _broad_topic_clarification(query: str, history: list[dict], top: list[Candidate],
                                   ranked: list[Candidate] | None = None) -> str:
        """For a bare topic request ("Tell me about the review.") with no confident match, ask which of the
        matching documented topics the user means instead of answering one of them. "" = answer normally.
        Options: the best passage, then other re-ranked passages whose title mentions the topic."""
        topic = convo.broad_topic(query, history)
        if not topic or not top or top[0].rerank_score >= BROAD_TOPIC_CONFIDENT:
            return ""
        stems = {stem(w) for w in re.findall(r"[a-z]{3,}", topic.lower())}
        options: list[str] = []
        pool = list(top) + [c for c in (ranked or [])[:15] if c.id not in {t.id for t in top}]
        for i, c in enumerate(pool):
            title = str(c.meta.get("question") or c.meta.get("section") or "")
            title = re.sub(r"^(?:admin|user|verifier|faq)\s*:\s*", "", title.replace("*", ""), flags=re.I).strip()
            title = title.rstrip("?").strip()
            if title.isupper():
                title = title.capitalize()
            mentions = bool(stems & {stem(w) for w in re.findall(r"[a-z]{3,}", title.lower())})
            if title and len(title) <= 90 and (i == 0 or mentions) \
                    and title.lower() not in {o.lower() for o in options}:
                options.append(title)
            if len(options) == 3:
                break
        if len(options) < 2:
            return ""
        bullets = "\n".join(f"- {o[0].upper()}{o[1:]}" for o in options)
        return f"Could you tell me which part of **{topic}** you mean? For example:\n{bullets}"

    @staticmethod
    def _known_word(word: str) -> bool:
        """A common English function word or a term that occurs in an indexed knowledge base."""
        if word in STOPWORDS or word in convo.QUESTION_WORDS:
            return True
        token = stem(word.lower())
        manager = get_index_manager()
        return any(token in manager.get(kb).doc_freq for kb in manager.kb_names())

    def _conversational_response(self, message: convo.MessageIntent, query: str, user: AuthenticatedUser,
                                 session_id: str, history: list[dict] | None = None) -> ChatResponse | None:
        """Direct reply (no retrieval, no LLM) for messages that are not knowledge questions; None otherwise."""
        if message.needs_rag:
            return None
        route = {"intent": "greeting", "knowledge_bases": [], **message.to_dict()}
        if message.intent in (convo.GREETING, convo.SMALL_TALK, convo.CLOSING):
            first = not any(h.get("role") == "assistant" for h in (history or []))
            return ChatResponse(answer=small_talk_reply(query, user.roles, first=first, intent=message.intent),
                                session_id=session_id,
                                message_id=None, answer_type="closing" if is_closing(query) else "greeting",
                                confidence=1.0, route=route)
        if message.intent == convo.FRUSTRATION:
            return ChatResponse(answer=message.reply, session_id=session_id, message_id=None,
                                answer_type="frustration", confidence=1.0, route=route | {"intent": "frustration"})
        if message.intent in (convo.ACCOUNT, convo.ESCALATION):
            return ChatResponse(answer=message.reply, session_id=session_id, message_id=None,
                                answer_type="account" if message.intent == convo.ACCOUNT else "escalation",
                                confidence=1.0, route=route | {"intent": message.intent.lower()})
        if message.intent == convo.OUT_OF_SCOPE:
            return ChatResponse(answer=self.settings.out_of_scope_text, session_id=session_id, message_id=None,
                                answer_type="out_of_scope", route=route | {"intent": "out_of_scope"})
        # typo / incomplete input, vague question, gibberish: ask the user - never guess, never refuse
        return ChatResponse(answer=message.reply or convo.rephrase_text(), session_id=session_id, message_id=None,
                            answer_type="clarification", confidence=1.0, route=route | {"intent": "clarify"})

    def _lenient_scope(self, route: RouteDecision, understanding: Understanding | None, query: str, roles: set[str],
                       allowed_kbs: list[str] | None, qvec) -> str:
        """Replace the keyword-only out-of-scope decision with an LLM / semantic check (lenient).

        * keyword router says out of scope ("unknown subject" / no KB matched): ask the LLM scope classifier (reusing
          the normalization verdict when there is one). IN_SCOPE, or a retrieved passage scoring at least
          SCOPE_RETRIEVAL_MIN_SCORE, -> answer from the knowledge bases; OUT_OF_SCOPE -> refuse; UNCLEAR -> clarify.
        * router found a knowledge base but the LLM said OUT_OF_SCOPE: refuse only if retrieval also finds nothing.
        Deliberate exclusions (HR topics) and assistant scoping are never overridden. Returns the action taken."""
        s = self.settings
        reason = route.reason or ""
        verdict = understanding.scope if understanding else None
        kbs = [kb for kb in self.router.kb_cfgs() if route_allows(kb, roles) and (allowed_kbs is None or kb in allowed_kbs)]
        if route.intent in ("kb_query", "comparison"):
            if verdict == "OUT_OF_SCOPE" and self._scope_score(query, roles, kbs, qvec) < s.scope_retrieval_min_score:
                route.intent, route.reason = "out_of_scope", "scope check: clearly unrelated"
                return "llm_out_of_scope"
            return ""
        if route.intent != "out_of_scope" or not (reason.startswith(("unknown subject", "no knowledge base matched"))
                                                  or route.method == "llm") or not kbs:
            return ""
        if verdict is None and self.llm is not None and s.normalization_llm:
            judged = llm_understand(query, self.llm)
            verdict = judged[1] if judged else None
        if verdict == "OUT_OF_SCOPE":
            return "llm_out_of_scope"
        if verdict == "GREETING":
            route.intent, route.reason = "greeting", "scope check: greeting"
            return "greeting"
        if verdict == "IN_SCOPE" or self._scope_score(query, roles, kbs, qvec) >= s.scope_retrieval_min_score:
            route.intent, route.knowledge_bases = "kb_query", kbs
            route.reason = f"scope check: {'llm in scope' if verdict == 'IN_SCOPE' else 'relevant documents found'}"
            return "llm_in_scope" if verdict == "IN_SCOPE" else "retrieval_in_scope"
        if verdict == "UNCLEAR" and not convo.opinion_kind(query):
            # ("any alternative" stays out of scope: the out-of-scope path replies to that point, never "clarify?")
            route.intent, route.reason = "clarify", "scope check: unclear"
            route.clarification = convo.INCOMPLETE_REPLY
            return "unclear"
        return "out_of_scope"

    def _scope_score(self, query: str, roles: set[str], kbs: list[str], qvec) -> float:
        """Best reranked passage score for the query across the given knowledge bases (semantic scope evidence)."""
        candidates: list[Candidate] = []
        for kb in kbs:
            candidates += self.retriever.retrieve(query, kb, roles, top_n=10, query_vector=qvec)
        ranked = self.reranker.rerank(query, candidates) if candidates else []
        return ranked[0].rerank_score if ranked else 0.0

    def _golden_response(self, golden_best: GoldenMatch, golden_info: dict | None, base: dict,
                         answer_type: str = "golden_direct") -> ChatResponse:
        cand = Candidate(chunk=golden_best.chunk, kb=golden_best.kb, rerank_score=golden_best.score)
        ref = SourceRef(sid="S1", candidate=cand, is_golden=True, metadata=golden_best.chunk.metadata)
        return ChatResponse(answer=golden_best.answer, answer_type=answer_type, knowledge_bases=[golden_best.kb],
                            citations=self._citations([ref], ["S1"]), golden_match=golden_info,
                            confidence=round(golden_best.score, 3),
                            contains_sample_data=bool(golden_best.chunk.metadata.get("is_sample")),
                            retrieved=[{"id": cand.id, "kb": cand.kb, "score": round(golden_best.score, 3),
                                        "doc_name": cand.meta.get("doc_name"), "qid": golden_best.qid}], **base)

    @staticmethod
    def _semantic_golden_ok(golden_best: GoldenMatch | None, ce_top: Candidate | None) -> bool:
        """Use an approved answer for a paraphrased question when its meaning is very close (dense similarity), or
        close enough AND the re-ranker's own top source is that same approved Q&A / the same FAQ entry."""
        if golden_best is None or ce_top is None:
            return False
        dense = float((golden_best.components or {}).get("dense", 0.0))
        if dense >= SEMANTIC_GOLDEN_STRONG:
            return True
        question = str(golden_best.chunk.metadata.get("question", "")).lower().rstrip("?. ")
        section = str(ce_top.meta.get("section", "")).lower()
        same_source = ce_top.id == golden_best.chunk.id or (bool(question) and question in section)
        return same_source and ce_top.rerank_score >= SEMANTIC_GOLDEN_RERANK and dense >= SEMANTIC_GOLDEN_MIN

    def _best_golden(self, query: str, route: RouteDecision, roles: set[str], qvec) -> GoldenMatch | None:
        """Best approved answer. Routed KBs are always searched; other accessible KBs only count when the
        match is strong enough for a direct answer (approved answers take priority over routing guesses)."""
        best: GoldenMatch | None = None
        for kb in get_index_manager().kb_names():
            if not get_index_manager().get(kb).golden or not (route_allows(kb, roles)):
                continue
            matches = self.golden.match(query, kb, roles, qvec, top_n=1)
            if not matches:
                continue
            m = matches[0]
            if kb not in route.knowledge_bases and not GoldenMatcher.is_direct(m):
                continue
            if best is None or m.score > best.score:
                best = m
        return best

    def _golden_assists(self, query: str, route: RouteDecision, roles: set[str], qvec,
                        exclude: GoldenMatch | None = None, n: int = GOLDEN_EXTRA_ASSISTS) -> list[GoldenMatch]:
        """Approved answers (routed, accessible KBs only) scoring >= the assist threshold, best first."""
        found: list[GoldenMatch] = []
        for kb in route.knowledge_bases:
            if not route_allows(kb, roles) or not get_index_manager().get(kb).golden:
                continue
            for m in self.golden.match(query, kb, roles, qvec, top_n=n + 1):
                if GoldenMatcher.is_assist(m) and (exclude is None or m.chunk.id != exclude.chunk.id):
                    found.append(m)
        return sorted(found, key=lambda m: m.score, reverse=True)[:n]

    def _retrieve_sub_questions(self, sub_questions: list[str], route: RouteDecision, roles: set[str],
                                candidates: list[Candidate]) -> dict[str, list[Candidate]]:
        """Retrieve every sub-question separately (plus its best approved answer) and add the results to
        `candidates`. Returns {sub_question: its own candidate pool} for the coverage check after re-ranking."""
        pools: dict[str, list[Candidate]] = {}
        by_id = {c.id: c for c in candidates}
        for sq in sub_questions:
            expanded = self.router.expand(sq)
            svec = self.embedder.embed_query(expanded)
            pool = self._retrieve(expanded, route, roles, svec, top_n=SUB_QUESTION_TOP_N)
            pool += [Candidate(chunk=m.chunk, kb=m.kb, fused_score=0.0)
                     for m in self._golden_assists(expanded, route, roles, svec, n=1)
                     if m.chunk.id not in {c.id for c in pool}]
            for c in pool:
                if c.id not in by_id:
                    by_id[c.id] = c
                    candidates.append(c)
            pools[sq] = pool
        return pools

    def _ensure_sub_question_coverage(self, top: list[Candidate], ranked: list[Candidate],
                                      pools: dict[str, list[Candidate]]) -> list[Candidate]:
        """Every part of a multi-part question gets at least one source: when none of a sub-question's two best
        chunks (re-ranked against the sub-question itself) made the context, its best one is added."""
        min_score = self.settings.min_rerank_score
        out, have = list(top), {c.id for c in top}
        by_id = {c.id: c for c in ranked}
        added = 0
        for sq, pool in pools.items():
            if not pool:
                continue
            # re-rank copies: the shared candidates keep their scores against the full question
            scored = [c for c in self.reranker.rerank(sq, [replace(c, debug=dict(c.debug)) for c in pool])
                      if c.rerank_score >= min_score]
            if not scored or any(c.id in have for c in scored[:2]):
                continue
            best = by_id.get(scored[0].id, scored[0])
            # a leading slot, right after the best source: for "difference between revoking and recalling" the Revoke
            # article answers half of the question, so it must not sit behind loosely related "difference" FAQs
            out.insert(min(len(out), 1 + added), best)
            added += 1
            have.add(best.id)
        return out

    def _retrieve(self, query: str, route: RouteDecision, roles: set[str], qvec, top_n: int = 25,
                  variants: list[str] | None = None) -> list[Candidate]:
        """Hybrid retrieval per routed KB. `variants`: further phrasings of the same question - when given (and
        MULTI_QUERY_RETRIEVAL is on) every phrasing is searched and the lists are fused (retrieve_multi)."""
        merged: dict[str, Candidate] = {}
        if route.intent == "comparison" and len(route.entities) >= 2:
            for entity in route.entities:
                others = [e for e in route.entities if e != entity]
                sub_query = query
                for o in others:
                    sub_query = sub_query.replace(o, " ")
                sub_query = f"{entity} {sub_query}"
                for kb in route.knowledge_bases:
                    for cand in self.retriever.retrieve(sub_query, kb, roles, top_n=15, entities=[entity]):
                        if cand.id not in merged or cand.fused_score > merged[cand.id].fused_score:
                            merged[cand.id] = cand
        else:
            phrasings = list(dict.fromkeys([query] + [v for v in (variants or []) if v and v.strip()]))
            phrasings = phrasings[: max(1, self.settings.multi_query_max)]
            multi = self.settings.multi_query_retrieval and len(phrasings) > 1
            vectors = [qvec] + [self.embedder.embed_query(v) for v in phrasings[1:]] if multi else None
            for kb in route.knowledge_bases:
                found = (self.retriever.retrieve_multi(phrasings, kb, roles, top_n=top_n, entities=route.entities,
                                                       query_vectors=vectors) if multi else
                         self.retriever.retrieve(query, kb, roles, top_n=top_n, entities=route.entities,
                                                 query_vector=qvec))
                for cand in found:
                    merged[cand.id] = cand
        return list(merged.values())

    @staticmethod
    def _ensure_entity_coverage(top: list[Candidate], ranked: list[Candidate], entities: list[str]) -> list[Candidate]:
        out = list(top)
        for entity in entities:
            if not any(c.meta.get("entity_name") == entity for c in out):
                extra = next((c for c in ranked if c.meta.get("entity_name") == entity), None)
                if extra:
                    out.append(extra)
        return out

    @staticmethod
    def _golden_covers(query: str, match: GoldenMatch, max_missing: float = GOLDEN_MAX_MISSING_INTENT) -> bool:
        """Safety check: the approved Q&A must cover the question's specific (high-IDF) terms, so an approved
        answer is never returned for a different question that merely looks similar."""
        m = match.chunk.metadata
        text = " ".join(str(m.get(k, "")) for k in ("question", "alternates", "keywords", "answer"))
        if weighted_coverage(query, text, match.kb) < 0.5:
            return False
        # The approved QUESTION side (question, rephrasings, keywords) must also carry the query's intent: e.g.
        # "What are the BENEFITS of blockchain certificates?" must not reuse "What is a blockchain certificate?".
        from backend.common.text_utils import content_terms
        terms = content_terms(query)
        q_side = content_terms(" ".join(str(m.get(k, "")) for k in ("question", "alternates", "keywords")))
        missing = terms - q_side
        if not terms or not missing:
            return True
        idx = get_index_manager().get(match.kb)
        weights = {t: idx.idf(t) for t in terms}
        return sum(weights[t] for t in missing) / (sum(weights.values()) or 1.0) <= max_missing

    @staticmethod
    def _apply_entity_focus(ranked: list[Candidate], entities: list[str]) -> None:
        """For questions about a specific product, prefer its chunks and demote chunks about other products."""
        wanted = {e.lower() for e in entities}
        for c in ranked:
            name = str(c.meta.get("entity_name", "")).lower()
            if name and name not in wanted:
                c.rerank_score *= 0.6
            elif c.entity_match:
                c.rerank_score = min(1.0, c.rerank_score + 0.1)
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)

    def _insufficient(self, query: str, top: list[Candidate]) -> bool:
        """Refuse when even the best chunks cover too little of the question's specific (high-IDF) terms."""
        if self.llm is not None and top and top[0].rerank_score >= LLM_CONFIDENT_RERANK:
            return False   # the reranker is confident; the LLM + grounding check decide (handles paraphrases)
        query = query.replace("(context:", "(")   # the follow-up marker word itself is not a term to cover
        best_cov = max(weighted_coverage(query, (c.meta.get("embed_text") or c.chunk.text), c.kb) for c in top[:3])
        threshold = self.settings.min_query_coverage_llm if self.llm is not None else self.settings.min_query_coverage
        return best_cov < threshold

    @staticmethod
    def _citations(refs: list[SourceRef], used: list[str]) -> list[Citation]:
        by_id = {r.sid: r for r in refs}
        out = []
        for sid in used:
            ref = by_id.get(sid)
            if not ref:
                continue
            m = ref.metadata
            page = m.get("page")
            out.append(Citation(
                id=sid, doc_name=str(m.get("doc_name", "")), kb=str(m.get("kb", "")),
                source_file=str(m.get("source_file", "")),
                page=int(page) if isinstance(page, (int, float)) and page and page > 0 else None,
                section=str(m.get("section", "")), doc_id=str(m.get("doc_id", "")),
                version=int(m["version"]) if str(m.get("version", "")).isdigit() else None, qid=str(m.get("qid", "")),
                snippet=(str(m.get("answer")) if ref.is_golden else ref.matched_text)[:300],
                score=round(ref.candidate.rerank_score, 3), is_sample=bool(m.get("is_sample")), is_golden=ref.is_golden,
                url=str(m.get("source_url") or ""), question=str(m.get("question") or "") if ref.is_golden else ""))
        return out


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service
