"""Chat orchestration - the complete RAG execution path for one user question.

  validate & screen query -> conversational intent layer (greeting / small talk / typo / clarification /
  gibberish answered directly; follow-ups resolved from history) -> follow-up rewrite + abbreviation expansion
  -> route (intent + KBs + ACL) -> golden fast path (approved answer verbatim) -> hybrid retrieval per KB
  (per entity for comparisons) -> rerank -> CRAG relevance grading (+ rewrite & re-retrieve when nothing is
  relevant) -> context selection -> LLM generation (or extractive fallback)
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
                                          llm_grounding, normalize_citations, remove_invalid_citations,
                                          remove_unknown_urls)
from backend.generation.llm_client import LLMClient, LLMRefusal, LLMResult, get_llm
from backend.generation.prompt_builder import SourceRef, build_answer_prompt, select_context
from backend.retrieval.crag import CragReport, RelevanceGrader, apply_grades
from backend.retrieval.golden_matcher import GoldenMatch, GoldenMatcher
from backend.retrieval.hybrid import Candidate, HybridRetriever
from backend.retrieval.kb_index import get_index_manager
from backend.retrieval.reranker import Reranker, get_reranker, weighted_coverage
from backend.common.text_utils import STOPWORDS, stem
from backend.routing import intent as convo
from backend.routing.decompose import decompose
from backend.routing.normalize import Understanding, llm_understand, understand
from backend.routing.router import QueryRouter, RouteDecision
from backend.security.auth import AuthenticatedUser
from backend.security.prompt_guard import BLOCKED_MESSAGE, check_query, restricted_topic, restricted_topic_reply
from config.settings import Settings, get_settings
from ingestion.embedding import Embedder, get_embedder

logger = logging.getLogger(__name__)

GREETING_TEXT = ("Hello! I'm the Sapphirus AI Assistant. Ask me about our company, services, products or "
                 "policies, and I'll answer from the approved knowledge base with sources.")
OUT_OF_SCOPE_TEXT = ("I'm sorry, but that's outside what I can help with. I can answer questions about Sapphirus and "
                     "its LearningChain platform - for example issuing, sharing or verifying blockchain certificates, "
                     "the admin and user portals, or Sapphirus company information.")


CLOSING_RE = re.compile(r"^\s*(no|nope|nah|no\s*,?\s*thanks?|no\s*,?\s*thank\s*(you|u)|no\s+more(\s+questions?)?|"
                        r"nothing(\s+else)?|that'?s\s+(all|it)|that\s+is\s+(all|it)|i'?m\s+done|im\s+done|"
                        r"all\s+good|i'?m\s+good|bye+|goodbye|good\s*bye|see\s+(you|ya)|take\s+care)\s*[!.?]*\s*$",
                        re.IGNORECASE)
CLOSING_TEXT = ("Thank you for chatting with the Sapphirus Assistant! Before you go, please rate this conversation "
                "below - your feedback helps us improve. Have a nice day!")


def is_closing(text: str) -> bool:
    """The user is ending the conversation (e.g. "no" after "anything else?", "that's all", "bye")."""
    return bool(CLOSING_RE.match(text or ""))


def small_talk_reply(text: str, roles: set[str]) -> str:
    """Friendly reply for greetings / small talk, tailored to the kind of message and the user's access."""
    t = text.strip().lower()
    if is_closing(t):
        return CLOSING_TEXT
    topics = ("Sapphirus Systems - our company, services and products - and LearningChain - features, issuing and "
              "verifying blockchain certificates, and how-to guides")
    examples = ("e.g. *\"How do I verify a student certificate?\"*, *\"What are the key features of LearningChain?\"* "
                "or *\"How do I revoke a certificate?\"*")
    if re.match(r"^(yes|yeah|yep|yup|sure|ya|of course|go ahead)", t):
        return f"Great! What would you like to know? You can ask me about {topics}."
    if re.match(r"^(thanks|thank|thx|ty|great|cool|nice|awesome|perfect|ok|okay|got it)", t):
        return "You're welcome! Is there anything else you'd like to know about Sapphirus or LearningChain?"
    if re.match(r"^(bye|goodbye|good\s*bye|see you|see ya|take care)", t):
        return ("Thank you for visiting Sapphirus! Have a great day. If you need anything later, reach us at "
                "info@sapphirus.in or +91 96531 57225.")
    if re.match(r"^how are you", t):
        return f"I'm doing great, thank you for asking! How can I help you today? You can ask me about {topics}."
    if re.match(r"^(who are you|what are you|what is your name|are you)", t):
        return ("I'm the Sapphirus Assistant - a virtual assistant from Sapphirus Systems. I answer questions about "
                f"{topics}, using Sapphirus's approved information and showing the sources.")
    if re.match(r"^(what can you do|how can you help|help)", t):
        return f"I can answer questions about {topics}. Try asking, {examples}."
    m = re.match(r"^good\s+(morning|afternoon|evening|day)", t)
    hello = f"Good {m.group(1)}!" if m else "Hello!"
    return f"{hello} Welcome to Sapphirus Systems. I'm the Sapphirus Assistant. How can I help you today? You can ask me about {topics}."
CONTACT_HINT = (" Our team will be happy to help - please contact us at info@sapphirus.in, call +91 96531 57225, "
                "or use https://sapphirus.in/contacts/.")
AMBIGUOUS_MIN_SCORE = 0.35
LLM_CONFIDENT_RERANK = 0.5        # with an LLM, skip the lexical pre-refusal when the top passage scores this high
GOLDEN_MAX_MISSING_INTENT = 0.35   # max IDF share of query terms absent from the approved question side
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
        self.llm: LLMClient | None = get_llm() if llm is True else (llm or None)
        self.retriever = retriever or HybridRetriever(self.settings, embedder=self.embedder)
        self.reranker = reranker or get_reranker()
        self.golden = golden or GoldenMatcher(self.embedder)
        self.router = router or QueryRouter(self.settings, self.embedder, llm=self.llm)
        self.grader = RelevanceGrader(self.llm, self.settings) if self.llm is not None else None

    # ================================================================== public API
    def answer(self, question: str, user: AuthenticatedUser, session_id: str | None = None,
               persist: bool = True, ip_address: str = "", history: list[dict] | None = None,
               allowed_kbs: list[str] | None = None, channel: str = "api", assistant: str = "all",
               record_metrics: bool | None = None) -> ChatResponse:
        """Answer one question.

        history      : prior turns [{"role", "content"}] supplied by the client (Open WebUI sends the whole
                       conversation); when omitted, history is loaded from the stored session.
        allowed_kbs  : restrict this assistant to these knowledge bases (e.g. the product-only chatbot); questions
                       routed elsewhere get the out-of-scope reply, so one assistant can never read another's KB.
        channel/assistant : labels for the performance metrics (dashboard filters).
        """
        started = time.perf_counter()
        session_id = repository.ensure_session(session_id, user.username) if persist else (session_id or "eval")
        if history is None:
            history = repository.get_history(session_id, user.username, limit=8) if persist else []
        error = ""
        with request_timings() as timings:
            try:
                response = self._answer(question, user, session_id, history, allowed_kbs, timings)
            except Exception as exc:   # never leak internals; record the failure for the dashboard
                logger.exception("Chat request failed")
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                response = ChatResponse(answer=self.settings.insufficient_info_text, session_id=session_id,
                                        message_id=None, answer_type="error", warnings=["internal_error"])
        if response.answer_type in ("refusal", "error"):
            # in scope, but no verified evidence: the fixed "insufficient information" reply, no sources
            response.answer, response.citations = self.settings.insufficient_info_text, []
        elif response.answer_type == "out_of_scope":
            response.answer, response.citations = self.settings.out_of_scope_text, []
        response.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        response.timings = timings.as_dict() | {"end_to_end_ms": response.latency_ms}
        if record_metrics if record_metrics is not None else persist:
            self._record_metrics(response, user, session_id, channel, assistant, error)
        if persist:
            repository.add_message(session_id, "user", question[: self.settings.max_query_chars])
            response.message_id = repository.add_message(
                session_id, "assistant", response.answer, knowledge_bases=response.knowledge_bases,
                citations=[asdict(c) for c in response.citations], route_info=response.route,
                answer_type=response.answer_type, latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd)
            if response.answer_type in ("blocked", "access_denied"):
                repository.write_audit(user.username, f"chat_{response.answer_type}", ",".join(response.route.get("denied_kbs", [])),
                                       outcome="denied", details={"question": question[:300]}, ip_address=ip_address)
        return response

    # ================================================================== pipeline
    def _answer(self, question: str, user: AuthenticatedUser, session_id: str, history: list[dict],
                allowed_kbs: list[str] | None = None, timings: Timings | None = None) -> ChatResponse:
        s = self.settings
        timings = timings or Timings()
        check = check_query(question)
        if not check.allowed:
            text = BLOCKED_MESSAGE if check.reason == "prompt_injection" else (
                "Please enter a question." if check.reason == "empty" else
                f"Your question is too long. Please keep it under {s.max_query_chars} characters.")
            return ChatResponse(answer=text, session_id=session_id, message_id=None,
                                answer_type="blocked" if check.reason == "prompt_injection" else "invalid",
                                route={"intent": "blocked", "flags": check.flags}, warnings=check.flags)
        query = original = check.cleaned
        # query normalization (backend/routing/normalize.py): typos, short forms and product-name variants are fixed
        # ("wt is saphirus?" -> "what is Sapphirus?"); the normalized text is used from here on, the original only
        # for logging. Greetings / small talk skip it (fixed replies, no LLM call).
        understanding: Understanding | None = None
        if s.query_normalization and not convo.is_small_talk(query):
            with stage("normalize_ms"):
                understanding = understand(query, known=self._known_word,
                                           llm=self.llm if s.normalization_llm else None)
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
            direct = self._conversational_response(message, query, user, session_id)
            if direct is not None:
                if understanding:
                    direct.route.update(understanding.to_dict())
                return direct
        with stage("query_rewrite_ms"):
            standalone = self.router.rewrite_follow_up(query, history, force=message.intent == convo.FOLLOW_UP)
            expanded = self.router.expand(standalone)
        with stage("query_embedding_ms"):
            qvec = self.embedder.embed_query(expanded)
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
                    route=route.to_dict() | {"rewritten_query": standalone} | info)

        if route.intent == "greeting":
            return ChatResponse(answer=small_talk_reply(query, user.roles),
                                answer_type="closing" if is_closing(query) else "greeting", confidence=1.0,
                                **base)
        if route.intent == "out_of_scope":
            return ChatResponse(answer=OUT_OF_SCOPE_TEXT, answer_type="out_of_scope", **base)
        if route.intent == "clarify":
            return ChatResponse(answer=route.clarification, answer_type="clarification", **base)
        if route.intent == "access_denied":
            names = ", ".join(s.kb_config()["knowledge_bases"][k].get("display_name", k) for k in route.denied_kbs)
            if not user.authenticated:
                text = (f"Information about {names} is internal and available to Sapphirus employees only. "
                        "If you're interested in working with us, see current openings at https://sapphirus.in/careers/ "
                        "or write to info@sapphirus.in.")
            else:
                text = (f"This question relates to {names}, which your account is not authorised to access. "
                        "Please contact HR/IT if you need access.")
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
        # near-exact match (e.g. the official FAQ "What is the difference between LearningChain and DigiLocker?")
        if golden_best and GoldenMatcher.is_direct(golden_best) \
                and (route.intent != "comparison" or golden_best.score >= GOLDEN_COMPARISON_MIN) \
                and self._golden_covers(golden_query, golden_best):
            return self._golden_response(golden_best, golden_info, base)

        # ---------------- multi-part questions: each part is also retrieved on its own (backend/routing/decompose.py)
        sub_questions: list[str] = []
        if route.intent != "comparison":
            with stage("decompose_ms"):
                sub_questions = decompose(standalone, self.llm)

        # ---------------- retrieval + rerank
        with stage("retrieval_ms"):
            candidates = self._retrieve(expanded, route, user.roles, qvec)
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
                         for c in reranked[:10]]
        kbs_used = sorted({c.kb for c in top}) or route.knowledge_bases
        if route.ambiguous and (not top or top[0].rerank_score < AMBIGUOUS_MIN_SCORE or self._insufficient(expanded, top)):
            return ChatResponse(answer=route.clarification, answer_type="clarification", knowledge_bases=kbs_used,
                                retrieved=retrieved_dbg, **base)
        if route.ambiguous and top:
            kbs_used = sorted({c.kb for c in top[:3]})
        # Corrective RAG: the LLM grades the selected passages; irrelevant ones are dropped, and when nothing relevant
        # was found the question is rewritten and the knowledge base searched once more (backend/retrieval/crag.py)
        crag: CragReport | None = None
        if self.grader is not None and s.crag_enabled:
            with stage("crag_ms"):
                weak = not top or self._insufficient(expanded, top)
                top, crag = self._corrective_rag(standalone, route, user.roles, [] if weak else top)
            base["route"]["crag"] = crag.to_dict()
            if top:
                kbs_used = sorted({c.kb for c in top})
        if not top or (not (crag and crag.verified) and self._insufficient(expanded, top)):
            return ChatResponse(answer=REFUSAL_TEXT, answer_type="refusal", knowledge_bases=kbs_used,
                                golden_match=golden_info, retrieved=retrieved_dbg, **base)

        with stage("context_ms"):
            refs = select_context(top, s.max_context_chars)
        comparison = route.intent == "comparison"
        llm_result: LLMResult | None = None
        report: GroundingReport | None = None
        warnings: list[str] = []
        answer_type = "extractive"
        answer = ""
        if self.llm is not None:
            try:
                with stage("context_ms"):
                    system, user_prompt = build_answer_prompt(standalone, refs, history, comparison, route.entities)
                llm_result = self.llm.complete(system, user_prompt)
                timings.set("llm_ttft_ms", llm_result.ttft_ms)
                timings.set("llm_generation_ms", llm_result.generation_ms)
                timings.set("llm_total_ms", llm_result.latency_ms)
                answer = remove_invalid_citations(normalize_citations(llm_result.text), {r.sid for r in refs})
                answer = remove_unknown_urls(answer, {str(r.metadata.get("source_url") or "") for r in refs})
                answer_type = "generated"
                with stage("grounding_ms"):
                    report = lexical_grounding(answer, refs, standalone, s.grounding_min_support)
                    if s.grounding_mode == "llm" and not report.is_refusal and report.supported:
                        report = llm_grounding(self.llm, answer, refs)
                if s.grounding_mode != "off" and not report.supported:
                    warnings.append("Generated answer failed grounding check; extractive fallback used")
                    answer = extractive_answer(expanded, refs, comparison, route.entities) + UNVERIFIED_NOTE
                    answer_type = "extractive_fallback"
            except LLMRefusal as exc:
                logger.warning("LLM declined to answer (%s)", exc)
                warnings.append("llm_refusal")
                answer, answer_type = REFUSAL_TEXT, "refusal"
            except Exception as exc:
                logger.exception("LLM generation failed; using extractive answer")
                warnings.append(f"LLM unavailable: {type(exc).__name__}")
                answer = ""
        if not answer:
            answer = extractive_answer(expanded, refs, comparison, route.entities)
            answer_type = "extractive"
        if report is None or answer_type != "generated":
            report = lexical_grounding(answer, refs, standalone, s.grounding_min_support)
        if is_refusal(answer):
            answer_type = "refusal"
        used = cited_ids(answer) or ([refs[0].sid] if answer_type != "refusal" else [])
        citations = self._citations(refs, used)
        confidence = round(min(1.0, top[0].rerank_score * (0.5 + 0.5 * report.faithfulness)), 3)
        resp = ChatResponse(answer=answer, answer_type=answer_type, knowledge_bases=kbs_used, citations=citations,
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
                total_cost_usd=round(resp.cost_usd + embedding_cost + rerank_cost, 8), timings=t)
        except Exception:   # metrics must never break a user request
            logger.exception("Could not record request metrics")

    # ------------------------------------------------------------------ conversational layer + CRAG
    @staticmethod
    def _known_word(word: str) -> bool:
        """A common English function word or a term that occurs in an indexed knowledge base."""
        if word in STOPWORDS or word in convo.QUESTION_WORDS:
            return True
        token = stem(word.lower())
        manager = get_index_manager()
        return any(token in manager.get(kb).doc_freq for kb in manager.kb_names())

    def _conversational_response(self, message: convo.MessageIntent, query: str, user: AuthenticatedUser,
                                 session_id: str) -> ChatResponse | None:
        """Direct reply (no retrieval, no LLM) for messages that are not knowledge questions; None otherwise."""
        if message.needs_rag:
            return None
        route = {"intent": "greeting", "knowledge_bases": [], **message.to_dict()}
        if message.intent in (convo.GREETING, convo.SMALL_TALK):
            return ChatResponse(answer=small_talk_reply(query, user.roles), session_id=session_id, message_id=None,
                                answer_type="closing" if is_closing(query) else "greeting", confidence=1.0,
                                route=route)
        if message.intent == convo.OUT_OF_SCOPE:
            return ChatResponse(answer=OUT_OF_SCOPE_TEXT, session_id=session_id, message_id=None,
                                answer_type="out_of_scope", route=route | {"intent": "out_of_scope"})
        # typo / incomplete input, vague question, gibberish: ask the user - never guess, never refuse
        return ChatResponse(answer=message.reply or convo.REPHRASE_TEXT, session_id=session_id, message_id=None,
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
        if verdict == "UNCLEAR":
            route.intent, route.reason = "clarify", "scope check: unclear"
            route.clarification = convo.INCOMPLETE_REPLY.format(hint=convo.INCOMPLETE_HINTS["what"])
            return "unclear"
        return "out_of_scope"

    def _scope_score(self, query: str, roles: set[str], kbs: list[str], qvec) -> float:
        """Best reranked passage score for the query across the given knowledge bases (semantic scope evidence)."""
        candidates: list[Candidate] = []
        for kb in kbs:
            candidates += self.retriever.retrieve(query, kb, roles, top_n=10, query_vector=qvec)
        ranked = self.reranker.rerank(query, candidates) if candidates else []
        return ranked[0].rerank_score if ranked else 0.0

    def _corrective_rag(self, question: str, route: RouteDecision, roles: set[str],
                        top: list[Candidate]) -> tuple[list[Candidate], CragReport]:
        """CRAG: grade -> keep relevant passages; nothing relevant -> rewrite, re-retrieve (KB only), grade again."""
        s = self.settings
        if top:
            verdicts = self.grader.grade(question, top)
            if verdicts is None:                       # grader unavailable: fail open, keep the reranked passages
                return top, CragReport(action="grader_error", graded=len(top), relevant=len(top))
            kept, dropped = apply_grades(top, verdicts)
            if kept:
                return kept, CragReport(action="correct" if not dropped else "filtered", graded=len(top),
                                        relevant=len(kept), dropped_ids=dropped)
            report = CragReport(action="incorrect", graded=len(top), dropped_ids=dropped)
        else:
            report = CragReport(action="incorrect")
        if not s.crag_retry:
            return [], report
        rewritten = self.grader.rewrite(question)
        report.retried, report.rewritten_query = True, rewritten
        if not rewritten:
            return [], report
        expanded = self.router.expand(rewritten)
        qvec = self.embedder.embed_query(expanded)
        candidates = self._retrieve(expanded, route, roles, qvec)
        present = {c.id for c in candidates}
        for m in self._golden_assists(expanded, route, roles, qvec):
            if m.chunk.id not in present:
                candidates.append(Candidate(chunk=m.chunk, kb=m.kb, fused_score=0.0))
        reranked = self.reranker.rerank(expanded, candidates) if candidates else []
        retry_top = [c for c in reranked if c.rerank_score >= s.min_rerank_score][:s.top_k_rerank]
        verdicts = self.grader.grade(question, retry_top)   # graded against the user's ORIGINAL question
        if not verdicts:
            return [], report
        kept, dropped = apply_grades(retry_top, verdicts)
        report.graded += len(retry_top)
        report.dropped_ids += dropped
        if kept:
            report.action, report.relevant = "corrected", len(kept)
        return kept, report

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
        for sq, pool in pools.items():
            if not pool:
                continue
            # re-rank copies: the shared candidates keep their scores against the full question
            scored = [c for c in self.reranker.rerank(sq, [replace(c, debug=dict(c.debug)) for c in pool])
                      if c.rerank_score >= min_score]
            if not scored or any(c.id in have for c in scored[:2]):
                continue
            best = by_id.get(scored[0].id, scored[0])
            out.append(best)
            have.add(best.id)
        return out

    def _retrieve(self, query: str, route: RouteDecision, roles: set[str], qvec, top_n: int = 25) -> list[Candidate]:
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
            for kb in route.knowledge_bases:
                for cand in self.retriever.retrieve(query, kb, roles, top_n=top_n, entities=route.entities, query_vector=qvec):
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
