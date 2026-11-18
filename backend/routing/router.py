"""Intent classification and knowledge-base routing.

Hybrid router (ROUTER_MODE=hybrid, default) combines three cheap signals per KB:
  1. keywords  - configured routing keywords/phrases found in the query
  2. examples  - max embedding similarity to configured example questions
  3. corpus    - data-driven term->KB association computed from the indexed chunks
                 (adapts automatically when real documents replace the samples)
plus product/entity detection. An LLM classifier is consulted only for ambiguous or
low-confidence cases when an LLM is configured (ROUTER_MODE=llm forces it always;
ROUTER_MODE=rules never uses it).

Outputs a RouteDecision with intent: kb_query | comparison | greeting | out_of_scope |
clarify | access_denied.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from backend.common.text_utils import normalize_text, stem, tokenize
from backend.retrieval.kb_index import KBIndexManager, get_index_manager
from config.settings import Settings, get_settings
from ingestion.embedding import Embedder, get_embedder

logger = logging.getLogger(__name__)

GREETING_RE = re.compile(
    r"^\s*((hi+|hello+|hey+|hiya|helo|hai|yo|greetings|namaste|namaskar)(\s+(there|team|sapphirus|bot|all))?|"
    r"good\s+(morning|afternoon|evening|day)|"
    r"(thanks|thank\s*(you|u)|thx|ty)(\s+(a\s+lot|so\s+much|very\s+much))?|great|cool|nice|awesome|perfect|"
    r"ok(ay)?|got\s+it|bye+|goodbye|good\s*bye|see\s+(you|ya)|take\s+care|"
    r"no|nope|nah|no\s*,?\s*thanks?|no\s*,?\s*thank\s*(you|u)|no\s+more(\s+questions?)?|nothing(\s+else)?|"
    r"that'?s\s+(all|it)|that\s+is\s+(all|it)|i'?m\s+done|im\s+done|all\s+good|i'?m\s+good|"
    r"who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|are\s+you\s+(a\s+)?(bot|human|robot)|"
    r"how\s+are\s+you(\s+doing)?|what\s+can\s+you\s+do|how\s+can\s+you\s+help(\s+me)?|help)"
    r"\s*[!.?]*\s*$", re.IGNORECASE)
COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference(s)?\s+between|differ|better\s+than|"
                           r"which\s+is\s+better)\b", re.IGNORECASE)
FOLLOW_UP_RE = re.compile(r"^\s*(and|also|what\s+about|how\s+about|and\s+what\s+about|same\s+for|what\s+of)\b|"
                          r"\b(it|its|it's|they|them|their|this|that|those|these|the\s+same)\b", re.IGNORECASE)


@dataclass
class RouteDecision:
    intent: str
    knowledge_bases: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    clarification: str = ""
    denied_kbs: list[str] = field(default_factory=list)
    method: str = "hybrid"
    reason: str = ""
    ambiguous: bool = False
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"intent": self.intent, "knowledge_bases": self.knowledge_bases,
                "scores": {k: round(v, 3) for k, v in self.scores.items()}, "entities": self.entities,
                "clarification": self.clarification, "denied_kbs": self.denied_kbs, "method": self.method,
                "reason": self.reason, "ambiguous": self.ambiguous}


class QueryRouter:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 indexes: KBIndexManager | None = None, llm=None) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder()
        self.indexes = indexes or get_index_manager()
        self.llm = llm
        self._example_vecs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> dict:
        return self.settings.kb_config()

    def kb_cfgs(self) -> dict[str, dict]:
        return {k: v for k, v in self.config["knowledge_bases"].items() if v.get("enabled", True)}

    def routing_cfg(self) -> dict:
        return {"min_route_score": 0.12, "multi_kb_ratio": 0.75, "ambiguity_margin": 0.06, "weak_score": 0.30,
                "weights": {"keywords": 0.30, "examples": 0.30, "corpus": 0.40}, **self.config.get("routing", {})}

    # ------------------------------------------------------------------ query preprocessing
    def expand(self, query: str) -> str:
        expansions = self.config.get("query_expansions", {}) or {}
        out = query
        for abbr, full in expansions.items():
            out = re.sub(rf"\b{re.escape(abbr)}\b", f"{abbr} ({full})", out, flags=re.IGNORECASE)
        return out

    def rewrite_follow_up(self, query: str, history: list[dict], force: bool = False) -> str:
        """Make short follow-ups standalone ("what about sick leave?" -> uses the previous question).
        force=True: the conversation layer already identified a follow-up ("tell me more", "explain the second
        point"), so the rewrite runs even without a pronoun, and the previous answer is given in full."""
        prev_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
        if not prev_user:
            return query
        words = tokenize(query)
        if not force and (len(words) > 6 or not FOLLOW_UP_RE.search(query)):
            return query
        if self.llm is not None:
            try:
                from backend.generation.prompt_builder import format_history, load_prompt

                prompt = load_prompt("rewrite_prompt.txt").format(
                    history=format_history(history, 3, last_answer_chars=1500 if force else None), question=query)
                result = self.llm.complete("You rewrite questions. Output only the question.", prompt, max_tokens=120,
                                           temperature=0.0)
                rewritten = result.text.strip().strip('"')
                if 5 < len(rewritten) < 400:
                    return rewritten
            except Exception as exc:
                logger.warning("LLM query rewrite failed: %s", exc)
        if force:   # no LLM: "explain the second point" -> that point of the previous answer
            from backend.routing.intent import ELABORATE_RE, last_assistant_answer, ordinal_item

            item = ordinal_item(query, last_assistant_answer(history))
            if item:
                return f"{item} (context: {prev_user})"
            if ELABORATE_RE.match(query):   # "tell me more" / "can you elaborate?" -> the previous topic itself
                return prev_user
        return f"{query} (context: {prev_user})"

    # ------------------------------------------------------------------ signals
    def _keyword_score(self, query: str, cfg: dict) -> float:
        q_norm = " " + re.sub(r"[^a-z0-9 ]+", " ", normalize_text(query).lower()) + " "
        q_stems = {stem(t) for t in q_norm.split()}
        hits = 0.0
        for kw in cfg.get("routing", {}).get("keywords", []):
            kw_l = kw.lower().strip()
            if " " in kw_l:
                if f" {kw_l} " in q_norm:
                    hits += 1.5
            elif stem(kw_l) in q_stems:
                hits += 1.0
        return min(1.0, hits / 2.0)

    def _example_score(self, qvec: np.ndarray, kb: str, cfg: dict) -> float:
        examples = cfg.get("routing", {}).get("examples", [])
        if not examples:
            return 0.0
        if kb not in self._example_vecs:
            self._example_vecs[kb] = self.embedder.embed_documents(examples)
        sims = self._example_vecs[kb] @ np.asarray(qvec, dtype=np.float32)
        return float(max(0.0, min(1.0, sims.max())))

    def _foreign_subject(self, query: str, kbs: list[str], rcfg: dict) -> str | None:
        """A word that appears nowhere in the knowledge base, in a question that never mentions our domain
        (scope_anchor_terms), means the question is about something else -> return that word."""
        anchors = [str(a).lower() for a in rcfg.get("scope_anchor_terms", []) or []]
        low = query.lower()
        if not anchors or any(re.search(rf"\b{re.escape(a)}", low) for a in anchors):
            return None
        expansion_keys = {stem(k) for k in (self.config.get("query_expansions") or {}) if " " not in k}
        idxs = [self.indexes.get(kb) for kb in kbs]
        for term in set(tokenize(query)):
            if len(term) < 3 or term.isdigit() or term in expansion_keys:
                continue
            if all(i.doc_freq.get(term, 0) == 0 for i in idxs):
                return term
        return None

    def _corpus_scores(self, query: str, kbs: list[str]) -> tuple[dict[str, float], float]:
        terms = set(tokenize(query))
        idxs = {kb: self.indexes.get(kb) for kb in kbs}
        total_docs = sum(max(i.size, 1) for i in idxs.values())
        if not terms:
            return {kb: 0.0 for kb in kbs}, 0.0
        global_df = {t: sum(i.doc_freq.get(t, 0) for i in idxs.values()) for t in terms}
        max_idf = np.log(1 + total_docs)
        weights = {t: (np.log(1 + total_docs / (1 + global_df[t])) if global_df[t] else max_idf) for t in terms}
        total_w = sum(weights.values()) or 1.0
        known_w = sum(w for t, w in weights.items() if global_df[t])
        coverage = known_w / total_w
        scores = {}
        for kb, idx in idxs.items():
            acc = 0.0
            for t in terms:
                if not global_df[t]:
                    continue
                rates = {k: i.doc_freq.get(t, 0) / max(i.size, 1) for k, i in idxs.items()}
                denom = sum(rates.values()) or 1.0
                acc += weights[t] * rates[kb] / denom
            scores[kb] = (acc / known_w if known_w else 0.0) * coverage
        return scores, coverage

    # ------------------------------------------------------------------ main entry
    def route(self, query: str, roles: set[str], query_vector: np.ndarray | None = None) -> RouteDecision:
        if GREETING_RE.match(query):
            return RouteDecision(intent="greeting", method="rules", reason="greeting/small talk")
        rcfg = self.routing_cfg()
        blocked = [t for t in rcfg.get("out_of_scope_topics", []) or []
                   if re.search(rf"\b{re.escape(str(t))}\b", query, re.IGNORECASE)]
        if blocked:  # topics deliberately not covered (e.g. HR / internal) -> decline instead of guessing
            return RouteDecision(intent="out_of_scope", method="rules", reason=f"excluded topic: {blocked[0]}")
        cfgs = self.kb_cfgs()
        kbs = list(cfgs)
        foreign = self._foreign_subject(query, kbs, rcfg)
        if foreign:  # e.g. "How much does a car cost?" - same shape as a FAQ, but about something we don't cover
            return RouteDecision(intent="out_of_scope", method="rules", reason=f"unknown subject: {foreign}")
        weights = rcfg["weights"]
        qvec = query_vector if query_vector is not None else self.embedder.embed_query(query)

        corpus, coverage = self._corpus_scores(query, kbs)
        scores: dict[str, float] = {}
        signals: dict[str, dict] = {}
        entities: list[str] = []
        entity_kb: str | None = None
        for kb in kbs:
            kw = self._keyword_score(query, cfgs[kb])
            ex = self._example_score(qvec, kb, cfgs[kb])
            score = weights["keywords"] * kw + weights["examples"] * ex + weights["corpus"] * corpus[kb]
            found = self.indexes.get(kb).detect_entities(query) if cfgs[kb].get("entity_fields") else []
            if found:
                entities, entity_kb = found, kb
                score = max(score, 0.6)
            scores[kb] = score
            signals[kb] = {"keywords": round(kw, 3), "examples": round(ex, 3), "corpus": round(corpus[kb], 3)}

        comparison = bool(COMPARISON_RE.search(query))
        decision = self._decide(scores, rcfg, roles, cfgs)
        if decision.intent == "kb_query" and not entities and len(tokenize(query.split("(context:")[0])) <= 1 \
                and len(query.split()) <= 2:
            best_kb = max(scores, key=scores.get)
            example = (cfgs[best_kb].get("routing", {}).get("examples") or [""])[0]
            decision.intent = "clarify"
            decision.clarification = ("Could you add a little more detail to your question?" +
                                      (f" For example: \"{example}\"" if example else ""))
            decision.reason = "query too short"
        decision.entities, decision.signals = entities, {"per_kb": signals, "term_coverage": round(coverage, 3)}
        entity_kbs = [kb for kb in decision.knowledge_bases if cfgs[kb].get("entity_fields")]
        if comparison and decision.intent == "kb_query" and entity_kbs:
            decision.intent = "comparison"
            if len(entities) == 1:
                decision.intent = "kb_query"   # e.g. editions/plans of a single product
            elif not entities:
                names = sorted(set(self.indexes.get(entity_kb or entity_kbs[0]).entity_aliases.values()))
                decision.intent = "clarify"
                decision.clarification = ("Which products would you like me to compare?" +
                                          (f" Available products include: {', '.join(names)}." if names else ""))

        use_llm = self.llm is not None and (self.settings.router_mode == "llm" or (
            self.settings.router_mode == "hybrid" and (decision.intent in ("clarify", "out_of_scope") or decision.ambiguous)
            and decision.signals.get("term_coverage", 0) > 0.25 and not (comparison and len(entities) < 2)))
        if use_llm:
            decision = self._llm_route(query, roles, cfgs, decision)
        return decision

    def _decide(self, scores: dict[str, float], rcfg: dict, roles: set[str], cfgs: dict) -> RouteDecision:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_kb, best = ranked[0]
        if best < rcfg["min_route_score"]:
            return RouteDecision(intent="out_of_scope", scores=scores, reason="no knowledge base matched")
        selected = [kb for kb, s in ranked if s >= rcfg["min_route_score"] and s >= best * rcfg["multi_kb_ratio"]]
        if len(ranked) > 1:
            second = ranked[1][1]
            if best < rcfg["weak_score"] and best - second < rcfg["ambiguity_margin"]:
                # Ambiguous: search both KBs; the chat service asks for clarification only if retrieval is weak too.
                names = [cfgs[k].get("display_name", k) for k, _ in ranked[:2]]
                selected = list(dict.fromkeys(selected + [ranked[1][0]]))
                allowed = [kb for kb in selected if roles & set(cfgs[kb].get("allowed_roles", [])) or "admin" in roles]
                if allowed:
                    return RouteDecision(intent="kb_query", scores=scores, knowledge_bases=allowed, ambiguous=True,
                                         denied_kbs=[kb for kb in selected if kb not in allowed],
                                         clarification=f"Could you clarify whether your question is about {names[0]} "
                                                       f"or {names[1]}?", reason="ambiguous scores")
        allowed = [kb for kb in selected if roles & set(cfgs[kb].get("allowed_roles", [])) or "admin" in roles]
        denied = [kb for kb in selected if kb not in allowed]
        if not allowed:
            return RouteDecision(intent="access_denied", scores=scores, denied_kbs=denied,
                                 reason="user role not permitted for the matching knowledge base")
        return RouteDecision(intent="kb_query", knowledge_bases=allowed, scores=scores, denied_kbs=denied,
                             reason=f"best={best_kb}")

    def _llm_route(self, query: str, roles: set[str], cfgs: dict, fallback: RouteDecision) -> RouteDecision:
        from backend.generation.prompt_builder import load_prompt

        desc = "\n".join(f"- {k}: {v.get('description', '')}" for k, v in cfgs.items())
        try:
            data, _ = self.llm.complete_json("You classify questions. Output JSON only.",
                                             load_prompt("router_prompt.txt").format(kb_descriptions=desc, question=query))
        except Exception as exc:
            logger.warning("LLM routing failed (%s); using hybrid decision", exc)
            return fallback
        kbs = [k for k in data.get("knowledge_bases", []) if k in cfgs]
        if data.get("out_of_scope") or (not kbs and not data.get("needs_clarification")):
            return RouteDecision(intent="out_of_scope", scores=fallback.scores, method="llm", reason=data.get("reason", ""))
        if data.get("needs_clarification"):
            return RouteDecision(intent="clarify", scores=fallback.scores, method="llm",
                                 clarification=data.get("clarification_question") or fallback.clarification)
        allowed = [kb for kb in kbs if roles & set(cfgs[kb].get("allowed_roles", [])) or "admin" in roles]
        if not allowed:
            return RouteDecision(intent="access_denied", denied_kbs=kbs, scores=fallback.scores, method="llm")
        intent = "comparison" if COMPARISON_RE.search(query) and len(fallback.entities) >= 2 else "kb_query"
        return RouteDecision(intent=intent, knowledge_bases=allowed, scores=fallback.scores, entities=fallback.entities,
                             denied_kbs=[k for k in kbs if k not in allowed], method="llm", reason=data.get("reason", ""))
