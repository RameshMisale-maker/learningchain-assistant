"""Product config files: schema, loading and validation (fail fast).

One YAML file per product in PRODUCT_CONFIG_DIR (default `config/products/`). Adding a product = adding a file (plus its
knowledge base in `config/knowledge_bases.yaml`); no pipeline code changes. Invalid files raise ProductConfigError at
startup, naming the file and the field.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ProductConfigError(RuntimeError):
    """A product config file is missing, unreadable or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GlossaryTerm(_Strict):
    term: str = Field(min_length=1)
    variants: list[str] = Field(default_factory=list)


class Sources(_Strict):
    sitemaps: list[str] = Field(default_factory=list)
    include_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    faq_document: str = ""                                    # file the website FAQ crawl is written to
    document_header: str = ""                                 # line added under the title of crawled help articles
    help_centre_name: str = ""
    help_centre_categories: dict[str, list[str]] = Field(default_factory=dict)   # category -> [file, audience, summary]
    faq_alternates: dict[str, list[str]] = Field(default_factory=dict)          # FAQ question -> rephrasings


class Retrieval(_Strict):
    namespace: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank_top: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0, le=1)


class Escalation(_Strict):
    email: str
    phone: str = ""
    hours: str = ""
    contact_url: str = ""
    careers_url: str = ""


class ModelRef(_Strict):
    provider: str = "groq"
    model: str


class Models(_Strict):
    primary: ModelRef
    fallback: list[ModelRef] = Field(default_factory=list)


class UI(_Strict):
    title: str
    subtitle: str = ""
    placeholder: str = "Ask a question..."
    footer: str = ""
    launcher_label: str = "Ask us"
    icon: str = ""
    welcome: str = ""
    suggestions: list[str] = Field(default_factory=list)
    spinner: str = "Searching the documentation..."
    api_model_id: str                                       # model id on the OpenAI-compatible API (Open WebUI)
    api_model_name: str


class Replies(_Strict):
    """Fixed replies. Product-specific ones are required; product-neutral ones have defaults."""
    out_of_scope: str
    no_confirmed_information: str
    blocked: str
    restricted_topic: str
    greeting_first: str                                     # {hello} = "Hello!" / "Good morning!"
    who_are_you: str
    what_can_you_do: str
    bye: str
    closing: str
    account: str
    escalation: str
    frustration_again: str
    rephrase: str
    access_denied_guest: str                                # {names} = knowledge bases the user may not read
    greeting_again: str = "Hi again! How can I help?"
    greeting_again_timed: str = "{hello} How can I help?"
    yes: str = "Great - what would you like to know?"
    thanks: str = "You're welcome! Anything else I can help with?"
    acknowledged: str = "Glad that helps - anything else I can help with?"
    how_are_you: str = "I'm doing well, thanks for asking! How can I help?"
    frustration: str = ("Sorry you're having a frustrating time. What's going wrong? I'll do my best to help, or I can "
                        "connect you with our support team.")
    access_denied_user: str = ("This question relates to {names}, which your account is not authorised to access. "
                               "Please contact HR/IT if you need access.")
    repeat_alternatives: dict[str, str] = Field(default_factory=dict)   # answer_type -> second wording
    # opinion / comparison / worry messages the help content does not answer ({product}, {support} filled in);
    # they answer the point the user raised without inventing facts (no "there are no alternatives" claims)
    opinion_alternatives: str = ("I can only help with {product}, so I can't compare other products or recommend "
                                 "alternatives. If you tell me what you need, I'll check whether {product} covers it - "
                                 "or our team can advise you: {support}.")
    opinion_limitations: str = ("I don't have a published list of drawbacks or limitations for {product}, so I can't "
                                "give you a fair list of negatives. If you have a specific concern - for example "
                                "setup, cost, security or a feature you need - tell me and I'll check what's "
                                "documented, or talk to our team: {support}.")
    # "why?" right after a decline: the reason, per answer_type of that decline ({product}, {support} filled in)
    why_declined: dict[str, str] = Field(default_factory=lambda: {
        "blocked": ("Because that message asked me to set aside my rules or reveal my internal instructions, which I "
                    "can't do - they keep this chat safe and accurate for everyone. I'm happy to help with any "
                    "question about {product}."),
        "restricted": ("Because that topic needs a qualified professional (for example a doctor, financial adviser or "
                       "lawyer), and a wrong answer from me could cause harm. I can help with questions about "
                       "{product}."),
        "out_of_scope": ("Because I'm set up to answer questions about {product} only, so I can't give reliable answers "
                         "on other topics. Is there something about {product} I can help with?"),
        "refusal": ("Because I only answer from {product}'s published information, so that I never give you a wrong "
                    "answer - and that point isn't covered there. Our support team can confirm it for you: {support}."),
        "account": ("Because, for your privacy and security, I can't see or change personal account records in this "
                    "chat. You can see them after logging in to your account, or our support team can check: "
                    "{support}."),
        "escalation": ("Because refunds, complaints and similar issues need a person to look at your case, so our "
                       "support team handles them directly: {support}."),
        "access_denied": ("Because that information is restricted to authorised accounts. If you need access, please "
                          "contact your administrator or our support team: {support}."),
    })
    opinion_value: str = ("{product} can help you with {help_topics}. Whether it's right for you depends on what you "
                          "need - tell me your use case and I'll show you how {product} handles it, or talk to our "
                          "team: {support}.")
    opinion_concern: str = ("I understand the concern. Tell me what you're trying to do or what's going wrong with "
                            "{product}, and I'll walk you through it step by step. You can also reach our support "
                            "team: {support}.")
    # every LLM provider down (LLM_FAILOVER_ENABLED): {support} = support contact, {faqs} = common questions
    service_unavailable: str = ("I can't put together a full answer right now - our AI service is temporarily busy. "
                                "Please try again in a minute, ask one of the common questions ({faqs}), or contact "
                                "our support team: {support}.")


class ProductConfig(_Strict):
    product_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    display_name: str
    company: str                                            # full legal / brand name ("Example Systems")
    company_short: str                                      # how users write it ("Example")
    bot_name: str
    description: str
    covers: list[str] = Field(default_factory=list)        # what the assistant answers about (prompt wording)
    help_topics: str = ""                                  # system prompt: what users ask about
    aliases: list[str] = Field(default_factory=list)        # legacy `assistant` values that select this product
    domains: list[str] = Field(default_factory=list)        # request Host names that select this product
    kb: str                                                 # the product's own knowledge base
    namespaces: list[str]                                   # knowledge bases it may search
    scopes: dict[str, list[str]] = Field(default_factory=dict)   # assistant value -> knowledge bases
    sources: Sources = Field(default_factory=Sources)
    glossary: list[GlossaryTerm]
    glossary_prompt: str = ""                              # optional verbatim glossary block for the clean-up prompt
    domain_terms: list[str] = Field(default_factory=list)
    other_products: list[str] = Field(default_factory=list)    # names the normalizer should know
    topics: list[str]
    examples: list[str]
    incomplete_hints: dict[str, str] = Field(default_factory=dict)
    enabled_tools: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    retrieval: Retrieval
    golden_threshold: float = Field(default=0.78, ge=0, le=1)
    semantic_cache_threshold: float = Field(default=0.95, ge=0, le=1)
    escalation: Escalation
    models: Models
    prompt_version: str
    welcome_message: str
    ui: UI
    replies: Replies

    @field_validator("namespaces")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one knowledge base is required")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "ProductConfig":
        if self.kb not in self.namespaces:
            raise ValueError(f"kb '{self.kb}' must be one of namespaces {self.namespaces}")
        for scope, kbs in self.scopes.items():
            extra = set(kbs) - set(self.namespaces)
            if extra:
                raise ValueError(f"scope '{scope}' uses knowledge bases outside namespaces: {sorted(extra)}")
        return self

    # ------------------------------------------------------------------ helpers used by the pipeline
    @property
    def covered(self) -> list[str]:
        return self.covers or [self.display_name]

    def covers_text(self, joiner: str = "and") -> str:
        c = self.covered
        return c[0] if len(c) == 1 else ", ".join(c[:-1]) + f" {joiner} " + c[-1]

    def glossary_lines(self) -> str:
        """Glossary for the normalisation prompt: one '- Term (also written: ...)' line per term + other products."""
        lines = [f"- {g.term}" + (f" (also written: {', '.join(g.variants)})" if g.variants else "") for g in self.glossary]
        if self.domain_terms:
            lines.append("- Domain terms: " + ", ".join(self.domain_terms[:20]))
        if self.other_products:
            lines.append(f"- {self.company_short} products: " + ", ".join(self.other_products))
        return chr(10).join(lines)

    def alias_sentence(self) -> str:
        """'"leaning chain", "learning chain" etc. refer to X, and "saphirus" etc. refer to Y.'"""
        parts = [", ".join(f'"{v}"' for v in g.variants[:2]) + f" etc. refer to {g.term}" for g in self.glossary if g.variants]
        return ("; ".join(parts) + ".") if parts else ""

    def prompt_vars(self) -> dict[str, str]:
        """Values for the [[placeholders]] in prompts/*.txt."""
        return {"company": self.company_short, "company_full": self.company, "bot_name": self.bot_name,
                "covers": self.covers_text("and"), "covers_or": self.covers_text("or"),
                "covers_slash": " / ".join(self.covered), "topics": ", ".join(self.topics),
                "glossary": self.glossary_prompt or self.glossary_lines(), "aliases": self.alias_sentence(),
                "help_topics": self.help_topics or ", ".join(self.topics), "blocked_reply": self.replies.blocked,
                "product": self.display_name, "description": self.description}

    @property
    def support_contact(self) -> str:
        parts = [f"email {self.escalation.email}"] + ([f"call {self.escalation.phone}"] if self.escalation.phone else [])
        return " or ".join(parts)

    @property
    def support_contact_markdown(self) -> str:
        parts = [f"email **{self.escalation.email}**"] + (
            [f"call **{self.escalation.phone}**"] if self.escalation.phone else [])
        return " or ".join(parts)

    @property
    def names(self) -> list[str]:
        """Every name users may type for this product / company (lower-case), for greetings and patterns."""
        out = {self.display_name, self.company_short, self.company, self.bot_name, self.product_id}
        out |= {g.term for g in self.glossary}
        return sorted({n.lower() for n in out if n})

    @property
    def examples_markdown(self) -> str:
        quoted = [f'*"{e}"*' for e in self.examples[:3]]
        return ", ".join(quoted[:-1]) + (" or " + quoted[-1] if len(quoted) > 1 else quoted[0] if quoted else "")

    def allowed_kbs(self, scope: str | None = None) -> list[str]:
        """Knowledge bases a request may search: the named scope (legacy `assistant` value) or all namespaces."""
        return list(self.scopes.get(scope or "", self.namespaces))


# ---------------------------------------------------------------------------------------------- loading
_lock = threading.Lock()
_products: dict[str, ProductConfig] | None = None


def config_dir() -> Path:
    from config.settings import BASE_DIR

    raw = os.getenv("PRODUCT_CONFIG_DIR", "config/products")
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _parse(path: Path) -> ProductConfig:
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProductConfigError(f"{path.name}: cannot read YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise ProductConfigError(f"{path.name}: top level must be a mapping")
    try:
        cfg = ProductConfig(**data)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ProductConfigError(f"{path.name}: {details}") from exc
    if cfg.product_id != path.stem:
        raise ProductConfigError(f"{path.name}: product_id '{cfg.product_id}' must match the file name")
    return cfg


def load_products(validate_kbs: bool = True) -> dict[str, ProductConfig]:
    """Load and validate every product file (cached). Raises ProductConfigError on the first invalid file."""
    global _products
    with _lock:
        if _products is not None:
            return _products
        folder = config_dir()
        files = sorted(folder.glob("*.yaml")) if folder.exists() else []
        if not files:
            raise ProductConfigError(f"no product config files found in {folder}")
        products = {p.stem: _parse(p) for p in files}
        if validate_kbs:
            _validate_against_kbs(products)
        aliases: dict[str, str] = {}
        for pid, cfg in products.items():
            for alias in cfg.aliases:
                if alias in aliases:
                    raise ProductConfigError(f"{pid}.yaml: alias '{alias}' already used by {aliases[alias]}.yaml")
                aliases[alias] = pid
        _products = products
        return products


def _validate_against_kbs(products: dict[str, ProductConfig]) -> None:
    from config.settings import get_settings

    known = set(get_settings().kb_config().get("knowledge_bases", {}))
    for pid, cfg in products.items():
        missing = [kb for kb in cfg.namespaces if kb not in known]
        if missing:
            raise ProductConfigError(f"{pid}.yaml: namespaces {missing} are not defined in knowledge_bases.yaml")


def reset_products() -> None:
    global _products
    with _lock:
        _products = None


def default_product_id() -> str:
    products = load_products()
    wanted = os.getenv("DEFAULT_PRODUCT", "")
    if wanted:
        if wanted not in products:
            raise ProductConfigError(f"DEFAULT_PRODUCT '{wanted}' has no config file")
        return wanted
    return sorted(products)[0] if len(products) == 1 else _widest(products)


def _widest(products: dict[str, ProductConfig]) -> str:
    """Without DEFAULT_PRODUCT: the product that searches the most knowledge bases (the combined assistant)."""
    return max(sorted(products), key=lambda pid: len(products[pid].namespaces))


def product_for_kb(kb: str) -> ProductConfig:
    """The product whose own knowledge base is `kb` (else the first product that searches it, else the default)."""
    products = load_products()
    owner = next((p for p in products.values() if p.kb == kb), None) or \
        next((p for p in products.values() if kb in p.namespaces), None)
    return owner or get_product()


def get_product(product_id: str | None = None) -> ProductConfig:
    products = load_products()
    pid = product_id or default_product_id()
    if pid not in products:
        raise ProductConfigError(f"unknown product '{pid}'")
    return products[pid]
