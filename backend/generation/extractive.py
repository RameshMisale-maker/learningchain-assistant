"""Extractive (no-LLM) answer generation.

Used when no LLM API key is configured, when the LLM call fails, or when an LLM answer
fails the grounding check. It never writes new facts: it only selects sentences,
catalogue fields or approved answers from the retrieved sources, with citations.
"""
from __future__ import annotations

import json
import re

from backend.common.text_utils import content_terms, split_sentences, tokenize
from backend.generation.grounding import REFUSAL_TEXT
from backend.generation.prompt_builder import SourceRef
from backend.retrieval.kb_index import get_index_manager

COMPARISON_FIELDS = ["category", "description", "key_features", "deployment_options", "pricing", "licensing",
                     "support", "integrations", "specifications", "current_version", "system_requirements"]
_FIELD_HINTS = {
    "pricing": {"price", "pric", "cost", "costs", "pricing", "expensive", "cheap", "fee", "subscription"},
    "deployment_options": {"deploy", "deployment", "cloud", "premise", "saas", "host"},
    "key_features": {"featur", "feature", "features", "capabl", "capabiliti", "capability", "function"},
    "specifications": {"spec", "specs", "specification", "specific", "capacity", "limit", "user", "uptime", "sla", "temperature"},
    "system_requirements": {"requirement", "requir", "ram", "cpu", "hardware", "browser"},
    "integrations": {"integr", "integration", "integrations", "connector"},
    "support": {"warranty"},
    "current_version": {"version", "release"},
    "licensing": {"licens", "licence", "license"},
}


def _idf_weights(terms: set[str], kb: str) -> dict[str, float]:
    idx = get_index_manager().get(kb)
    return {t: idx.idf(t) for t in terms}


def _score_sentence(sentence: str, weights: dict[str, float]) -> float:
    terms = content_terms(sentence)
    total = sum(weights.values()) or 1.0
    return sum(w for t, w in weights.items() if t in terms) / total


def _record(ref: SourceRef) -> dict | None:
    raw = ref.metadata.get("record_json")
    if not raw:
        return None
    try:
        return {str(k).strip().lower(): v for k, v in json.loads(raw).items()}
    except (TypeError, ValueError):
        return None


def _wanted_fields(question: str) -> list[str]:
    q_terms = set(tokenize(question)) | set(tokenize(question, do_stem=False))
    fields = [f for f, hints in _FIELD_HINTS.items() if q_terms & hints]
    return fields


_SKIP_PREFIXES = ("fictional", "sample /", "> fictional", "policy id", "version:", "product name:", "product code:",
                  "product id:", "product short name:", "is sample:", "effective date:",
                  # document metadata lines in the knowledge-base files
                  "source:", "product:", "article:", "applies to:", "(video walkthrough", "updated:", "last updated:")
_FAQ_RE = re.compile(r"\*{0,2}Q:\s*(.+?)\*{0,2}\s*\n+\s*\*{0,2}A:\s*(.+?)(?=\n\s*\*{0,2}Q:|\Z)", re.DOTALL)


_BULLET_GLYPHS = "❖➢✓✔•▪►◦●○■□–"
_TERMINAL = (".", "?", "!", ":", ";")


def _join_wrapped_lines(body: str) -> list[str]:
    """Lines of a chunk with soft-wrapped text rejoined. Slides and PDFs break one sentence over several lines
    ("...digital ledger that" / "records transactions, ..."); a line that does not end a sentence and is followed by
    a line starting in lower case is the same sentence."""
    out: list[str] = []
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            out.append("")
            continue
        prev = out[-1] if out else ""
        if (prev and not prev.endswith(_TERMINAL) and not prev.startswith(("|", "#"))
                and line[0].islower() and not line.startswith("|")):
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return out


def _is_title_line(line: str) -> bool:
    """A bare heading such as a slide title ("Brief towards Blockchain Technology"): short, no sentence punctuation,
    mostly capitalised words. It names a topic but says nothing, so it is never quoted as an answer."""
    words = re.findall(r"[A-Za-z][\w'-]*", line)
    if line.rstrip().endswith(":") and 0 < len(words) <= 4:
        return True                      # a label that introduces a list ("Key Characteristics:")
    # 3+ words: a short value such as "Microsoft Azure" or "PDF only" can be the answer itself
    if len(words) < 3 or len(words) > 7 or line.rstrip().endswith(_TERMINAL):
        return False
    return sum(w[0].isupper() for w in words) / len(words) >= 0.6


def _is_fragment(text: str) -> bool:
    """Pieces of a sentence broken across slide lines ('Select "Internal', 'University, Nanded', 'LNCT University,'):
    too short to stand alone, an unclosed quote, or a dangling comma. Quoting them reads as nonsense."""
    words = re.findall(r"\w+", text)
    return len(words) < 4 or text.count('"') % 2 == 1 or text.rstrip().endswith((",", "(", "-", "–"))


def _units(body: str) -> list[tuple[str, str]]:
    """Split a chunk into (text_to_score, text_to_show) units: FAQ pairs, table rows and sentences."""
    units: list[tuple[str, str]] = []
    for q, a in _FAQ_RE.findall(body):
        a = " ".join(a.split())
        units.append((f"{q} {a}", a))
    body = _FAQ_RE.sub("\n", body)
    for line in _join_wrapped_lines(body):
        line = line.strip().lstrip(_BULLET_GLYPHS).strip()
        if not line or set(line) <= set("|-: ") or _HEADING_RE.match(line):
            continue          # markdown headings name a section; they are not answer sentences
        if not line.startswith("|") and _is_title_line(line.strip(" -*#>")):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2:
                text = f"{cells[0]}: " + "; ".join(cells[1:])
                units.append((text, text))
            continue
        for sent in split_sentences(line):
            sent = sent.strip(" -*#>")
            if sent and not _is_fragment(sent):
                units.append((sent, sent))
    return units


def comparison_table(question: str, entities: list[str], refs: list[SourceRef]) -> str | None:
    records: dict[str, tuple[dict, str]] = {}
    for ref in refs:
        rec = _record(ref)
        name = ref.metadata.get("entity_name")
        if rec and name in entities and name not in records:
            records[name] = (rec, ref.sid)
    if len(records) < 2:
        return None
    fields = _wanted_fields(question) or ["category", "key_features", "deployment_options", "pricing", "support",
                                          "current_version"]
    names = [e for e in entities if e in records]
    lines = ["| Attribute | " + " | ".join(f"{n} [{records[n][1]}]" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for f in fields:
        values = [str(records[n][0].get(f) or "Not specified").replace("|", "/") for n in names]
        lines.append(f"| {f.replace('_', ' ').title()} | " + " | ".join(values) + " |")
    sample = any(str(records[n][0].get("is_sample", "")).lower() == "true" for n in names)
    note = "\n\n_Values come from the product catalogue" + (" (sample data)._" if sample else "._")
    return f"Here is a comparison of {', '.join(names)} based on the product catalogue:\n\n" + "\n".join(lines) + note


_GUIDE_META_RE = re.compile(r"^\s*(Article:|Applies to:|\(Video walkthrough:|Updated:|Last updated:)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
GUIDE_MAX_CHARS = 1800


_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
GUIDE_MIN_TITLE_SCORE = 0.4


def _blocks(text: str) -> list[tuple[str, list[str]]]:
    """Split chunk text into (heading, lines) blocks; lines before the first heading get heading ''."""
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in text.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            blocks.append((m.group(1).strip(), []))
        else:
            blocks[-1][1].append(line)
    return [b for b in blocks if b[0] or any(l.strip() for l in b[1])]


def guide_answer(question: str, refs: list[SourceRef], weights: dict[str, float]) -> str | None:
    """How-to / help-centre articles: when an article title in the best passage matches the question, return that
    article's own text (steps in order) plus links to the full article and video, instead of loose sentences."""
    top = refs[0]
    if top.is_golden:
        return None
    kb = str(top.metadata.get("kb", ""))
    if kb:  # query words that appear nowhere in the KB (e.g. "forgot" vs "forget") cannot match any title
        df = get_index_manager().get(kb).doc_freq
        weights = {t: w for t, w in weights.items() if df.get(t, 0) > 0} or weights
    label = str(top.metadata.get("section") or "")
    titles = [h for h, _ in _blocks(top.text) if h]
    if label and label.lower() not in {t.lower() for t in titles}:
        titles.append(label)  # continuation chunk: its heading sits in an earlier chunk
    if not titles:
        return None
    section = max(titles, key=lambda t: _score_sentence(t, weights))
    if _score_sentence(section, weights) < GUIDE_MIN_TITLE_SCORE:
        return None
    src = top.metadata.get("source_file")
    parts = sorted((r for r in refs if not r.is_golden and r.metadata.get("source_file") == src),
                   key=lambda r: int(r.metadata.get("chunk_index") or 0))
    article = video = None
    lines: list[str] = []
    seen: set[str] = set()
    for r in parts:
        blocks = _blocks(r.text)
        own = [b for h, b in blocks if h.lower() == section.lower()]
        if not own and str(r.metadata.get("section", "")).lower() == section.lower():
            own = [b for h, b in blocks if not h]  # continuation text before any heading
        for line in (l for b in own for l in b):
            s = line.strip()
            if not s or s.lower().rstrip(" ?.:") == section.lower().rstrip(" ?.:"):
                continue
            if _GUIDE_META_RE.match(s):
                url = _URL_RE.search(s)
                if url and s.lower().startswith("article:"):
                    article = article or url.group(0).rstrip(").")
                elif url:
                    video = video or url.group(0).rstrip(").")
                continue
            if s in seen:  # chunk overlap repeats lines
                continue
            seen.add(s)
            lines.append(s)
    if len(lines) < 2:
        return None
    body, used = [], 0
    for s in lines:
        if used + len(s) > GUIDE_MAX_CHARS:
            body.append("- ...")
            break
        body.append(s)
        used += len(s)
    title = re.sub(r"^(Admin|User|Verifier|FAQ|Release notes):\s*", "", section).strip()
    out = f"**{title}** [{top.sid}]\n\n" + "\n".join(body)
    links = []
    if article:
        links.append(f"Full guide: {article}")
    if video:
        links.append(f"Video walkthrough: {video}")
    return out + ("\n\n" + " · ".join(links) if links else "")


def extractive_answer(question: str, refs: list[SourceRef], comparison: bool = False,
                      entities: list[str] | None = None, max_sentences: int = 4) -> str:
    if not refs:
        return REFUSAL_TEXT
    if comparison and entities and len(entities) >= 2:
        table = comparison_table(question, entities, refs)
        if table:
            return table

    top = refs[0]
    kb = str(top.metadata.get("kb", ""))
    q_terms = content_terms(question)
    weights = _idf_weights(q_terms, kb) if kb else {t: 1.0 for t in q_terms}

    if top.is_golden:
        golden_text = " ".join(str(top.metadata.get(k, "")) for k in ("question", "alternates", "keywords"))
        if _score_sentence(golden_text, weights) >= 0.5:
            return f"{top.metadata.get('answer', '')} [{top.sid}]"

    if not comparison and _record(top) is None:
        guide = guide_answer(question, refs, weights)
        if guide:
            return guide

    # Prefer a catalogue record about the requested entity when the question targets a catalogue field
    wanted_entities = {e.lower() for e in (entities or [])}
    record_ref = next((r for r in refs[:3] if _record(r) is not None and (
        not wanted_entities or str(r.metadata.get("entity_name", "")).lower() in wanted_entities)), None)
    if record_ref is not None and (record_ref is top or _wanted_fields(question)):
        top = record_ref
    record = _record(top)
    if record:
        fields = _wanted_fields(question)
        name = (top.metadata.get("entity_name") or record.get("name") or record.get("title") or record.get("question")
                or top.metadata.get("doc_name"))
        if "answer" in record and "question" in record:
            return f"{record['answer']} [{top.sid}]"
        lines = []
        chosen = fields or [k for k in ("description", "key_features") if k in record] or list(record)[:6]
        for f in chosen:
            if record.get(f):
                lines.append(f"- **{f.replace('_', ' ').title()}:** {record[f]}")
        if not fields:  # also include any other field that directly matches the question terms
            for k, v in record.items():
                if k not in chosen and v and _score_sentence(f"{k} {v}", weights) >= 0.5:
                    lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        if lines:
            return f"**{name}** [{top.sid}]\n" + "\n".join(lines[:8])

    scored: list[tuple[float, int, int, str, str]] = []
    for r_i, ref in enumerate(refs[:3]):
        body = str(ref.metadata.get("answer", "")) if ref.is_golden else ref.text
        doc_name = str(ref.metadata.get("doc_name", ""))
        for u_i, (score_text, unit) in enumerate(_units(body)):
            low = unit.lower().rstrip(".")
            if len(unit) < 15 or low == doc_name.lower() or low.startswith(_SKIP_PREFIXES):
                continue
            # the document title is implicit context for every unit (e.g. product or policy name)
            score = (0.7 * _score_sentence(score_text, weights) + 0.3 * _score_sentence(f"{score_text} {doc_name}", weights)) * (1.0 - 0.15 * r_i)
            if score > 0:
                scored.append((score, r_i, u_i, unit, ref.sid))
    if not scored:
        return REFUSAL_TEXT
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][0]
    picked = [s for s in scored if s[0] >= best * 0.6][:max_sentences]
    picked.sort(key=lambda x: (x[1], x[2]))
    lines = []
    for _, _, _, unit, sid in picked:
        unit = unit if unit.endswith((".", "?", "!", ":")) else unit + "."
        lines.append(f"- {unit} [{sid}]")
    return "Based on the knowledge base:\n" + "\n".join(lines)
