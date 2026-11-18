"""Structure-aware chunking.

* Records (CSV/XLSX/JSON rows) and tables stay whole (split only if very long).
* Prose is split on Markdown / numbered headings first, then packed paragraph-by-
  paragraph up to `chunk_size` characters with sentence-level overlap.
* strategy="structure_recursive": heading tree (breadcrumb path in the header) + recursive splitting
  paragraph > line > sentence > clause > word, merged back up to `chunk_size` with whole-unit overlap.
* strategy="parent_child": structure_recursive parents, recursively split into small indexed children.
* Every chunk gets a contextual header (document + section) prepended to the text
  that is embedded, which improves retrieval for short chunks.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from backend.common.text_utils import split_sentences
from ingestion.document_loader import LoadedDocument, Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$|^((?:\d+\.)+\d*|[A-Z]\.)\s+([A-Z][^\n]{2,80})$|^([A-Z][A-Z0-9 &/,\-()]{4,80}):?$")


MIN_SECTION_CHARS = 220


@dataclass
class Chunk:
    chunk_id: str
    text: str                 # text shown to the LLM / user
    embed_text: str           # text used for embeddings + BM25 (with contextual header)
    page: int | None
    section: str
    kind: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _heading_of(line: str) -> str | None:
    m = _HEADING_RE.match(line.strip())
    if not m:
        return None
    if m.group(2):
        return m.group(2).strip()
    if m.group(4):
        return f"{m.group(3)} {m.group(4)}".strip()
    if m.group(5) and len(m.group(5).split()) <= 10:
        return m.group(5).strip().title()
    return None


def _split_by_headings(section: Section) -> list[tuple[str, str]]:
    """Return [(heading, text)] for a prose section."""
    parts: list[tuple[str, str]] = []
    heading = section.section
    buf: list[str] = []
    for line in section.text.split("\n"):
        h = _heading_of(line)
        if h:
            if "\n".join(buf).strip():
                parts.append((heading, "\n".join(buf).strip()))
            buf = [line]
            heading = h
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        parts.append((heading, "\n".join(buf).strip()))
    # merge very small sections into the following one so chunks keep useful context
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for head, text in parts:
        if carry:
            head, text = (head or carry[0]), carry[1] + "\n\n" + text
            carry = None
        if len(text) < MIN_SECTION_CHARS:
            carry = (head, text)
        else:
            merged.append((head, text))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + carry[1])
        else:
            merged.append(carry)
    return merged


def _split_long(text: str, size: int) -> list[str]:
    """Split text longer than `size` on sentence boundaries (hard split as a last resort)."""
    pieces: list[str] = []
    current = ""
    for sent in split_sentences(text) or [text]:
        while len(sent) > size:
            pieces.append(sent[:size])
            sent = sent[size:]
        if len(current) + len(sent) + 1 > size and current:
            pieces.append(current.strip())
            current = ""
        current += (" " if current else "") + sent
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _pack(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        units.extend(_split_long(para, size) if len(para) > size else [para])
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current.strip())
            tail = " ".join(split_sentences(current)[-3:])
            current = (tail[-overlap:] + "\n\n" if overlap and tail else "")
            if len(current) + len(unit) > size:
                current = ""
        current += ("\n\n" if current and not current.endswith("\n\n") else "") + unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ----------------------------------------------------------------------------------------------------------
# Structure-based + recursive chunking
# ----------------------------------------------------------------------------------------------------------
# Separators from coarsest to finest. A finer separator is only used for a piece that is still too long, so
# chunks break at the most natural boundary available (paragraph > line/list item > sentence > clause > word).
_SENTENCE = "<sentence>"
RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", _SENTENCE, "; ", ", ", " ")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+")
_NUMBERED_RE = re.compile(r"^((?:\d+\.)+\d*)\s+")


def _heading_level(line: str) -> int:
    """Nesting level of a heading line: '#'=1 ... '######'=6, '2.3.1'=3; other heading styles are leaves (7)."""
    stripped = line.strip()
    m = _MD_HEADING_RE.match(stripped)
    if m:
        return len(m.group(1))
    m = _NUMBERED_RE.match(stripped)
    if m:
        return len([p for p in m.group(1).split(".") if p])
    return 7


def _split_by_heading_tree(section: Section) -> list[tuple[str, str, str]]:
    """Return [(heading, heading_path, text)] for a prose section.

    heading_path is the breadcrumb of enclosing headings (e.g. "Admin Guide > Certificates > Revoke"), so a
    chunk deep inside a document still says where it belongs. Very small sections are merged into the next one.
    """
    parts: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []          # (level, heading) of the enclosing headings
    if section.section:
        stack.append((0, section.section))
    heading, buf = section.section, []

    def path() -> str:
        return " > ".join(h for _, h in stack)

    current_path = path()
    for line in section.text.split("\n"):
        h = _heading_of(line)
        if h:
            if "\n".join(buf).strip():
                parts.append((heading, current_path, "\n".join(buf).strip()))
            level = _heading_level(line)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, h))
            heading, current_path, buf = h, path(), [line]
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        parts.append((heading, current_path, "\n".join(buf).strip()))

    merged: list[tuple[str, str, str]] = []
    carry: tuple[str, str, str] | None = None
    for head, hpath, text in parts:
        if carry:   # a tiny section (e.g. a heading followed by one line) joins the next section
            head, hpath, text = (head or carry[0]), (hpath or carry[1]), carry[2] + "\n\n" + text
            carry = None
        if len(text) < MIN_SECTION_CHARS:
            carry = (head, hpath, text)
        else:
            merged.append((head, hpath, text))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + "\n\n" + carry[2])
        else:
            merged.append(carry)
    return merged


def _split_on(text: str, sep: str) -> list[str]:
    if sep == _SENTENCE:
        return split_sentences(text)
    if sep.strip():   # "; " / ", ": keep the punctuation with the left-hand piece
        pieces = text.split(sep)
        return [p.strip() + (sep.strip() if i < len(pieces) - 1 else "") for i, p in enumerate(pieces) if p.strip()]
    return [p.strip() for p in text.split(sep) if p.strip()]


def _joiner(sep: str) -> str:
    return sep if sep in ("\n\n", "\n") else " "


def _merge(pieces: list[str], joiner: str, size: int, overlap: int) -> list[str]:
    """Greedily pack pieces (each <= size) into chunks <= size. Consecutive chunks share whole trailing pieces
    of up to `overlap` characters, so overlap never cuts through a sentence or word."""
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len(joiner.join(current + [piece])) > size:
            chunks.append(joiner.join(current))
            # keep a tail of whole pieces as overlap, as long as it leaves room for the new piece
            while current and (len(joiner.join(current)) > overlap
                               or len(joiner.join(current + [piece])) > size):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(joiner.join(current))
    return chunks


def _absorb_fragments(pieces: list[str], size: int, joiner: str) -> list[str]:
    """Merge fragments shorter than size/4 (e.g. the last line of a list left over after packing) into the
    previous - or else the next - piece when the result still fits, so no chunk is a context-free scrap."""
    min_len = size // 4
    out: list[str] = []
    for piece in pieces:
        if out and (len(piece) < min_len or len(out[-1]) < min_len) \
                and len(out[-1]) + len(joiner) + len(piece) <= size:
            out[-1] = out[-1] + joiner + piece
        else:
            out.append(piece)
    return out


def recursive_split(text: str, size: int, overlap: int = 0,
                    separators: tuple[str, ...] = RECURSIVE_SEPARATORS) -> list[str]:
    """Recursive character splitting: try the coarsest separator first and only recurse into finer separators
    for pieces that are still longer than `size`. Every returned chunk is <= size characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        pieces = _split_on(text, sep)
        if len(pieces) > 1:
            break
    else:   # no separator left (e.g. one enormous token): hard split as a last resort
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step)]
    finer = separators[i + 1:]
    joiner = _joiner(sep)
    out: list[str] = []
    fitting: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            fitting.append(piece)
            continue
        if fitting:
            out.extend(_merge(fitting, joiner, size, overlap))
            fitting = []
        out.extend(recursive_split(piece, size, overlap, finer))
    if fitting:
        out.extend(_merge(fitting, joiner, size, overlap))
    out = _absorb_fragments(out, size, joiner)
    return out


def _parent_child(parents: list[Chunk], doc_id: str, title: str, child_size: int,
                  child_overlap: int) -> list[Chunk]:
    """Split every prose parent into small child chunks.

    Children are what gets embedded and BM25-indexed (small chunks match a question precisely); each child
    carries `parent_id` + `parent_text` so the answer context can be widened to the whole parent section.
    Records, tables and parents that already fit in one child are kept as a single chunk (no parent link).
    """
    children: list[Chunk] = []
    for p_idx, parent in enumerate(parents):
        parent_id = f"{doc_id}:p{p_idx:04d}"
        pieces = [parent.text] if parent.kind in ("record", "table") or len(parent.text) <= child_size \
            else recursive_split(parent.text, child_size, child_overlap)
        header = parent.embed_text.split("\n", 1)[0]   # the parent's contextual header (incl. heading path)
        for c_idx, piece in enumerate(pieces):
            idx = len(children)
            digest = hashlib.sha1(f"{doc_id}:{idx}:{piece}".encode()).hexdigest()[:12]
            meta = dict(parent.metadata)
            if len(pieces) > 1:
                meta.update(parent_id=parent_id, parent_text=parent.text, child_index=c_idx, child_count=len(pieces))
            children.append(Chunk(chunk_id=f"{doc_id}:{idx:04d}:{digest}", text=piece,
                                  embed_text=f"{header}\n{piece}", page=parent.page, section=parent.section,
                                  kind=parent.kind, chunk_index=idx, metadata=meta))
    return children


def chunk_document(doc: LoadedDocument, doc_id: str, chunk_size: int = 900, chunk_overlap: int = 150,
                   title: str | None = None, strategy: str = "structure",
                   child_size: int = 400, child_overlap: int = 80) -> list[Chunk]:
    """strategy="structure" (default): split on headings, then pack paragraphs with sentence overlap.
    strategy="fixed": recursive paragraph/sentence packing that ignores headings (benchmark baseline).
    strategy="structure_recursive": split on the heading tree (each chunk's header carries the heading path),
    then recursively split long sections paragraph > line > sentence > clause > word, merging pieces back up to
    `chunk_size` with whole-sentence/paragraph overlap.
    strategy="parent_child": structure_recursive parents of `chunk_size` chars, each recursively split into
    children of `child_size` chars (`child_overlap` overlap). Children are indexed; parents go to the LLM."""
    title = title or doc.metadata.get("title") or doc.name
    if strategy == "parent_child":
        parents = chunk_document(doc, doc_id, chunk_size, chunk_overlap, title=title, strategy="structure_recursive")
        return _parent_child(parents, doc_id, title, child_size, child_overlap)
    chunks: list[Chunk] = []

    def add(text: str, section: Section, heading: str, heading_path: str = "") -> None:
        idx = len(chunks)
        # heading breadcrumb without a leading copy of the document title (often the H1), which is already
        # in the header - a repeated title only dilutes the text the embedder and re-ranker score
        crumbs = [p for p in (heading_path or "").split(" > ") if p.strip()]
        while len(crumbs) > 1 and crumbs[0].strip().lower() == str(title).strip().lower():
            crumbs.pop(0)
        where = " > ".join(crumbs) or heading
        header = f"Document: {title}" + (f" | Section: {where}" if where else "") + \
                 (f" | Page: {section.page}" if section.page else "")
        digest = hashlib.sha1(f"{doc_id}:{idx}:{text}".encode()).hexdigest()[:12]
        chunks.append(Chunk(chunk_id=f"{doc_id}:{idx:04d}:{digest}", text=text, embed_text=f"{header}\n{text}",
                            page=section.page, section=heading, kind=section.kind, chunk_index=idx,
                            metadata=dict(section.metadata)))

    for section in doc.sections:
        if section.kind in ("record", "table"):
            pieces = [section.text] if len(section.text) <= chunk_size * 2 else _split_long(section.text, chunk_size)
            for piece in pieces:
                add(piece, section, section.section)
            continue
        if strategy == "fixed":
            for piece in _pack(section.text, chunk_size, chunk_overlap):
                add(piece, section, section.section or "")
            continue
        if strategy == "structure_recursive":
            for heading, hpath, text in _split_by_heading_tree(section):
                prev = next((c for c in reversed(chunks) if c.kind == "text"), None)
                if (len(text) < MIN_SECTION_CHARS and prev is not None
                        and len(prev.text) + len(text) + 2 <= chunk_size):
                    prev.text = f"{prev.text}\n\n{text}"
                    prev.embed_text = f"{prev.embed_text}\n\n{text}"
                    continue
                for piece in recursive_split(text, chunk_size, chunk_overlap):
                    add(piece, section, heading, hpath)
            continue
        for heading, text in _split_by_headings(section):
            prev = next((c for c in reversed(chunks) if c.kind == "text"), None)
            if (len(text) < MIN_SECTION_CHARS and prev is not None
                    and len(prev.text) + len(text) <= chunk_size):
                # tiny trailing fragment (e.g. text spilling onto a new PDF page): append to previous chunk
                prev.text = f"{prev.text}\n\n{text}"
                prev.embed_text = f"{prev.embed_text}\n\n{text}"
                continue
            for piece in _pack(text, chunk_size, chunk_overlap):
                add(piece, section, heading)
    return chunks
