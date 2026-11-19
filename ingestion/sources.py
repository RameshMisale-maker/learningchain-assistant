"""Structured content sources used by the automated ingestion pipeline.

* Zendesk Help Center  -> one Markdown file per help-centre category (article = "## " section with its URL and
  last-updated date), so a change in one article re-indexes only that category file.
* Product website FAQ page -> approved Q&A rows (golden dataset) + a searchable FAQ document.

All HTTP goes through ingestion.web.fetch (SSRF-validated, size-limited). Content is treated as untrusted data.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

from ingestion.web import fetch

def _source_product(product=None):
    """Product whose help-centre data is used: the given one, else the product that defines help-centre data."""
    from backend.products import current_product, load_products
    if product is not None:
        return product
    return next((p for p in load_products().values()
                 if p.sources.help_centre_categories or p.sources.faq_alternates), None) or current_product()


SKIP_TITLES = {"Welcome to your Help Centre!"}  # Zendesk placeholder article


class ToText(HTMLParser):
    """Minimal HTML -> Markdown-ish text: paragraphs, bullet/numbered lists, headings, bold, links, video links."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.lists, self.href = [], [], None

    def nl(self, n=1):
        self.out.append("\n" * n)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("p", "div", "section", "figure", "table", "tr"):
            self.nl()
        elif tag == "br":
            self.nl()
        elif tag in ("h1", "h2", "h3", "h4", "h5"):
            self.nl(2)
        elif tag in ("ul", "ol"):
            self.lists.append([tag, 0]); self.nl()
        elif tag == "li":
            self.nl()
            depth = max(len(self.lists) - 1, 0)
            if self.lists and self.lists[-1][0] == "ol" and depth == 0 and not self.has_step_labels:
                # "Step N:" (not "N.") so the chunker doesn't mistake numbered steps for section headings
                self.lists[-1][1] += 1
                self.out.append(f"Step {self.lists[-1][1]}: ")
            else:
                self.out.append("  " * depth + "- ")
        elif tag in ("strong", "b"):
            pass
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "a":
            self.href = a.get("href")
        elif tag == "iframe" and a.get("src"):
            src = a["src"].split("?")[0]
            self.nl(); self.out.append(f"(Video walkthrough: {src})"); self.nl()

    def handle_endtag(self, tag):
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr"):
            self.nl()
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self.nl()
        elif tag in ("strong", "b"):
            pass
        elif tag == "a":
            self.href = None

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data.replace("�", " ").replace("\xa0", " "))
        if text.strip() and self.href and self.href.startswith("http") and self.href not in text:
            text = f"{text} ({self.href})"
        self.out.append(text)


def to_text(body: str) -> str:
    p = ToText()
    # articles that already label their steps ("Step 1: ...") use numbered lists for sub-steps -> render as bullets
    p.has_step_labels = bool(re.search(r"Step\s*\d+\s*:", body or ""))
    p.feed(body or "")
    t = "".join(p.out)
    t = re.sub(r"\*\*\s*\*\*", "", t)                    # empty bold markers
    t = re.sub(r"\*\*(\s+)", r"\1**", t)                  # move spaces outside bold
    t = re.sub(r"(\S)(\s+)\*\*(?=\s|$)", r"\1**\2", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]*(?=[^\s-]|$)", "\n", t)          # strip stray indentation (keeps nested list indents)
    t = re.sub(r"^\s*-\s*$", "", t, flags=re.M)            # empty bullets
    t = re.sub(r"^(\d+)\.\s+(?=[A-Z])", r"Step \1: ", t, flags=re.M)   # numbered paragraphs outside lists
    t = re.sub(r"\n{3,}", "\n\n", t)
    return html.unescape(t).strip()




def zendesk_export(base_url: str, locale: str = "en-gb") -> dict[str, list]:
    """Download categories, sections and articles from the public Help Center API (all pages)."""
    base = base_url.rstrip("/")
    out: dict[str, list] = {}
    for name in ("categories", "sections", "articles"):
        url, items = f"{base}/api/v2/help_center/{locale}/{name}.json?per_page=100", []
        while url:
            data = json.loads(fetch(url).text)
            items += data[name]
            url = data.get("next_page")
        out[name] = items
    return out


def zendesk_to_markdown(export: dict[str, list], help_centre_url: str, product=None) -> dict[str, str]:
    """Convert a Help Center export to {file name: markdown}. Unknown categories get their own file."""
    cfg = _source_product(product)
    centre = cfg.sources.help_centre_name or f"{cfg.display_name} Help Centre"
    prefix = cfg.display_name.replace(" ", "") + "_"
    cats = {c["id"]: c for c in export["categories"]}
    secs = {s["id"]: s for s in export["sections"]}
    by_file: dict[str, list] = {}
    meta: dict[str, tuple[str, str]] = {}
    for a in export["articles"]:
        if a.get("draft") or a["title"].strip() in SKIP_TITLES:
            continue
        sec = secs.get(a["section_id"])
        if not sec:
            continue
        cat = cats.get(sec["category_id"], {}).get("name", "General").strip()
        fname, audience, intro = (tuple(cfg.sources.help_centre_categories.get(cat) or ()) or (
            prefix + re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_") + ".md", "", f"Help-centre articles: {cat}."))
        meta[fname] = (cat, intro)
        by_file.setdefault(fname, []).append((sec, a, audience))
    files: dict[str, str] = {}
    for fname, items in by_file.items():
        cat, intro = meta[fname]
        items.sort(key=lambda x: (x[0].get("position", 0), x[1].get("position", 0),
                                  -int(x[1]["id"]) if "Release" in x[0]["name"] else 0))
        title = cat.replace("I'm a Admin", "I'm an Admin")
        lines = [f"# {centre} - {title}", "", cfg.sources.document_header,
                 f"Source: {centre} {help_centre_url}", "", intro, ""]
        for sec, a, audience in items:
            t = re.sub(r"\s+", " ", a["title"]).strip().rstrip(":")
            heading = f"{audience.split(' (')[0]}: {t}" if audience else t
            if "Release" in sec["name"]:
                heading = f"Release notes: {t}"
            elif "FAQ" in sec["name"]:
                heading = f"FAQ: {t}"
            lines += [f"## {heading}", "", f"Article: {a['html_url']}", ""]
            updated = str(a.get("edited_at") or a.get("updated_at") or "")[:10]
            if updated:
                lines += [f"Updated: {updated}", ""]
            if audience:
                lines += [f"Applies to: {audience}.", ""]
            lines += [to_text(a["body"]) or "(No text in this article.)", ""]
        files[fname] = "\n".join(lines).rstrip() + "\n"
    return files


# ----------------------------------------------------------------------------- product website FAQ page
GOLDEN_COLS = ["Question ID", "Category", "Question", "Approved Answer", "Keywords", "Alternate Questions",
               "Access Roles", "Status", "Source URL"]
ROLES = "public;employee;hr;admin"



def clean(fragment: str) -> str:
    x = re.sub(r"<br\s*/?>", "\n", fragment)
    x = re.sub(r"</(p|li)>", "\n", x)
    x = re.sub(r"<li[^>]*>", "- ", x)
    x = re.sub(r"<[^>]+>", "", x)
    x = html.unescape(x).replace("\xa0", " ").replace("�", "'").replace("’", "'")
    return "\n".join(line.strip() for line in x.splitlines() if line.strip())


def parse(page: str) -> list[tuple[str, str]]:
    items = re.findall(r'<span class="acc-toggle"[^>]*>(.*?)<i class="down.*?<div class="acc-content">(.*?)</div>\s*</div>',
                       page, re.S)
    return [(clean(q), clean(a)) for q, a in items if clean(q) and clean(a)]


def alternates_for(question: str, product=None) -> list[str]:
    q = question.lower()
    alternates = _source_product(product).sources.faq_alternates
    return next((list(alts) for key, alts in alternates.items() if q.startswith(key)), [])


def keywords_for(question: str) -> str:
    stop = {"what", "is", "the", "a", "an", "of", "to", "do", "does", "how", "why", "we", "my", "i", "in", "on",
            "and", "for", "with", "can", "will", "after", "if", "have", "need", "using", "make", "makes", "than"}
    words = re.findall(r"[a-z][a-z\-]+", question.lower())
    return ", ".join(dict.fromkeys(w for w in words if w not in stop))




def faq_golden_csv(faqs: list[tuple[str, str]], url: str, existing_csv: str = "", product=None) -> str:
    """Approved-answer CSV: previous non-FAQ rows (P001...) are kept, FAQ rows F01.. are regenerated."""
    kept = []
    if existing_csv.strip():
        kept = [r for r in csv.DictReader(io.StringIO(existing_csv)) if not re.fullmatch(r"F\d+", r["Question ID"])]
    cfg = _source_product(product)
    rows = kept + [{"Question ID": f"F{i:02d}", "Category": f"{cfg.display_name} FAQ", "Question": q, "Approved Answer": a,
                    "Keywords": keywords_for(q), "Alternate Questions": " | ".join(alternates_for(q, cfg)),
                    "Access Roles": ROLES, "Status": "Approved", "Source URL": url}
                   for i, (q, a) in enumerate(faqs, 1)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=GOLDEN_COLS, quoting=csv.QUOTE_ALL, restval="", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def faq_document(faqs: list[tuple[str, str]], url: str, product=None) -> str:
    cfg = _source_product(product)
    lines = [f"# {cfg.display_name} - Frequently Asked Questions (official website)", "",
             cfg.sources.document_header,
             f"Source: {url}", ""]
    for q, a in faqs:
        lines += [f"## FAQ: {q}", "", a, ""]
    return "\n".join(lines).rstrip() + "\n"
