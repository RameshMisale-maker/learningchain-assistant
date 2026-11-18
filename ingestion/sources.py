"""Structured content sources used by the automated ingestion pipeline.

* Zendesk Help Center  -> one Markdown file per help-centre category (article = "## " section with its URL and
  last-updated date), so a change in one article re-indexes only that category file.
* LearningChain FAQ page -> approved Q&A rows (golden dataset) + a searchable FAQ document.

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

# category id -> (file name, audience label used in article headings, intro line)
CATEGORY_FILES = {
    "Introduction to LearningChain": ("LearningChain_01_Introduction.md", "", "What LearningChain is, why to choose it, key features and how it works for universities."),
    "Introduction to Blockchain Technology": ("LearningChain_02_Blockchain_Technology.md", "", "Blockchain basics and why blockchain-secured certificates and QR codes are safer."),
    "My Landing Page": ("LearningChain_03_Getting_Started.md", "", "Logging in, resetting a forgotten password and navigating the modules."),
    "I'm a User": ("LearningChain_04_User_Guide.md", "User (institution staff)", "Step-by-step guides for institution users: securing/generating certificates, reports, bulk emails, profile."),
    "I'm a Admin": ("LearningChain_05_Admin_Guide.md", "Admin", "Step-by-step guides for institution administrators: users, certificate types, student review, revoke/recall/approve, payments, credits, settings."),
    "I'm a Verifier": ("LearningChain_06_Verifier_Guide.md", "Verifier (employer / third party)", "How employers and third parties verify student certificates."),
    "General": ("LearningChain_07_FAQ_and_Release_Notes.md", "", "Frequently asked questions and product release notes (what's new in each release)."),
}
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


def zendesk_to_markdown(export: dict[str, list], help_centre_url: str) -> dict[str, str]:
    """Convert a Help Center export to {file name: markdown}. Unknown categories get their own file."""
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
        fname, audience, intro = CATEGORY_FILES.get(cat) or (
            "LearningChain_" + re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_") + ".md", "", f"Help-centre articles: {cat}.")
        meta[fname] = (cat, intro)
        by_file.setdefault(fname, []).append((sec, a, audience))
    files: dict[str, str] = {}
    for fname, items in by_file.items():
        cat, intro = meta[fname]
        items.sort(key=lambda x: (x[0].get("position", 0), x[1].get("position", 0),
                                  -int(x[1]["id"]) if "Release" in x[0]["name"] else 0))
        title = cat.replace("I'm a Admin", "I'm an Admin")
        lines = [f"# LearningChain Help Centre - {title}", "",
                 "Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.",
                 f"Source: LearningChain Help Centre {help_centre_url}", "", intro, ""]
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


# ----------------------------------------------------------------------------- LearningChain FAQ page
GOLDEN_COLS = ["Question ID", "Category", "Question", "Approved Answer", "Keywords", "Alternate Questions",
               "Access Roles", "Status", "Source URL"]
ROLES = "public;employee;hr;admin"

# Extra phrasings people use for each FAQ (matched by the start of the official question, lower-case).
ALTERNATES = {
    "what is learningchain?": ["Tell me about LearningChain", "What does LearningChain do?", "Explain LearningChain",
                               "What is the LearningChain platform?"],
    "what is blockchain technology?": ["What is blockchain?", "Explain blockchain", "How does blockchain work?"],
    "why do we need learningchain?": ["Why should we use LearningChain?", "What problems does LearningChain solve?",
                                      "What are the benefits of LearningChain?"],
    "how does blockchain make my documents tamper-proof?": ["How are certificates made tamper-proof?",
                                                            "Can blockchain certificates be tampered with?",
                                                            "How does blockchain prevent tampering?"],
    "where does learningchain store my data?": ["Where is the data stored?", "Which cloud does LearningChain use?",
                                                "Is LearningChain hosted on Azure?"],
    "how does learningchain work?": ["How does the LearningChain process work?", "Explain how LearningChain works",
                                     "How are certificates secured in LearningChain?"],
    "how is a blockchain qr code safer": ["Why is a blockchain QR code more secure than a normal QR code?",
                                          "Blockchain QR code vs normal QR code",
                                          "Is the LearningChain QR code secure?"],
    "what are blockchain-based digital credentials?": ["What are blockchain credentials?",
                                                       "What is a blockchain digital certificate?"],
    "what happens if i make changes to a document": ["What if I edit a document after sharing the QR code?",
                                                     "Do I need a new QR code if the document changes?",
                                                     "Can I modify a certificate after issuing it?"],
    "what makes learningchain's verification process different?": [
        "How is LearningChain verification different from manual verification?",
        "Why is LearningChain verification faster?", "How fast is certificate verification?"],
    "does learningchain support different file formats": ["Which file formats are supported?",
                                                          "Can I upload Word or image files?",
                                                          "Does LearningChain accept only PDF?"],
    "what is the validity of the data on the blockchain network": ["How long is the data valid?",
                                                                   "Do blockchain certificates expire?",
                                                                   "How long are records kept on the blockchain?"],
    "can we integrate learningchain with other existing applications?": [
        "Does LearningChain have an API?", "Can LearningChain integrate with our ERP?",
        "Can LearningChain connect to our existing systems?"],
    "does learningchain have control over the data?": ["Can LearningChain access our data?",
                                                       "Who controls the data in LearningChain?",
                                                       "Is our data private?"],
    "what is the difference between learningchain and digilocker?": [
        "LearningChain vs DigiLocker", "Is LearningChain better than DigiLocker?",
        "How is LearningChain different from DigiLocker?"],
    "how will students access and share their records?": ["How do students get their certificates?",
                                                          "How can students share their credentials?",
                                                          "Do students get a digital wallet?"],
    "how does learningchain help in institute brand building?": [
        "How does LearningChain help our institution's brand?", "What are the branding benefits for universities?"],
    "does the institution need to pay repeatedly": ["Do we pay every year to keep certificates?",
                                                    "Is there a recurring fee to maintain records?",
                                                    "Is it a one-time payment per certificate?"],
}


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


def alternates_for(question: str) -> list[str]:
    q = question.lower()
    return next((alts for key, alts in ALTERNATES.items() if q.startswith(key)), [])


def keywords_for(question: str) -> str:
    stop = {"what", "is", "the", "a", "an", "of", "to", "do", "does", "how", "why", "we", "my", "i", "in", "on",
            "and", "for", "with", "can", "will", "after", "if", "have", "need", "using", "make", "makes", "than"}
    words = re.findall(r"[a-z][a-z\-]+", question.lower())
    return ", ".join(dict.fromkeys(w for w in words if w not in stop))




def faq_golden_csv(faqs: list[tuple[str, str]], url: str, existing_csv: str = "") -> str:
    """Approved-answer CSV: previous non-FAQ rows (P001...) are kept, FAQ rows F01.. are regenerated."""
    kept = []
    if existing_csv.strip():
        kept = [r for r in csv.DictReader(io.StringIO(existing_csv)) if not re.fullmatch(r"F\d+", r["Question ID"])]
    rows = kept + [{"Question ID": f"F{i:02d}", "Category": "LearningChain FAQ", "Question": q, "Approved Answer": a,
                    "Keywords": keywords_for(q), "Alternate Questions": " | ".join(alternates_for(q)),
                    "Access Roles": ROLES, "Status": "Approved", "Source URL": url}
                   for i, (q, a) in enumerate(faqs, 1)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=GOLDEN_COLS, quoting=csv.QUOTE_ALL, restval="", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def faq_document(faqs: list[tuple[str, str]], url: str) -> str:
    lines = ["# LearningChain - Frequently Asked Questions (official website)", "",
             "Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.",
             f"Source: {url}", ""]
    for q, a in faqs:
        lines += [f"## FAQ: {q}", "", a, ""]
    return "\n".join(lines).rstrip() + "\n"
