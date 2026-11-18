"""Convert a Zendesk Help Center export (categories/sections/articles JSON) into Markdown KB files.

Usage: python -I zendesk_to_md.py <export_dir> <output_dir>
"""
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

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


cats = {c["id"]: c for c in json.loads((SRC / "categories.json").read_text(encoding="utf-8"))}
secs = {s["id"]: s for s in json.loads((SRC / "sections.json").read_text(encoding="utf-8"))}
arts = json.loads((SRC / "articles.json").read_text(encoding="utf-8"))

by_file: dict[str, list] = {}
for a in arts:
    if a.get("draft") or a["title"].strip() in SKIP_TITLES:
        continue
    cat = cats[secs[a["section_id"]]["category_id"]]["name"].strip()
    fname, audience, _ = CATEGORY_FILES[cat]
    by_file.setdefault(fname, []).append((secs[a["section_id"]], a, audience, cat))

written = 0
for cat_name, (fname, _, intro) in CATEGORY_FILES.items():
    items = by_file.get(fname, [])
    items.sort(key=lambda x: (x[0].get("position", 0), x[1].get("position", 0), -int(x[1]["id"]) if "Release" in x[0]["name"] else 0))
    lines = [f"# LearningChain Help Centre - {cat_name.replace(chr(39) + 'm a Admin', chr(39) + 'm an Admin')}", "",
             "Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.",
             "Source: LearningChain Help Centre https://learningchainai.zendesk.com/hc/en-gb", "", intro, ""]
    for sec, a, audience, _ in items:
        title = re.sub(r"\s+", " ", a["title"]).strip().rstrip(":")
        heading = f"{audience.split(' (')[0]}: {title}" if audience else title
        if "Release" in sec["name"]:
            heading = f"Release notes: {title}"
        elif "FAQ" in sec["name"]:
            heading = f"FAQ: {title}"
        body = to_text(a["body"])
        lines += [f"## {heading}", "", f"Article: {a['html_url']}", ""]
        if audience:
            lines += [f"Applies to: {audience}.", ""]
        lines += [body or "(No text in this article.)", ""]
        written += 1
    (OUT / fname).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"{fname}: {len(items)} articles")
print("total articles written:", written)
