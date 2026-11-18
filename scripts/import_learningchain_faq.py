"""Import the official LearningChain FAQ page (https://learningchain.in/faqs/) into the products knowledge base.

Writes:
  * datasets/products_golden_dataset.csv  - each FAQ as an approved answer (rows F01..Fnn; other rows are kept)
  * knowledge_bases/products/LearningChain_00_Website_FAQ.md - the same FAQs as a searchable document

Then run:  python -m ingestion.cli sync --kb products

Usage: python scripts/import_learningchain_faq.py [--url https://learningchain.in/faqs/]
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "datasets/products_golden_dataset.csv"
DOC = ROOT / "knowledge_bases/products/LearningChain_00_Website_FAQ.md"
COLS = ["Question ID", "Category", "Question", "Approved Answer", "Keywords", "Alternate Questions", "Access Roles",
        "Status", "Source URL"]
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


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SapphirusAssistantImporter)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://learningchain.in/faqs/")
    args = ap.parse_args()
    faqs = parse(fetch(args.url))
    if not faqs:
        print("No FAQs found on the page - the page layout may have changed.")
        return 1

    # 1) approved answers: replace previous F-rows, keep the rest (P001.. pricing/demo/support)
    kept = []
    if GOLDEN.exists():
        with open(GOLDEN, encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if not re.fullmatch(r"F\d+", r["Question ID"])]
    rows = kept + [{"Question ID": f"F{i:02d}", "Category": "LearningChain FAQ", "Question": q, "Approved Answer": a,
                    "Keywords": keywords_for(q), "Alternate Questions": " | ".join(alternates_for(q)),
                    "Access Roles": ROLES, "Status": "Approved", "Source URL": args.url}
                    for i, (q, a) in enumerate(faqs, 1)]
    with open(GOLDEN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_ALL, restval="")
        w.writeheader()
        w.writerows(rows)

    # 2) searchable document
    lines = ["# LearningChain - Frequently Asked Questions (official website)", "",
             "Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.",
             f"Source: {args.url}", ""]
    for q, a in faqs:
        lines += [f"## FAQ: {q}", "", a, ""]
    DOC.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Imported {len(faqs)} FAQs -> {GOLDEN.relative_to(ROOT)} and {DOC.relative_to(ROOT)}")
    print("Next: python -m ingestion.cli sync --kb products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
