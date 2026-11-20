"""Combine the two LearningChain FAQ sources into ONE file in the FAQ_Master format.

    python scripts/products/learningchain/combine_faq_sources.py <LearningChain_Chatbot_FAQ.xlsx>
    python scripts/products/learningchain/combine_faq_sources.py <New_Leaningchain_FAQs.csv> <Data_LC.csv>

Output: datasets/source/LearningChain_FAQ_Combined.csv - the FAQ rows unchanged, plus one row per certificate /
QR generation mode (TM-01 .. TM-10) built from the Template Modes table, and LC-033 answered with the full list of
modes (its original answer pointed to the 'Template Modes' sheet, which chatbot users cannot see).
Then load it with import_faq_master.py (one file -> one approved-answer set, no repeated questions).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "datasets" / "source" / "LearningChain_FAQ_Combined.csv"

# extra phrasings users type for each mode (row order of the Template Modes table)
MODE_ALTERNATES = [
    "Digital only certificate mode|Certificates without a template|What happens if no template option is enabled?",
    "Embed QR code mode|Put a blockchain QR on my existing PDF certificates|QR on final PDFs",
    "Pre-printed stationery mode|Print certificates on pre-printed paper|Provisional certificate printing",
    "Use our own certificate design mode|Institution-supplied template mode",
    "Original Degree certificate mode|QR position by coordinates|Pre-printed and customer supplied together",
    "Certificates with a student photo|SSC certificate with photo|Photo field mode",
    "Internal printing without API|How do COEP and DY Patil University print certificates?|Printer generated certificates",
    "External printing with API|How does YCMOU issue certificates?|API printing mode",
    "Secure already issued certificates|QR for legacy or passed-out students' certificates",
    "Issue participation or appreciation certificates digitally|Direct digital issuance",
]


def mode_name(raw: str) -> str:
    return re.sub(r"^\d+\.\s*", "", str(raw)).strip()


def load(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(paths) == 1 and paths[0].lower().endswith((".xlsx", ".xls")):
        return (pd.read_excel(paths[0], sheet_name="FAQ_Master"), pd.read_excel(paths[0], sheet_name="Template Modes"))
    if len(paths) == 2:
        return pd.read_csv(paths[0], encoding="utf-8-sig"), pd.read_csv(paths[1], encoding="utf-8-sig")
    raise SystemExit(__doc__)


def main(paths: list[str]) -> int:
    faq, modes = load(paths)
    faq = faq.copy()
    alt_col = next(c for c in faq.columns if c.lower().startswith("alternate"))
    rows, names = [], []
    for i, m in modes.reset_index(drop=True).iterrows():
        name = mode_name(m["Mode"])
        names.append(f"({i + 1}) {name}")
        answer = (f"{name}. Admin configuration: {m['Admin configuration']}. What is uploaded: "
                  f"{m['What is uploaded / supplied']}. QR code placement: {m['QR code placement']}. Who generates it "
                  f"and how: {m['Who generates and how']}. Typical use / example: {m['Typical use / example']}.")
        rows.append({"FAQ ID": f"TM-{i + 1:02d}", "Category": "4. Templates & QR Modes", "Audience": "Admin, User",
                     "Question": f"What is the '{name}' certificate mode?",
                     alt_col: MODE_ALTERNATES[i] if i < len(MODE_ALTERNATES) else "",
                     "Answer": re.sub(r"\.\.", ".", answer), "Keywords": f"mode, template, QR, {name.lower()}",
                     "Source": str(m.get("Source", "")), "Review Status": "Source-verified"})
    overview = ("LearningChain can issue certificates in 10 ways: " + "; ".join(names) + ". Templates are configured "
                "per certificate type by the Admin; ask about any mode for its configuration, uploads and QR placement.")
    mask = faq["FAQ ID"].astype(str).str.strip() == "LC-033"
    if mask.any():
        faq.loc[mask, "Answer"] = overview
        current = str(faq.loc[mask, alt_col].iloc[0])
        extra = "How many ways can LearningChain issue certificates?|What are all the certificate generation modes?"
        faq.loc[mask, alt_col] = f"{current}|{extra}" if current and current != "nan" else extra
    combined = pd.concat([faq, pd.DataFrame(rows)], ignore_index=True)[list(faq.columns)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{OUT.relative_to(ROOT)}: {len(faq)} FAQ rows + {len(rows)} mode rows = {len(combined)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
