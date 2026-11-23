"""Import the LearningChain FAQ workbook (FAQ_Master sheet, or its CSV export) as approved Q&A, without duplicates.

    python scripts/products/learningchain/import_faq_master.py datasets/source/LearningChain_FAQ_Combined.csv
    (also accepts the original workbook <LearningChain_Chatbot_FAQ.xlsx> or a FAQ_Master CSV)
    python -m ingestion.cli sync --kb products                     # index the result

What it does
* Review Status "Source-verified" / "Derived - confirm" -> active; "Needs input" -> draft (stored, never served,
  because those answers still contain placeholders such as "Please add the exact steps here"). LC-111 (the generic
  fallback row) is not a question and is skipped.
* Repeated questions are removed so every question has exactly one approved answer:
  - automatically: a question whose normalised text equals another entry's question or alternate;
  - by meaning (MERGES / REPLACES below, reviewed by hand): the kept entry gets the removed entry's question and
    alternates as extra alternate phrasings, so no wording that used to match is lost.
  - an alternate phrasing that equals another entry's question is dropped from the alternates (the question wins).
* Writes datasets/learningchain_faq_master.csv (golden format). In datasets/products_golden_dataset.csv the rows the
  FAQ replaces get Status "retired" (never served, kept for history), so the import can be re-run safely. Previous
  versions are copied to data/faq_history/ (local only, not in git).
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FAQ_OUT = ROOT / "datasets" / "learningchain_faq_master.csv"
OLD_GOLDEN = ROOT / "datasets" / "products_golden_dataset.csv"
HISTORY = ROOT / "data" / "faq_history"          # local backups (data/ is not committed)
STATUS = {"source-verified": "active", "derived - confirm": "active", "needs input": "draft"}
SKIP = {"LC-111"}
# Answers taken from the LearningChain Help Centre link to it; answers only from the presentation have no public page.
HELP_CENTRE_URL = "https://learningchainai.zendesk.com/hc/en-gb"

# Older approved answers (products_golden_dataset.csv) that the FAQ now answers: old id -> FAQ id that replaces it.
REPLACES = {"F01": "LC-001", "F02": "LC-004", "F03": "LC-003", "F13": "LC-069", "F15": "LC-008"}
# FAQ entries that repeat an older or another FAQ entry: removed id -> kept id (kept entry absorbs the wording).
MERGES = {"LC-096": "F16",          # students' access to records: the older answer is fuller (email QR + wallet + sharing)
          "LC-045": "LC-034"}       # "secure a digital certificate without a template": admin setup + user steps
# Extra phrasings users type, added as alternate questions of an existing answer (the FAQ wording is not changed).
# Found in testing: these were declined because the answer's own wording never says "steal" / "theft".
EXTRA_ALTERNATES = {
    "LC-006": ["Can someone steal my certificate?", "Can my certificate be stolen or misused?",
               "Is my certificate safe from theft?", "Can someone copy my certificate and use it?"],
}
# Alternate phrasings / keywords in the FAQ that contradict the Help Centre and are dropped on import (the source file
# is not changed). Revoke and Recall are two separate actions (Admin Guide: "How do I Revoke the Certificates" vs
# "How do I Recall the Certificates"; a revoked certificate can no longer be verified), so a revoke question must not
# be answered with the recall steps. "Cancel an issued certificate" describes revoking (verification stops), not recall.
DROP_ALTERNATES = {"LC-090": ["Revoke certificate", "Cancel an issued certificate"]}
DROP_KEYWORDS = {"LC-090": ["revoke"]}
COMBINED_ANSWERS = {
    "LC-034": ("Admin setup: in the certificate type's Template Configuration card, enable 'Blockchain QR code "
               "generation without templates' (this is also the default when no template option is enabled). "
               "Then the User generates it: Certificates > Generate Single, select the certificate type, enter the "
               "student details (Student Name, Email, Mobile Number, Academic Reference Number, Certificate Name, Issue "
               "Month and Year), upload the digital document, tick the guidelines checkbox and click 'Generate "
               "Certificate'. The certificate is pushed to the blockchain and receives a unique QR code; find it under "
               "Reports."),
}
GOLDEN_COLUMNS = ["Question ID", "Category", "Question", "Approved Answer", "Keywords", "Alternate Questions",
                  "Access Roles", "Status", "Source URL"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def split_alts(value) -> list[str]:
    return [a.strip() for a in re.split(r"\|", "" if pd.isna(value) else str(value)) if a.strip()]


def read_faq(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name="FAQ_Master")
    return pd.read_csv(path, encoding="utf-8-sig")


def backup(path: Path) -> None:
    if path.exists():
        HISTORY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, HISTORY / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")


def main(source: str) -> int:
    raw = read_faq(Path(source))
    alt_col = next(c for c in raw.columns if c.lower().startswith("alternate"))
    faq = []
    for _, r in raw.iterrows():
        qid = str(r["FAQ ID"]).strip()
        if qid in SKIP or not str(r["Question"]).strip():
            continue
        faq.append({"Question ID": qid, "Category": re.sub(r"^\d+\.\s*", "", str(r["Category"])).strip(),
                    "Question": str(r["Question"]).strip(),
                    "Approved Answer": COMBINED_ANSWERS.get(qid, str(r["Answer"]).strip()),
                    "Keywords": "" if pd.isna(r["Keywords"]) else str(r["Keywords"]).strip(),
                    "alts": split_alts(r[alt_col]), "Access Roles": "",
                    "Status": STATUS.get(str(r["Review Status"]).strip().lower(), "draft"),
                    "Source URL": HELP_CENTRE_URL if "help centre" in str(r.get("Source", "")).lower() else ""})
    old = pd.read_csv(OLD_GOLDEN, encoding="utf-8-sig")
    old_rows = {str(r["Question ID"]): r for _, r in old.iterrows()}
    retired = set()
    faq_by_id = {f["Question ID"]: f for f in faq}
    removed: list[str] = []

    # 1. hand-reviewed duplicates by meaning
    for old_id, new_id in REPLACES.items():
        if old_id in old_rows and new_id in faq_by_id:
            r = old_rows.pop(old_id)
            retired.add(old_id)
            faq_by_id[new_id]["alts"] += [str(r["Question"])] + split_alts(r["Alternate Questions"])
            removed.append(f"{old_id} (old) -> kept {new_id}: {r['Question']}")
    keep_old_alts: dict[str, list[str]] = {}
    for gone, kept in MERGES.items():
        if gone not in faq_by_id:
            continue
        g = faq_by_id.pop(gone)
        if kept in faq_by_id:
            faq_by_id[kept]["alts"] += [g["Question"]] + g["alts"]
        elif kept in old_rows:
            keep_old_alts.setdefault(kept, []).extend([g["Question"]] + g["alts"])
        removed.append(f"{gone} (FAQ) -> kept {kept}: {g['Question']}")

    for qid, extra in EXTRA_ALTERNATES.items():
        if qid in faq_by_id:
            faq_by_id[qid]["alts"] += extra
    for qid, drop in DROP_ALTERNATES.items():
        if qid in faq_by_id:
            gone = {norm(d) for d in drop}
            faq_by_id[qid]["alts"] = [a for a in faq_by_id[qid]["alts"] if norm(a) not in gone]
    for qid, drop in DROP_KEYWORDS.items():
        if qid in faq_by_id:
            words = [k.strip() for k in faq_by_id[qid]["Keywords"].split(",") if k.strip()]
            faq_by_id[qid]["Keywords"] = ", ".join(k for k in words if k.lower() not in {d.lower() for d in drop})

    # 2. exact repeats: the same normalised QUESTION twice (older answers first, then FAQ order) - first one wins
    seen: dict[str, str] = {norm(r["Question"]): qid for qid, r in old_rows.items()}
    for qid, f in list(faq_by_id.items()):
        owner = seen.get(norm(f["Question"]))
        if owner and owner != qid and f["Status"] == "active":
            target = faq_by_id.get(owner)
            if target is not None:
                target["alts"] += [f["Question"]] + f["alts"]
            faq_by_id.pop(qid)
            removed.append(f"{qid} (FAQ, exact repeat) -> kept {owner}: {f['Question']}")
            continue
        seen.setdefault(norm(f["Question"]), qid)
    questions = set(seen)            # an alternate may never be another entry's question

    # write the FAQ golden file (alternates de-duplicated, never equal to the entry's own question)
    out = []
    for f in faq_by_id.values():
        alts, have = [], {norm(f["Question"])}
        for a in f["alts"]:
            if norm(a) not in have and norm(a) not in questions:
                have.add(norm(a))
                alts.append(a)
        out.append({**{k: f[k] for k in GOLDEN_COLUMNS if k in f}, "Alternate Questions": " | ".join(alts)})
    backup(FAQ_OUT)
    pd.DataFrame(out, columns=GOLDEN_COLUMNS).to_csv(FAQ_OUT, index=False, encoding="utf-8")

    # rewrite the older products golden file without the replaced rows (merged wording added as alternates)
    kept_old = old.copy()
    ids = kept_old["Question ID"].astype(str)
    kept_old.loc[ids.isin(retired), "Status"] = "retired"
    for qid, extra in keep_old_alts.items():
        mask = ids == qid
        current = split_alts(kept_old.loc[mask, "Alternate Questions"].iloc[0])
        merged = current + [a for a in extra if norm(a) not in {norm(c) for c in current} and norm(a) not in questions]
        kept_old.loc[mask, "Alternate Questions"] = " | ".join(merged)
    for qid in old_rows:              # older entries: drop alternates that are now another entry's question
        mask = ids == qid
        own = norm(old_rows[qid]["Question"])
        alts = [a for a in split_alts(kept_old.loc[mask, "Alternate Questions"].iloc[0])
                if norm(a) == own or norm(a) not in questions]
        kept_old.loc[mask, "Alternate Questions"] = " | ".join(alts)
    backup(OLD_GOLDEN)
    kept_old.to_csv(OLD_GOLDEN, index=False, encoding="utf-8")
    active_old = int((kept_old["Status"].astype(str).str.lower() != "retired").sum())

    status = pd.DataFrame(out)["Status"].value_counts().to_dict()
    print(f"FAQ: {len(out)} entries {status} -> {FAQ_OUT.relative_to(ROOT)}")
    print(f"Older approved answers: {len(old)} rows, {active_old} served ({len(old) - active_old} retired) -> "
          f"{OLD_GOLDEN.relative_to(ROOT)}")
    print("Removed as repeated:")
    for line in removed:
        print("  -", line)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
