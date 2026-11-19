"""Download a Zendesk Help Center (categories, sections, articles) as JSON via its public API.

Usage:
    python scripts/fetch_zendesk.py https://learningchainai.zendesk.com en-gb data/zendesk_export
    python scripts/zendesk_to_md.py data/zendesk_export knowledge_bases/products
    python -m ingestion.cli sync --kb products
"""
import json
import sys
import urllib.request
from pathlib import Path

base, locale, out = sys.argv[1].rstrip("/"), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
for name in ("categories", "sections", "articles"):
    url, items = f"{base}/api/v2/help_center/{locale}/{name}.json?per_page=100", []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "SapphirusAssistantImporter/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        items += data[name]
        url = data.get("next_page")
    (out / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(items)}")
