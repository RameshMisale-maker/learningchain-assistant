"""Firecrawl scraping backend (ingestion/web.py) - the Firecrawl API is mocked, no network or API key needed."""
import json

import httpx
import pytest

from config.settings import get_settings
from ingestion import web


class _Resp:
    def __init__(self, status: int, payload: dict):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


@pytest.fixture
def firecrawl(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "web_scraper", "firecrawl")
    monkeypatch.setattr(s, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(s, "firecrawl_api_url", "https://api.firecrawl.dev/v2")
    monkeypatch.setattr(s, "web_allowed_domains", ["learningchain.in"])
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 443))])
    calls = []

    def respond(status=200, data=None, success=True):
        def post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return _Resp(status, {"success": success, "data": data or {}} if success else data or {})
        monkeypatch.setattr(web.httpx, "post", post)
    return respond, calls


def test_scrape_page_uses_firecrawl_markdown(firecrawl):
    respond, calls = firecrawl
    respond(data={"markdown": "# Features\n\nInstant verification of blockchain certificates." * 3,
                  "metadata": {"title": "Features - LearningChain", "url": "https://learningchain.in/features/",
                               "sourceURL": "https://learningchain.in/features", "statusCode": 200}})
    res = web.scrape_page("https://learningchain.in/features")
    assert res.backend == "firecrawl" and res.url == "https://learningchain.in/features/"
    assert res.page.title == "Features - LearningChain" and "Instant verification" in res.page.markdown
    body = calls[0]["json"]
    assert calls[0]["url"] == "https://api.firecrawl.dev/v2/scrape"
    assert calls[0]["headers"]["Authorization"] == "Bearer fc-test"
    assert body["formats"] == ["markdown"] and body["onlyMainContent"] is True and body["maxAge"] == 0


def test_removed_page_raises_http_404_like_fetch(firecrawl):
    respond, _ = firecrawl
    respond(data={"markdown": "", "metadata": {"statusCode": 404, "url": "https://learningchain.in/old"}})
    with pytest.raises(ValueError, match=r"HTTP 404"):
        web.scrape_page("https://learningchain.in/old")


def test_firecrawl_api_errors_are_reported(firecrawl):
    respond, _ = firecrawl
    respond(status=402, data={"success": False, "error": "Insufficient credits", "code": "insufficient_credits"},
            success=False)
    with pytest.raises(web.FirecrawlError, match="insufficient_credits"):
        web.scrape_page("https://learningchain.in/")
    respond(status=429, data={"error": "Request rate limit exceeded."}, success=False)
    with pytest.raises(web.FirecrawlError, match="rate limit"):
        web.scrape_page("https://learningchain.in/")


def test_domain_allow_list_applies_to_request_and_redirect(firecrawl):
    respond, calls = firecrawl
    with pytest.raises(web.UnsafeURLError):
        web.scrape_page("https://evil.example.com/")
    assert not calls                                   # rejected before any API call
    respond(data={"markdown": "x" * 200, "metadata": {"statusCode": 200, "url": "https://evil.example.com/landing"}})
    with pytest.raises(web.UnsafeURLError):
        web.scrape_page("https://learningchain.in/redirecting")


def test_fetch_html_uses_raw_html_format(firecrawl):
    respond, calls = firecrawl
    respond(data={"rawHtml": "<html><body><h3>FAQ</h3></body></html>",
                  "metadata": {"statusCode": 200, "url": "https://learningchain.in/faqs/"}})
    assert "<h3>FAQ</h3>" in web.fetch_html("https://learningchain.in/faqs/")
    assert calls[0]["json"]["formats"] == ["rawHtml"]


def test_without_api_key_falls_back_to_builtin(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "web_scraper", "firecrawl")
    monkeypatch.setattr(s, "firecrawl_api_key", "")
    monkeypatch.setattr(web, "fetch", lambda url, *a, **k: web.FetchResult(
        url, 200, "text/html", "<html><body><main><h1>Title</h1><p>" + "Body text. " * 20 + "</p></main></body></html>"))
    res = web.scrape_page("https://learningchain.in/")
    assert res.backend == "builtin" and "Body text" in res.page.markdown


def test_generic_pdf_titles_are_replaced_by_file_name():
    from pathlib import Path

    from ingestion.document_loader import LoadedDocument, Section
    from ingestion.pipeline import IngestionPipeline

    doc = LoadedDocument(path=Path("x.pdf"), name="x.pdf", file_type="pdf",
                         sections=[Section(text="www.sapphirus.in\nUSER OPERATIONS", page=1)],
                         metadata={"title": "PowerPoint Presentation"})
    title = IngestionPipeline._title_for(doc, Path("LearningChain_3.0_User_Training_Manual_-_User_Level.pdf"))
    assert title == "LearningChain 3.0 User Training Manual - User Level"
