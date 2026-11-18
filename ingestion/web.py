"""Safe website fetching and main-content extraction for the website monitor and HTML uploads.

SSRF protection (all URLs are untrusted input, including URLs typed by administrators and webhook payloads):
  * only http/https, default ports (80/443), no credentials in the URL
  * host must be in WEB_ALLOWED_DOMAINS (or a subdomain of one)
  * every resolved IP must be public (no private, loopback, link-local, multicast, reserved or unspecified ranges)
  * redirects are followed manually and every hop is validated again (max 5)
  * response size is capped (WEB_MAX_PAGE_MB) and only HTML / text / JSON / XML content types are accepted

Content extraction keeps structure: trafilatura returns the main article as Markdown with headings, lists, tables
and links, and drops navigation, footers, cookie banners and ads.

Scraping backend (WEB_SCRAPER): with "firecrawl" and a FIRECRAWL_API_KEY, monitored web pages are scraped through the
Firecrawl API (`scrape_page`, `fetch_html`), which renders JavaScript-built pages and returns clean main-content
Markdown. The same domain allow-list is enforced for the requested URL and the final URL after redirects. JSON / XML
endpoints (Zendesk Help Center API, sitemaps) are not scraping and keep using `fetch`.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "application/xml",
                         "text/xml")
USER_AGENT = "SapphirusAssistantCrawler/1.0 (+knowledge-base indexer)"


class UnsafeURLError(ValueError):
    """The URL is not allowed (scheme, domain, port or resolves to a non-public address)."""


@dataclass
class FetchResult:
    url: str                 # final URL after redirects
    status: int
    content_type: str
    text: str
    etag: str = ""
    last_modified: str = ""
    not_modified: bool = False


def _host_allowed(host: str, allowed: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in allowed)


def _ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved
                or addr.is_unspecified or (addr.version == 6 and addr.ipv4_mapped is not None
                                           and not _ip_is_public(str(addr.ipv4_mapped))))


def validate_url(url: str, allowed_domains: list[str] | None = None, resolve: bool = True) -> str:
    """Return the normalised URL or raise UnsafeURLError."""
    allowed = [d.lower() for d in (allowed_domains if allowed_domains is not None else get_settings().web_allowed_domains)]
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs are allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with credentials are not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError("URL has no host")
    if parts.port not in (None, 80, 443):
        raise UnsafeURLError("Only the default ports 80/443 are allowed")
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = False
    if is_ip:
        raise UnsafeURLError("IP-address URLs are not allowed; use the site's domain name")
    if not _host_allowed(host, allowed):
        raise UnsafeURLError(f"Domain '{host}' is not in WEB_ALLOWED_DOMAINS")
    if resolve:
        try:
            infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeURLError(f"Cannot resolve host '{host}'") from exc
        for info in infos:
            if not _ip_is_public(info[4][0]):
                raise UnsafeURLError(f"Host '{host}' resolves to a non-public address")
    return parts.geturl()


def fetch(url: str, etag: str = "", last_modified: str = "", allowed_domains: list[str] | None = None,
          max_redirects: int = 5) -> FetchResult:
    """GET a URL safely. Uses conditional headers so unchanged pages cost a 304 only."""
    s = get_settings()
    limit = s.web_max_page_mb * 1024 * 1024
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json;q=0.9,*/*;q=0.5"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    current = url
    with httpx.Client(timeout=s.web_fetch_timeout_seconds, follow_redirects=False) as client:
        for _ in range(max_redirects + 1):
            current = validate_url(current, allowed_domains)
            with client.stream("GET", current, headers=headers) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    if not location:
                        raise ValueError(f"Redirect without location from {current}")
                    current = urljoin(current, location)
                    continue
                if resp.status_code == 304:
                    return FetchResult(current, 304, "", "", etag, last_modified, not_modified=True)
                if resp.status_code >= 400:
                    raise ValueError(f"HTTP {resp.status_code} for {current}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not ctype.startswith(ALLOWED_CONTENT_TYPES):
                    raise ValueError(f"Unsupported content type '{ctype}' for {current}")
                declared = int(resp.headers.get("content-length") or 0)
                if declared > limit:
                    raise ValueError(f"Page too large ({declared} bytes)")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > limit:
                        raise ValueError(f"Page larger than {s.web_max_page_mb} MB")
                encoding = resp.charset_encoding or "utf-8"
                return FetchResult(current, resp.status_code, ctype, body.decode(encoding, errors="replace"),
                                   resp.headers.get("etag", ""), resp.headers.get("last-modified", ""))
    raise ValueError(f"Too many redirects starting at {url}")


@dataclass
class ExtractedPage:
    title: str
    markdown: str
    published: str = ""

    @property
    def content_hash(self) -> str:
        normalised = re.sub(r"\s+", " ", self.markdown).strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def extract_main_content(html: str, url: str = "") -> ExtractedPage:
    """Main article content as Markdown (headings, lists, tables kept; navigation/ads/boilerplate removed)."""
    import trafilatura

    markdown = trafilatura.extract(html, url=url or None, output_format="markdown", include_tables=True,
                                   include_links=True, include_images=False, include_comments=False,
                                   include_formatting=True, favor_recall=True, deduplicate=True) or ""
    meta = trafilatura.extract_metadata(html, default_url=url or None)
    title = (getattr(meta, "title", "") or "").strip()
    published = (getattr(meta, "date", "") or "").strip()
    page_has_headings = re.search(r"<h[1-4][\s>]", html, re.IGNORECASE) is not None
    if not markdown.strip() or (page_has_headings and not re.search(r"^#{1,4} ", markdown, re.MULTILINE)):
        # structured fallback: headings / paragraphs / lists / tables from <main>/<article> (boilerplate removed),
        # used when trafilatura cannot segment the page or its output lost the heading structure
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        lines = []
        for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
            if el.name == "tr":
                cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
                if cells:
                    lines.append("| " + " | ".join(cells) + " |")
                continue
            if el.find_parent("table"):
                continue
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if el.name in ("h1", "h2", "h3", "h4"):
                lines.append("#" * int(el.name[1]) + " " + text)
            elif el.name == "li":
                lines.append("- " + text)
            else:
                lines.append(text)
        markdown = re.sub(r"\|\n\n\|", "|\n|", "\n\n".join(lines))     # keep table rows together
        title = title or (soup.title.get_text(strip=True) if soup.title else "")
    return ExtractedPage(title=title or url, markdown=markdown.strip(), published=published)


def page_to_document(page: ExtractedPage, url: str) -> str:
    """Markdown document written to the knowledge-base folder (the 'Source:' line gives every chunk its link)."""
    header = [f"# {page.title}", "", f"Source: {url}"]
    if page.published:
        header.append(f"Last updated: {page.published}")
    body = page.markdown
    if body.lstrip().startswith("# "):           # avoid a duplicated title heading
        body = body.split("\n", 1)[1] if "\n" in body else ""
    return "\n".join(header) + "\n\n" + body.strip() + "\n"


# ------------------------------------------------------------------------------------------------ Firecrawl
class FirecrawlError(RuntimeError):
    """The Firecrawl API rejected the request or failed (auth, credits, rate limit, service error)."""


_warned_no_key = False


def firecrawl_enabled() -> bool:
    """Firecrawl is used when WEB_SCRAPER=firecrawl and an API key is configured; otherwise the builtin scraper."""
    global _warned_no_key
    s = get_settings()
    if s.web_scraper != "firecrawl":
        return False
    if not s.firecrawl_api_key:
        if not _warned_no_key:
            logger.warning("WEB_SCRAPER=firecrawl but FIRECRAWL_API_KEY is not set - using the builtin scraper")
            _warned_no_key = True
        return False
    return True


def firecrawl_scrape(url: str, formats: tuple[str, ...] = ("markdown",)) -> dict:
    """Scrape one page with the Firecrawl API (POST {FIRECRAWL_API_URL}/scrape). Returns the response `data` object
    (markdown / rawHtml / metadata). The URL and the final URL after redirects must pass the same domain allow-list
    as direct fetching. A page that answers with HTTP >= 400 raises ValueError("HTTP <code> for <url>") exactly like
    `fetch`, so removed pages are detected the same way."""
    s = get_settings()
    url = validate_url(url)
    body = {"url": url, "formats": list(formats), "onlyMainContent": True,
            "timeout": s.firecrawl_timeout_seconds * 1000, "maxAge": s.firecrawl_max_age_ms}
    try:
        resp = httpx.post(f"{s.firecrawl_api_url.rstrip('/')}/scrape", json=body,
                          headers={"Authorization": f"Bearer {s.firecrawl_api_key}"},
                          timeout=s.firecrawl_timeout_seconds + 15)
    except httpx.HTTPError as exc:
        raise FirecrawlError(f"Firecrawl request failed: {type(exc).__name__}: {exc}") from exc
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code == 429:
        raise FirecrawlError("Firecrawl rate limit exceeded - try again later")
    if resp.status_code >= 400 or not payload.get("success", False):
        code = payload.get("code") or resp.status_code
        raise FirecrawlError(f"Firecrawl error ({code}): {str(payload.get('error') or resp.text)[:300]}")
    data = payload.get("data") or {}
    meta = data.get("metadata") or {}
    status = int(meta.get("statusCode") or 200)
    final_url = meta.get("url") or meta.get("sourceURL") or url
    if status >= 400:
        raise ValueError(f"HTTP {status} for {final_url}")
    validate_url(final_url, resolve=False)        # a redirect must not leave the allowed domains
    data["final_url"] = final_url
    return data


def _firecrawl_title(meta: dict) -> str:
    title = meta.get("title") or meta.get("ogTitle") or ""
    if isinstance(title, list):
        title = next((t for t in title if t), "")
    return str(title).strip()


@dataclass
class ScrapeResult:
    url: str                       # final URL after redirects
    page: ExtractedPage | None     # None when the page was not modified (conditional GET, builtin scraper only)
    etag: str = ""
    last_modified: str = ""
    not_modified: bool = False
    backend: str = "builtin"


def scrape_page(url: str, etag: str = "", last_modified: str = "") -> ScrapeResult:
    """Main content of a web page as Markdown, using Firecrawl when configured, else fetch + trafilatura."""
    if firecrawl_enabled():
        data = firecrawl_scrape(url, ("markdown",))
        meta = data.get("metadata") or {}
        page = ExtractedPage(title=_firecrawl_title(meta) or data["final_url"],
                             markdown=str(data.get("markdown") or "").strip(),
                             published=str(meta.get("publishedTime") or meta.get("modifiedTime") or "").strip())
        return ScrapeResult(url=data["final_url"], page=page, backend="firecrawl")
    fetched = fetch(url, etag, last_modified)
    if fetched.not_modified:
        return ScrapeResult(url=fetched.url, page=None, etag=etag, last_modified=last_modified, not_modified=True)
    return ScrapeResult(url=fetched.url, page=extract_main_content(fetched.text, fetched.url), etag=fetched.etag,
                        last_modified=fetched.last_modified)


def fetch_html(url: str) -> str:
    """Raw HTML of a page (for structure-aware parsers such as the FAQ parser): Firecrawl `rawHtml` when configured,
    else a direct fetch."""
    if firecrawl_enabled():
        return str(firecrawl_scrape(url, ("rawHtml",)).get("rawHtml") or "")
    return fetch(url).text


def slug_for(url: str) -> str:
    parts = urlsplit(url)
    path = re.sub(r"[^a-zA-Z0-9]+", "-", (parts.netloc + parts.path).strip("/")).strip("-").lower()
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{path[:80]}-{digest}"
