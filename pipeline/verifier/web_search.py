"""Small web search/fetch helper for local-LLM grounding."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import os
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


_USER_AGENT = "Mozilla/5.0 (compatible; VeriLecGrounder/1.0)"


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            value = re.sub(r"\s+", " ", data).strip()
            if value:
                self.parts.append(value)


def _timeout() -> float:
    try:
        return max(1.0, float(os.getenv("VERIFIER_WEB_SEARCH_TIMEOUT_SEC", "12") or 12))
    except ValueError:
        return 12.0


def _read_url(url: str, *, max_bytes: int = 1_500_000) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "ko,en;q=0.8"})
    with urlopen(request, timeout=_timeout()) as response:
        content_type = str(response.headers.get("Content-Type", "") or "")
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read(max_bytes)
        return body.decode(charset, errors="replace"), content_type


def _direct_result_url(raw_url: str) -> str:
    value = unescape(raw_url or "").strip()
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return value


def search_web(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    rss_url = (
        f"https://www.bing.com/search?q={quote_plus(query)}"
        "&format=rss&setlang=en-US&cc=US"
    )
    rss, _ = _read_url(rss_url)
    try:
        root = ElementTree.fromstring(rss)
        results = []
        for item in root.findall("./channel/item"):
            url = str(item.findtext("link") or "").strip()
            if not re.match(r"^https?://", url, flags=re.IGNORECASE):
                continue
            results.append(
                {
                    "title": str(item.findtext("title") or "").strip(),
                    "url": url,
                    "snippet": re.sub(r"\s+", " ", str(item.findtext("description") or "")).strip(),
                }
            )
            if len(results) >= max(1, limit):
                return results
        if results:
            return results
    except ElementTree.ParseError:
        pass

    html, _ = _read_url(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
    anchor_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", item)).strip() for item in snippet_pattern.findall(html)]
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, (raw_url, raw_title) in enumerate(anchor_pattern.findall(html)):
        url = _direct_result_url(raw_url)
        if not re.match(r"^https?://", url, flags=re.IGNORECASE) or url in seen:
            continue
        seen.add(url)
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(raw_title))).strip()
        results.append({"title": title, "url": url, "snippet": snippets[index] if index < len(snippets) else ""})
        if len(results) >= max(1, limit):
            break
    return results


def fetch_page_text(url: str, *, max_chars: int = 5000) -> str:
    html, content_type = _read_url(url)
    if "html" not in content_type.lower():
        return ""
    parser = _TextParser()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:max_chars]


def collect_web_evidence(query: str, *, result_limit: int = 5, page_limit: int = 3) -> list[dict[str, Any]]:
    results = search_web(query, limit=result_limit)
    for result in results[: max(0, page_limit)]:
        try:
            result["page_text"] = fetch_page_text(result["url"])
        except Exception as exc:
            result["page_text"] = ""
            result["fetch_error"] = str(exc)
    return results
