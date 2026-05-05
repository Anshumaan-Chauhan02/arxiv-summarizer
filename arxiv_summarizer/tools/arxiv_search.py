"""
search_arxiv — query the arxiv Atom API for papers matching a keyword or phrase.

The arxiv Atom API is free, requires no authentication, and returns structured XML.
We use the `all:` field prefix so the query matches title, abstract, and author fields.
Results are sorted by relevance (arxiv's default ranking).

The response is formatted as a plain-text block that the model can read directly —
each paper gets its ID, title, authors, date, and the first 300 chars of its abstract.
The ID extracted here (e.g. "2301.00001v2") is what the model will pass to fetch_paper().

XML namespace:
  arxiv's Atom feed uses the standard Atom namespace for all elements, so every
  tag lookup must be prefixed with {http://www.w3.org/2005/Atom}. NS holds this
  prefix to avoid repeating the full URL on every find() call.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

NS = "http://www.w3.org/2005/Atom"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arxiv and return a formatted list of matching papers.

    Returns a plain-text block, not JSON, because this goes directly into the
    model's context. Plain text is easier for the model to parse than JSON in
    free-form reasoning.

    Abstract is truncated at 300 chars — enough for the model to judge relevance
    without bloating the context when many results are returned.
    """
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    resp = httpx.get(url, params=params, timeout=20)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries = root.findall(f"{{{NS}}}entry")
    if not entries:
        return "No results found."

    lines = [f"Found {len(entries)} results:\n"]
    for entry in entries:
        # The <id> element contains the full URL; split on /abs/ to get just the ID
        arxiv_id = _get_text(entry, f"{{{NS}}}id").split("/abs/")[-1].strip()
        title = _get_text(entry, f"{{{NS}}}title").replace("\n", " ").strip()
        abstract = _get_text(entry, f"{{{NS}}}summary").replace("\n", " ").strip()
        authors = [a.findtext(f"{{{NS}}}name", "") for a in entry.findall(f"{{{NS}}}author")]
        # Published date is ISO 8601 — slice to YYYY-MM-DD
        published = _get_text(entry, f"{{{NS}}}published")[:10]

        lines.append(f"ID: {arxiv_id}")
        lines.append(f"Title: {title}")
        # Cap at 3 authors to keep output compact; ellipsis signals there are more
        lines.append(f"Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
        lines.append(f"Published: {published}")
        lines.append(f"Abstract: {abstract[:300]}...")
        lines.append("")

    return "\n".join(lines)


def _get_text(elem: ET.Element, tag: str) -> str:
    """Safe text extraction from an XML element — returns empty string if tag is absent."""
    child = elem.find(tag)
    return (child.text or "") if child is not None else ""
