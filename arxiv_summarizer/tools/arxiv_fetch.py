"""
fetch_paper and read_section tools — 3-tier paper content cascade.

Getting the full text of an arxiv paper is non-trivial. We try three approaches
in order, falling back when the previous one fails:

  Tier 1 — ar5iv HTML (preferred):
    ar5iv.labs.arxiv.org converts arxiv LaTeX source to clean, section-tagged HTML.
    BeautifulSoup extracts <section> elements, giving us named sections with no OCR
    noise. Covers ~85% of papers published since 2020.

  Tier 2 — PDF via pypdf (fallback):
    Downloads the PDF directly from arxiv.org and extracts raw text using pypdf.
    No section structure — SectionSplitter applies heuristic regex to recover
    section boundaries. Covers all papers but lower text quality.

  Tier 3 — Abstract only (last resort):
    The Atom API metadata always includes the abstract. If both HTML and PDF fail,
    fetch_paper() still returns metadata; the model produces a shorter summary
    annotated as "(abstract-only summary)".

In-memory cache (_paper_cache):
  Sections are stored in a module-level dict keyed by arxiv_id. This survives
  across multiple tool calls within the same process, so read_section() never
  re-downloads a paper that was already fetched this session.
  An empty dict {} is stored for tier-3 papers so _warm_cache() doesn't retry.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from arxiv_summarizer.parsing import html_parser, pdf_parser
from arxiv_summarizer.parsing.section_splitter import split_sections
from arxiv_summarizer.tools.sandbox import output_clip

NS = "http://www.w3.org/2005/Atom"

# Module-level cache: {arxiv_id: {section_name: text}}
# An empty dict means the paper was tried but only abstract is available.
_paper_cache: dict[str, dict[str, str]] = {}


def fetch_paper(arxiv_id: str) -> str:
    """
    Fetch metadata + abstract for a paper and warm the section cache.

    After this call, read_section() can serve individual sections without
    making additional network requests. The returned string is what the model
    sees — it includes the list of available section names so the model knows
    what to ask for next.
    """
    arxiv_id = arxiv_id.strip()
    meta = _fetch_metadata(arxiv_id)
    _warm_cache(arxiv_id)

    lines = [
        f"ID: {arxiv_id}",
        f"Title: {meta.get('title', 'Unknown')}",
        f"Authors: {meta.get('authors', 'Unknown')}",
        f"Published: {meta.get('published', 'Unknown')}",
        f"Categories: {meta.get('categories', 'Unknown')}",
        "",
        "Abstract:",
        meta.get("abstract", "(no abstract)"),
        "",
    ]

    if arxiv_id in _paper_cache and _paper_cache[arxiv_id]:
        section_names = list(_paper_cache[arxiv_id].keys())
        lines.append(f"Available sections: {', '.join(section_names)}")
        lines.append("Use read_section(arxiv_id, section_name) to read each section.")
    else:
        lines.append("Full text not available. Abstract-only summary will be produced.")

    return "\n".join(lines)


def read_section(arxiv_id: str, section_name: str) -> str:
    """
    Return the text of one section, clipped to 4000 chars.

    Supports fuzzy matching: if the model asks for "intro" it will match
    "introduction" because we fall back to substring search. This is intentional —
    section names from HTML parsing can vary (e.g. "1_introduction" vs "introduction").
    """
    arxiv_id = arxiv_id.strip()
    if arxiv_id not in _paper_cache:
        _warm_cache(arxiv_id)

    sections = _paper_cache.get(arxiv_id, {})
    if not sections:
        return f"No sections available for {arxiv_id}. Use fetch_paper first."

    # Exact match first, then substring
    if section_name in sections:
        return output_clip(sections[section_name], 4000)

    matches = [k for k in sections if section_name.lower() in k.lower()]
    if matches:
        return output_clip(sections[matches[0]], 4000)

    available = ", ".join(sections.keys())
    return f"Section '{section_name}' not found. Available: {available}"


def list_sections(arxiv_id: str) -> str:
    """List all cached section names for a paper. Triggers cache warm if needed."""
    arxiv_id = arxiv_id.strip()
    if arxiv_id not in _paper_cache:
        _warm_cache(arxiv_id)
    sections = _paper_cache.get(arxiv_id, {})
    if not sections:
        return f"No sections cached for {arxiv_id}."
    return f"Sections for {arxiv_id}: " + ", ".join(sections.keys())


def _warm_cache(arxiv_id: str) -> None:
    """
    Populate _paper_cache for one paper using the 3-tier cascade.

    Tier 3 stores an empty dict ({}) rather than leaving the key absent.
    This way a second call to _warm_cache() for the same ID returns immediately
    instead of retrying the failed fetch.
    """
    if arxiv_id in _paper_cache:
        return

    # Tier 1: ar5iv HTML
    sections = html_parser.fetch_sections(arxiv_id)
    if sections:
        _paper_cache[arxiv_id] = sections
        return

    # Tier 2: PDF text extraction
    try:
        text = pdf_parser.fetch_pdf_text(arxiv_id)
        if text and len(text) > 200:
            _paper_cache[arxiv_id] = split_sections(text)
            return
    except Exception:
        pass

    # Tier 3: abstract only — mark as tried but unavailable
    _paper_cache[arxiv_id] = {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_metadata(arxiv_id: str) -> dict[str, str]:
    """
    Query the arxiv Atom API for one paper's metadata.
    Uses id_list= rather than a search query to get an exact match.
    Retries up to 3 times with exponential backoff on network errors.
    """
    url = "https://export.arxiv.org/api/query"
    resp = httpx.get(url, params={"id_list": arxiv_id, "max_results": 1}, timeout=20)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find(f"{{{NS}}}entry")
    if entry is None:
        return {"title": arxiv_id, "abstract": "Could not fetch metadata."}

    authors = [a.findtext(f"{{{NS}}}name", "") for a in entry.findall(f"{{{NS}}}author")]
    categories = [c.get("term", "") for c in entry.findall(f"{{{NS}}}category")]

    return {
        "title": (entry.findtext(f"{{{NS}}}title") or "").replace("\n", " ").strip(),
        # Cap authors at 5 to keep the metadata block readable
        "authors": ", ".join(authors[:5]) + ("..." if len(authors) > 5 else ""),
        "abstract": (entry.findtext(f"{{{NS}}}summary") or "").replace("\n", " ").strip(),
        "published": (entry.findtext(f"{{{NS}}}published") or "")[:10],
        "categories": ", ".join(categories),
    }
