"""
ar5iv HTML parser — tier 1 paper content fetcher.

ar5iv (ar5iv.labs.arxiv.org) is a service run by arxiv that automatically converts
LaTeX source to HTML using LaTeXML. The resulting HTML is:
  - Clean, structured text (no OCR noise like PDF)
  - Section-tagged with <section> elements and <h1>/<h2>/<h3> headings
  - Math rendered in MathML (which we strip — it's noise for summarisation)
  - Available for ~85% of papers published since 2020

Two extraction strategies (tried in order):
  1. Structured <section> tags: ar5iv wraps each section in a <section> element
     with a heading. This is the cleanest result. We use the heading text as the
     section name, normalised to a safe key (lowercase, spaces → underscores).
  2. Heading-based split: if no <section> tags are found, we walk all elements
     and start a new section whenever we encounter an h1/h2/h3. This handles
     older papers where ar5iv's LaTeXML didn't produce structured sections.

Duplicate section keys:
  A paper may have multiple sections with the same heading (e.g. two "Appendix"
  sections). We append an incrementing index (_1, _2...) to avoid silent overwrites.

_normalise_name:
  Converts heading text like "1. Experimental Results" → "experimental_results".
  The leading number and punctuation are stripped because section numbers vary
  between papers and we want consistent keys for the fuzzy matcher in arxiv_fetch.py.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
def fetch_sections(arxiv_id: str) -> dict[str, str] | None:
    """
    Fetch and parse an arxiv paper from ar5iv.

    Returns {section_name: text} on success, or None if:
      - The paper isn't in ar5iv yet (returns 404)
      - The request times out or errors
      - The parsed HTML yields no usable sections

    Returning None (rather than raising) lets the caller fall through to tier 2.
    """
    url = f"{AR5IV_BASE}/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.RequestError:
        return None

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Strip navigation chrome and non-content elements before text extraction.
    # Math (<script type="math/..."> and <style>) would appear as garbage characters.
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    sections: dict[str, str] = {}

    # Strategy 1: structured <section> tags (preferred)
    structured = soup.find_all("section")
    if structured:
        for sec in structured:
            heading = sec.find(["h1", "h2", "h3"])
            name = _normalise_name(heading.get_text(strip=True) if heading else "section")
            text = sec.get_text(separator="\n", strip=True)
            if len(text) > 50:  # skip trivially short sections (e.g. empty appendix stubs)
                key = name
                idx = 1
                while key in sections:
                    key = f"{name}_{idx}"
                    idx += 1
                sections[key] = text
        if sections:
            return sections

    # Strategy 2: split on heading elements (fallback for unstructured HTML)
    current_name = "preamble"
    current_lines: list[str] = []
    for elem in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if elem.name in ("h1", "h2", "h3"):
            if current_lines:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = _normalise_name(elem.get_text(strip=True))
            current_lines = []
        else:
            current_lines.append(elem.get_text(strip=True))
    if current_lines:
        sections[current_name] = "\n".join(current_lines).strip()

    return sections if sections else None


def fetch_abstract(arxiv_id: str) -> str | None:
    """
    Extract just the abstract from the ar5iv HTML page.
    Used as a lightweight alternative when the full fetch isn't needed.
    Matches any <div> whose class contains the word "abstract".
    """
    url = f"{AR5IV_BASE}/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    abstract_div = soup.find("div", class_=lambda c: c and "abstract" in c.lower())
    if abstract_div:
        return abstract_div.get_text(separator=" ", strip=True)
    return None


def _normalise_name(heading: str) -> str:
    """
    Convert an arbitrary heading string into a safe dict key.

    "1. Experimental Results" → "experimental_results"
    "RELATED WORK"            → "related_work"
    ""                        → "section"

    Capped at 40 chars so keys don't become unwieldy in log output.
    """
    import re
    name = heading.lower().strip()
    name = re.sub(r"^\d+\.?\s*", "", name)   # strip leading section numbers
    name = re.sub(r"[^a-z0-9]+", "_", name)  # any non-alnum run → single underscore
    return name[:40] or "section"
