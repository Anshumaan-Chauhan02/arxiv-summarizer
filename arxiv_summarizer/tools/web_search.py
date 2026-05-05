"""
web_search — DuckDuckGo Instant Answer API for prerequisite topic research.

Why DuckDuckGo?
  No API key required, no rate limits for reasonable use, and returns structured
  JSON with an AbstractText field (sourced from Wikipedia/Wikidata) that is exactly
  the kind of concise, factual explanation we want for prerequisite topics.

Limitations:
  The Instant Answer API only has answers for well-known topics — it won't find
  niche ML concepts or very recent papers. When it has no answer, we return a
  guidance message telling the model to try more specific search terms.
  The model handles this gracefully because it sees the empty result and can
  try an alternative query or skip the web source for that topic.

Output is clipped to 2000 chars (not the default 4000) because prerequisite
research happens many times per paper (once per topic) and we want each result
to be concise enough that the accumulated context stays manageable.

skip_disambig=1:
  Without this, DuckDuckGo returns a disambiguation page for ambiguous queries
  (e.g. "transformer") instead of the most likely match. The flag forces it to
  pick the best match directly.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from arxiv_summarizer.tools.sandbox import output_clip


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def web_search(query: str) -> str:
    """
    Search the web for background information on a topic.

    Returns a plain-text block with the DuckDuckGo abstract (if found) and up to
    5 related topic snippets. The model uses this to write the Background section
    of the summary, citing the source URL.
    """
    resp = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        timeout=15,
        follow_redirects=True,
        # Identify ourselves so DuckDuckGo can contact us if there's an issue
        headers={"User-Agent": "arxiv-summarizer/0.1 (educational use)"},
    )
    resp.raise_for_status()
    data = resp.json()

    lines = [f"Web search results for: {query}\n"]

    # AbstractText is the main Wikipedia/Wikidata summary — most useful field
    abstract = data.get("AbstractText", "")
    if abstract:
        lines.append(f"Summary: {abstract}")
        lines.append(f"Source: {data.get('AbstractURL', '')}\n")

    # RelatedTopics are sidebar snippets — useful when the main abstract is thin
    topics = data.get("RelatedTopics", [])[:5]
    if topics:
        lines.append("Related topics:")
        for t in topics:
            if isinstance(t, dict) and t.get("Text"):
                lines.append(f"  - {t['Text'][:200]}")

    if len(lines) <= 2:
        # DDG returned no usable content — guide the model to retry differently
        lines.append(f"No instant answer found for '{query}'.")
        lines.append("Consider searching for more specific terms or checking Wikipedia.")

    # Clip at 2000 (not the default 4000) — this tool is called multiple times per paper
    return output_clip("\n".join(lines), 2000)
