"""
PDF text extraction via pypdf — tier 2 fallback.

Used when ar5iv does not have the paper (older papers, very recent submissions,
or those with unusual LaTeX that LaTeXML can't convert).

Why pypdf over pdfplumber or pdfminer?
  pypdf is pure Python, has no system dependencies, and handles the vast majority
  of standard PDF layouts. Academic papers are usually single-column or two-column
  text with relatively clean encoding, so pypdf's extraction quality is acceptable.
  pdfplumber gives better layout reconstruction but is significantly heavier.

Limitations:
  - Text order can be wrong in two-column layouts (columns may interleave)
  - Mathematical formulas become garbled or disappear
  - Tables are extracted as flat text with no structure
  These are acceptable trade-offs for summarisation — the model can usually
  reconstruct meaning from imperfect text.

_PYPDF_AVAILABLE guard:
  pypdf is listed as a required dependency in pyproject.toml, so this flag should
  always be True. The guard is defensive — if someone installs without pypdf for
  any reason, the failure is a clear RuntimeError rather than a confusing ImportError
  buried inside a retry loop.

timeout=60:
  PDFs can be large (some are 30MB+). We allow a longer timeout than the HTML
  fetcher to accommodate slow downloads, but still fail fast relative to an
  unbounded wait.
"""

from __future__ import annotations

import io

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def fetch_pdf_text(arxiv_id: str) -> str:
    """
    Download the PDF for `arxiv_id` and extract all text as a single string.

    Pages are joined with newlines. Empty pages (title page, blank separators)
    are skipped. The caller (arxiv_fetch._warm_cache) then passes this text
    to section_splitter.split_sections() to recover section boundaries.

    Raises RuntimeError if pypdf is not installed (should not happen with a
    standard install). Raises httpx.HTTPStatusError for 404/5xx responses,
    which tenacity will retry before propagating.
    """
    if not _PYPDF_AVAILABLE:
        raise RuntimeError("pypdf not installed")

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    # Download the entire PDF into memory — avoid streaming to keep PdfReader simple
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()

    reader = PdfReader(io.BytesIO(resp.content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # skip blank pages (no text layer, or purely graphical)
            pages.append(text)

    return "\n".join(pages)
