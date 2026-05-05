"""
Heuristic section boundary detection for plain text extracted from PDFs.

PDF text has no structure — it's a flat stream of characters in visual order.
This module recovers section boundaries by scanning each line against a set of
regex patterns that match common academic paper heading formats:

  Pattern 1: Numbered headings  — "1. Introduction", "2.1 Related Work"
  Pattern 2: ALL-CAPS headings  — "INTRODUCTION", "RELATED WORK"
  Pattern 3: Known title words  — catches mixed-case variants of the canonical
             section names (Abstract, Methods, Results, etc.)

_CANONICAL normalisation:
  Different papers use different names for the same section: "Methodology" vs
  "Methods" vs "Method", "Experiment" vs "Experiments" vs "Evaluation". The
  canonical map collapses these variants to a single consistent key so that
  code calling read_section("methods") works regardless of which variant the
  paper used.

  Names not in _CANONICAL are kept as-is (with spaces replaced by underscores)
  so appendices and paper-specific sections like "Ablation Study" are preserved.

"preamble":
  Text before the first detected heading (typically the title, author list,
  and abstract in papers where the abstract isn't clearly labelled) is
  collected under the key "preamble". The model can read this section to get
  an overview before deciding which sections to focus on.

False positives:
  A line matching a header pattern but mid-sentence (e.g. a caption "Figure 1.
  Introduction to our model") would incorrectly trigger a new section. In
  practice this is rare because academic paper captions appear within paragraphs
  and the patterns require the match to span the entire line (^\s*...\s*$).
"""

from __future__ import annotations

import re


_HEADER_PATTERNS = [
    # "1. Introduction" or "2.1 Background" — numbered, at least 4 chars of title
    re.compile(r"^\s*(\d+\.?\s+[A-Z][A-Za-z ]{3,})\s*$"),
    # "INTRODUCTION" — all caps, at least 5 chars total
    re.compile(r"^\s*([A-Z][A-Z ]{4,})\s*$"),
    # Well-known section names regardless of capitalisation — catches "Abstract",
    # "abstract", "ABSTRACT", etc. The \s*$ anchor prevents mid-line matches.
    re.compile(r"^\s*(Abstract|Introduction|Related Work|"
               r"Background|Methodology|Methods|Experiments?|"
               r"Results?|Discussion|Conclusion|References?|"
               r"Acknowledgements?)\s*$", re.IGNORECASE),
]

# Maps raw (lowercased, number-stripped) heading text to a canonical section key.
# Variants that aren't listed here fall through to raw.replace(" ", "_").
_CANONICAL = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "methodology": "methods",
    "method": "methods",
    "methods": "methods",
    "experiment": "experiments",
    "experiments": "experiments",
    "evaluation": "experiments",
    "result": "results",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "references",
}


def split_sections(text: str) -> dict[str, str]:
    """
    Split plain PDF text into {section_name: content} dict.

    Walks the text line by line. When a header pattern matches, a new section
    is started. All subsequent lines accumulate under that section until the
    next header is detected.

    Empty sections (no content lines) are filtered out in the final dict
    comprehension to avoid cluttering the section list with blank entries.
    """
    lines = text.split("\n")
    sections: dict[str, list[str]] = {}
    current = "preamble"
    sections[current] = []

    for line in lines:
        header = _detect_header(line)
        if header:
            current = header
            if current not in sections:
                sections[current] = []
        else:
            sections[current].append(line)

    # Filter out empty sections and join each section's lines back into a string
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def _detect_header(line: str) -> str | None:
    """
    Test one line against all header patterns.

    Returns the canonical section key if the line is a header, None otherwise.
    The leading number is stripped before the canonical lookup so "1. methods"
    and "2. Methods" both resolve to "methods".
    """
    for pattern in _HEADER_PATTERNS:
        m = pattern.match(line)
        if m:
            raw = m.group(1).strip().lower()
            raw = re.sub(r"^\d+\.?\s*", "", raw).strip()
            return _CANONICAL.get(raw, raw.replace(" ", "_"))
    return None
