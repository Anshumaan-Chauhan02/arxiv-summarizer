"""
MarkdownStore — persists paper summaries as human-readable markdown files.

Each summary is saved to data/summaries/{arxiv_id}.md with a YAML front matter
block at the top. The front matter stores metadata (id, title, authors, date,
model, eval_score) separately from the summary text, so the CLI can list and
filter summaries without reading the full text of every file.

Why markdown with YAML front matter?
  - Human-readable: you can open any summary in any text editor or markdown viewer
  - Git-diffable: changes between summary versions are meaningful diffs
  - Grep-searchable: ripgrep can find papers by title, topic, or any keyword
  - python-frontmatter parses both the metadata and the body in one call
  - The format is identical to how static site generators (Jekyll, Hugo) store posts,
    so the summaries could be published as a blog with minimal extra work

arxiv_id normalisation:
  arxiv IDs can contain a version suffix (e.g. "1706.03762v5") and occasionally
  a slash in older IDs (e.g. "cs/0612026"). We replace "/" with "_" so the ID is
  safe to use as a filename on all operating systems.

eval_score conditional:
  We only include eval_score in the front matter if one was computed. Summaries
  generated without the evaluator (e.g. abstract-only) should not show a score
  field rather than showing a misleading None/null.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import frontmatter


class MarkdownStore:
    """
    Reads and writes paper summaries as .md files with YAML front matter.

    The summaries directory is created on first use (mkdir parents=True)
    so the caller doesn't need to pre-create it.
    """

    def __init__(self, summaries_dir: Path) -> None:
        self._dir = summaries_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        arxiv_id: str,
        summary: str,
        title: str = "",
        authors: str = "",
        model: str = "",
        eval_score: float | None = None,
    ) -> Path:
        """
        Write a summary to disk, overwriting any previous version for this ID.

        Uses python-frontmatter's Post class to combine YAML metadata with the
        summary body into a single file. frontmatter.dumps() serialises it to
        the standard "---\\n<yaml>\\n---\\n<body>" format.

        Returns the Path of the written file so the tool can report it to the model.
        """
        post = frontmatter.Post(
            summary,
            id=arxiv_id,
            title=title,
            authors=authors,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            model=model,
            # Only include eval_score key when a score was actually computed
            **({"eval_score": round(eval_score, 2)} if eval_score is not None else {}),
        )
        path = self._dir / f"{arxiv_id.replace('/', '_')}.md"
        path.write_text(frontmatter.dumps(post))
        return path

    def load(self, arxiv_id: str) -> str | None:
        """
        Load the body text of a previously saved summary.
        Returns None (not an exception) if the paper hasn't been summarized yet —
        callers treat None as "not found" and proceed to generate a new summary.
        """
        path = self._dir / f"{arxiv_id.replace('/', '_')}.md"
        if not path.exists():
            return None
        post = frontmatter.load(str(path))
        return post.content

    def list_all(self, filter_text: str | None = None) -> list[dict]:
        """
        Scan all .md files and return their front matter as a list of dicts.

        filter_text is matched against title + id (case-insensitive substring).
        Files with broken front matter are silently skipped — a corrupt summary
        file should not break the list command.
        """
        results = []
        for md_file in sorted(self._dir.glob("*.md")):
            try:
                post = frontmatter.load(str(md_file))
                entry = {
                    "id": str(post.get("id", md_file.stem)),
                    "title": str(post.get("title", "")),
                    "date": str(post.get("date", "")),
                    "model": str(post.get("model", "")),
                    "score": post.get("eval_score"),
                }
                if filter_text:
                    haystack = (entry["title"] + " " + entry["id"]).lower()
                    if filter_text.lower() not in haystack:
                        continue
                results.append(entry)
            except Exception:
                pass
        return results

    def exists(self, arxiv_id: str) -> bool:
        """Check whether a summary for this paper has already been saved."""
        return (self._dir / f"{arxiv_id.replace('/', '_')}.md").exists()
