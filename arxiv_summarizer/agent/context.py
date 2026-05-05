"""
WorkspaceContext — a snapshot of the repository state, loaded once at startup.

This is Layer 1 of the static prompt prefix. It tells the model:
  - what git branch we're on and what recent commits look like
  - which papers have already been summarized (so the model can avoid re-doing work)

Why load it once?
  The static prefix is built once and reused on every model call within a session.
  This avoids re-running git and re-scanning the summaries directory on every turn.
  Call refresh_papers() after a new summary is saved so the paper list stays current.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter


@dataclass
class PaperRef:
    """Lightweight reference to a previously summarized paper, read from YAML front matter."""
    arxiv_id: str
    title: str
    date: str


@dataclass
class WorkspaceContext:
    """
    Snapshot of the project's git state and saved summaries.

    Build once with WorkspaceContext.build(root). The resulting object is injected
    into the static prompt prefix via to_text(), giving the model awareness of what
    has already been done.
    """
    git_branch: str
    git_log: str
    summarized_papers: list[PaperRef]
    root_dir: Path

    @classmethod
    def build(cls, root: Path) -> "WorkspaceContext":
        """
        Snapshot the workspace. Runs two git commands and scans data/summaries/.
        Safe to call in a repo with no commits — git failures return empty strings.
        """
        branch = _run_git(root, ["git", "branch", "--show-current"]) or "main"
        log = _run_git(root, ["git", "log", "--oneline", "-10"]) or "(no commits)"
        papers = _load_paper_refs(root / "data" / "summaries")
        return cls(
            git_branch=branch,
            git_log=log,
            summarized_papers=papers,
            root_dir=root,
        )

    def to_text(self) -> str:
        """Render as a plain-text block for injection into the static prompt prefix."""
        lines = [
            f"Repository: {self.root_dir.name}  Branch: {self.git_branch}",
            f"Recent commits:\n{self.git_log}",
        ]
        if self.summarized_papers:
            lines.append(f"\nPreviously summarized papers ({len(self.summarized_papers)}):")
            # Cap at 50 to keep the prefix from growing too large in well-used repos
            for p in self.summarized_papers[:50]:
                lines.append(f"  - {p.arxiv_id}: {p.title} ({p.date})")
        else:
            lines.append("\nNo papers summarized yet.")
        return "\n".join(lines)

    def refresh_papers(self) -> None:
        """
        Rescan data/summaries/ and update the paper list.
        Called by save_summary() after a new .md file is written so the next
        prompt prefix reflects the just-saved paper.
        """
        self.summarized_papers = _load_paper_refs(self.root_dir / "data" / "summaries")


def _run_git(cwd: Path, cmd: list[str]) -> str:
    """Run a git command and return stdout. Returns empty string on any failure."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return ""


def _load_paper_refs(summaries_dir: Path) -> list[PaperRef]:
    """
    Parse YAML front matter from every .md file in the summaries directory.
    Falls back to the filename stem as the ID if front matter is missing or broken.
    """
    if not summaries_dir.exists():
        return []
    refs = []
    for md_file in sorted(summaries_dir.glob("*.md")):
        try:
            post = frontmatter.load(str(md_file))
            refs.append(PaperRef(
                arxiv_id=str(post.get("id", md_file.stem)),
                title=str(post.get("title", "Unknown title")),
                date=str(post.get("date", "")),
            ))
        except Exception:
            refs.append(PaperRef(arxiv_id=md_file.stem, title="(parse error)", date=""))
    return refs
