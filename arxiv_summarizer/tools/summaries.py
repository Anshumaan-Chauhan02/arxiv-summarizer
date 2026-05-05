"""
Summary management tools — save_summary, list_summaries, compare_papers.

Why factories instead of plain functions?
  All three tools need access to a MarkdownStore instance, and compare_papers
  also needs the fetch_paper function. We can't pass these as global state
  (that would make testing hard and create hidden dependencies). Instead, each
  function is created via a factory that closes over the dependencies it needs.

  main.py calls these factories at startup, passing in the store and fetch_paper,
  and then registers the resulting closures with the ToolRegistry. The model
  never sees this wiring — it just sees tool names and schemas.

context_refresher in make_save_summary:
  After saving a summary, the workspace context (WorkspaceContext.summarized_papers)
  would be stale — it was built at startup and doesn't know about the new file.
  Passing context_refresher=context.refresh_papers ensures the paper list in the
  static prompt prefix is updated before the next model call.
"""

from __future__ import annotations

from pathlib import Path


def make_save_summary(store, context_refresher=None):
    """
    Returns the save_summary tool function bound to `store`.

    context_refresher is an optional callable (WorkspaceContext.refresh_papers)
    that re-scans data/summaries/ after the new file is written, keeping the
    workspace context current for the rest of the session.
    """
    def save_summary(arxiv_id: str, summary: str, title: str = "", authors: str = "") -> str:
        path = store.save(arxiv_id, summary, title=title, authors=authors)
        if context_refresher:
            context_refresher()
        return f"Summary saved to {path}"
    return save_summary


def make_list_summaries(store):
    """
    Returns the list_summaries tool function bound to `store`.

    The filter argument lets the model search by title or ID fragment —
    useful when the user asks "have I already summarized any attention papers?"
    """
    def list_summaries(filter: str = "") -> str:
        entries = store.list_all(filter_text=filter or None)
        if not entries:
            return "No summaries found."
        lines = [f"Saved summaries ({len(entries)}):\n"]
        for e in entries:
            score = f"  score={e['score']}" if e.get("score") else ""
            lines.append(f"  {e['id']}  [{e['date']}]  {e['title']}{score}")
        return "\n".join(lines)
    return list_summaries


def make_compare_papers(fetch_paper_fn):
    """
    Returns the compare_papers tool function that uses `fetch_paper_fn` to
    load metadata for each paper.

    The returned string is a structured prompt fragment — it fetches each paper's
    metadata and prepends a comparison rubric so the model knows what dimensions
    to compare across. The model then writes the actual comparison as its final answer.
    Capped at 4 papers to keep the combined metadata from flooding the context.
    """
    def compare_papers(ids: list[str]) -> str:
        if len(ids) < 2:
            return "Provide at least 2 paper IDs to compare."
        papers = []
        for arxiv_id in ids[:4]:
            info = fetch_paper_fn(arxiv_id)
            papers.append(f"=== {arxiv_id} ===\n{info}\n")
        header = (
            "Compare these papers. For each dimension below, provide a detailed answer:\n"
            "1. Problem being solved\n2. Core approach/method\n3. Key contributions\n"
            "4. Datasets / benchmarks used\n5. Main results\n6. Limitations\n\n"
        )
        return header + "\n".join(papers)
    return compare_papers
