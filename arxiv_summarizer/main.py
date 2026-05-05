"""CLI entry point — arxiv-summarizer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(
    name="arxiv-summarizer",
    help="Local agent harness for rich arxiv paper summaries using Ollama.",
    add_completion=False,
)
console = Console()

ROOT = Path(__file__).parent.parent


def _build_agent(
    model_name: str,
    context_limit: int,
    session_id: Optional[str],
    interactive: bool = True,
) -> tuple:
    """Wire up all components and return (harness, session_store, store)."""
    from arxiv_summarizer.agent.context import WorkspaceContext
    from arxiv_summarizer.agent.harness import AgentHarness
    from arxiv_summarizer.agent.session import SessionStore
    from arxiv_summarizer.eval.evaluator import SummaryEvaluator
    from arxiv_summarizer.model.ollama_client import OllamaModelClient
    from arxiv_summarizer.storage.markdown_store import MarkdownStore
    from arxiv_summarizer.tools.arxiv_fetch import fetch_paper, read_section, list_sections
    from arxiv_summarizer.tools.arxiv_search import search_arxiv
    from arxiv_summarizer.tools.delegate import make_delegate
    from arxiv_summarizer.tools.registry import ToolRegistry, ToolSpec
    from arxiv_summarizer.tools.sandbox import path_is_within_root
    from arxiv_summarizer.tools.summaries import make_compare_papers, make_list_summaries, make_save_summary
    from arxiv_summarizer.tools.web_search import web_search

    sessions_dir = ROOT / ".sessions"
    summaries_dir = ROOT / "data" / "summaries"

    model = OllamaModelClient(model=model_name, context_limit=context_limit)
    store = MarkdownStore(summaries_dir)
    session_store = SessionStore(sessions_dir)

    if session_id:
        try:
            session = session_store.load(session_id)
            console.print(f"[dim]Resumed session {session_id}[/dim]")
        except FileNotFoundError:
            console.print(f"[yellow]Session {session_id} not found, starting new[/yellow]")
            session = session_store.new_session()
    else:
        session = session_store.new_session()

    context = WorkspaceContext.build(ROOT)

    registry = ToolRegistry(allowlist_path=sessions_dir / "tool_allowlist.json")

    # Register all tools
    registry.register(
        ToolSpec("search_arxiv", "Search arxiv for papers matching a query",
                 {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                  "required": ["query"]}, "network"),
        search_arxiv,
    )
    registry.register(
        ToolSpec("fetch_paper", "Fetch metadata and section list for an arxiv paper",
                 {"type": "object", "properties": {"arxiv_id": {"type": "string"}}, "required": ["arxiv_id"]},
                 "network"),
        fetch_paper,
    )
    registry.register(
        ToolSpec("read_section", "Read a specific section of a fetched paper",
                 {"type": "object", "properties": {"arxiv_id": {"type": "string"}, "section_name": {"type": "string"}},
                  "required": ["arxiv_id", "section_name"]}, "safe"),
        read_section,
    )
    registry.register(
        ToolSpec("list_sections", "List all available sections for a fetched paper",
                 {"type": "object", "properties": {"arxiv_id": {"type": "string"}}, "required": ["arxiv_id"]},
                 "safe"),
        list_sections,
    )
    registry.register(
        ToolSpec("web_search", "Search the web for background information on a topic",
                 {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                 "network"),
        web_search,
    )

    save_fn = make_save_summary(store, context_refresher=context.refresh_papers)
    registry.register(
        ToolSpec("save_summary",
                 "Save the completed paper summary to disk",
                 {"type": "object",
                  "properties": {"arxiv_id": {"type": "string"}, "summary": {"type": "string"},
                                 "title": {"type": "string"}, "authors": {"type": "string"}},
                  "required": ["arxiv_id", "summary"]}, "write"),
        save_fn,
    )
    registry.register(
        ToolSpec("list_summaries", "List all saved paper summaries",
                 {"type": "object", "properties": {"filter": {"type": "string"}}, "required": []},
                 "safe"),
        make_list_summaries(store),
    )
    registry.register(
        ToolSpec("compare_papers", "Compare multiple arxiv papers side by side",
                 {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                  "required": ["ids"]}, "network"),
        make_compare_papers(fetch_paper),
    )

    harness = AgentHarness(
        model=model,
        session=session,
        registry=registry,
        context=context,
        store=session_store,
        interactive=interactive,
    )

    # Wire delegate tool now that harness exists
    from arxiv_summarizer.agent.subagent import SubagentPool

    def pool_factory():
        return SubagentPool(model=model, parent_depth=0)

    from arxiv_summarizer.tools.delegate import make_delegate
    registry.register(
        ToolSpec("delegate", "Spawn a read-only subagent to process a section",
                 {"type": "object",
                  "properties": {"instruction": {"type": "string"}, "context": {"type": "string"},
                                 "allowed_tools": {"type": "array", "items": {"type": "string"}}},
                  "required": ["instruction", "context"]}, "safe"),
        make_delegate(pool_factory),
    )
    harness.refresh_prefix()

    return harness, session_store, store


@app.command()
def summarize(
    target: str = typer.Argument(help="arxiv ID (e.g. 1706.03762) or search query"),
    model: str = typer.Option("gemma4:e2b", "--model", "-m", help="Ollama model name"),
    context_limit: int = typer.Option(8192, "--context-limit", help="Model context window size"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Resume session by ID"),
):
    """Summarize an arxiv paper — the full rich format."""
    harness, session_store, store = _build_agent(model, context_limit, session)

    if not harness._model.health_check():
        console.print("[red]✗ Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    from arxiv_summarizer.agent.router import RequestRouter
    router = RequestRouter()
    _, instruction = router.route_message(target)

    console.print(f"\n[bold]Summarizing:[/bold] {target}")
    console.print(f"[dim]Session: {harness._session.session_id}  Model: {model}[/dim]\n")

    result = harness.ask(instruction)

    console.print("\n")
    console.print(Markdown(result))
    console.print(f"\n[dim]Session saved: {harness._session.session_id}[/dim]")


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    max_results: int = typer.Option(5, "--max", "-n", help="Max number of results"),
    model: str = typer.Option("gemma4:e2b", "--model", "-m"),
):
    """Search arxiv for papers."""
    from arxiv_summarizer.tools.arxiv_search import search_arxiv as _search
    console.print(f"\n[bold]Searching arxiv:[/bold] {query}\n")
    result = _search(query, max_results=max_results)
    console.print(result)


@app.command(name="list")
def list_summaries(
    filter: Optional[str] = typer.Argument(None, help="Filter by title or ID"),
):
    """List all saved paper summaries."""
    from arxiv_summarizer.storage.markdown_store import MarkdownStore
    store = MarkdownStore(ROOT / "data" / "summaries")
    entries = store.list_all(filter_text=filter)

    if not entries:
        console.print("[dim]No summaries saved yet. Run: arxiv-summarizer summarize <id>[/dim]")
        return

    table = Table(title=f"Saved Summaries ({len(entries)})", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Date", style="dim")
    table.add_column("Model", style="dim")
    table.add_column("Score", style="green")

    for e in entries:
        table.add_row(
            e["id"], e["title"][:60], e["date"], e["model"],
            str(e.get("score") or "—"),
        )
    console.print(table)


@app.command()
def compare(
    ids: list[str] = typer.Argument(help="2-4 arxiv IDs to compare"),
    model: str = typer.Option("gemma4:e2b", "--model", "-m"),
    context_limit: int = typer.Option(8192, "--context-limit"),
):
    """Compare multiple arxiv papers side by side."""
    if len(ids) < 2:
        console.print("[red]Provide at least 2 arxiv IDs[/red]")
        raise typer.Exit(1)

    harness, _, _ = _build_agent(model, context_limit, None)
    instruction = f"Compare these arxiv papers in detail: {', '.join(ids)}"
    result = harness.ask(instruction)
    console.print(Markdown(result))


@app.command()
def sessions():
    """List saved sessions."""
    from arxiv_summarizer.agent.session import SessionStore
    store = SessionStore(ROOT / ".sessions")
    ids = store.list_sessions()
    if not ids:
        console.print("[dim]No saved sessions.[/dim]")
        return
    for sid in ids:
        console.print(f"  {sid}")


def main():
    app()


if __name__ == "__main__":
    main()
