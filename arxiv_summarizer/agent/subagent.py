"""
SubagentPool — parallel read-only workers for processing long papers.

Why subagents?
  Some papers exceed the model's practical context window. Rather than cramming
  the full text into one prompt, the orchestrator (main AgentHarness at depth=0)
  splits the paper into sections and delegates each to a worker. Workers run in
  parallel threads, each with their own independent AgentHarness instance.

Worker constraints (enforced by read_only=True and depth=parent+1):
  - Cannot call save_summary, delete, or any write/destructive tool
  - Cannot spawn further subagents past max_depth (prevents infinite recursion)
  - Have their own in-memory session that is NOT persisted to disk
  - Receive the section text pre-loaded in the task context, so they don't
    need network access to re-fetch it

Thread safety:
  Each worker gets a fresh AgentHarness and Session — no shared mutable state.
  The OllamaModelClient is shared but httpx.Client is thread-safe for concurrent
  requests (each request gets its own connection from the pool).

Circular import:
  SubagentPool._run_task() lazy-imports AgentHarness at call time to break the
  circular dependency: harness.py imports subagent.py (for the delegate tool),
  and subagent.py needs to import harness.py (to create workers). Lazy import
  resolves this without restructuring the modules.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arxiv_summarizer.model.ollama_client import OllamaModelClient


@dataclass
class SubagentTask:
    """
    One unit of work for a worker agent.

    `context` contains the section text pre-loaded by the orchestrator.
    Workers receive it directly rather than calling read_section() themselves —
    this keeps workers read-only and avoids redundant network calls.
    `allowed_tools` is informational; actual enforcement happens via read_only=True
    on the worker's AgentHarness.
    """
    task_id: str
    instruction: str    # e.g. "Summarize the Methods section in 200 words"
    context: str        # pre-loaded section text
    allowed_tools: list[str] = field(default_factory=lambda: ["read_section"])


class SubagentPool:
    """
    Runs a list of SubagentTasks in parallel using a thread pool.

    Each task becomes an independent AgentHarness at depth=parent_depth+1
    with read_only=True. Results are collected as {task_id: result_text}.
    """

    def __init__(
        self,
        model: "OllamaModelClient",
        parent_depth: int = 0,
        max_workers: int = 4,
    ) -> None:
        self._model = model
        self._parent_depth = parent_depth
        # Cap workers at the number of tasks to avoid spawning idle threads
        self._max_workers = max_workers

    def run_parallel(self, tasks: list[SubagentTask]) -> dict[str, str]:
        """
        Submit all tasks to the thread pool and collect results.

        Uses as_completed() so results are gathered as workers finish rather than
        waiting for all to complete — faster if some sections are shorter than others.
        Any worker exception is caught and returned as an error string so one failed
        section doesn't block the rest.
        """
        if not tasks:
            return {}
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(tasks))) as pool:
            futures = {pool.submit(self._run_task, task): task.task_id for task in tasks}
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as e:
                    results[task_id] = f"[subagent error] {e}"
        return results

    def _run_task(self, task: SubagentTask) -> str:
        """
        Run one task in its own AgentHarness instance.

        Lazy imports here break the circular dependency with harness.py.
        Workers get an empty WorkspaceContext — they don't need git state or the
        paper list, they just process the text passed in task.context.
        """
        from arxiv_summarizer.agent.harness import AgentHarness
        from arxiv_summarizer.agent.session import Session, WorkingMemory
        from arxiv_summarizer.tools.registry import ToolRegistry
        from arxiv_summarizer.agent.context import WorkspaceContext
        from pathlib import Path

        session = Session(
            session_id=f"worker_{task.task_id}",
            created_at="",
            memory=WorkingMemory(task_description=task.instruction),
        )

        # Workers don't need git state or the summaries index
        ctx = WorkspaceContext(
            git_branch="", git_log="", summarized_papers=[], root_dir=Path(".")
        )

        registry = ToolRegistry(allowlist_path=Path(".sessions/tool_allowlist.json"))

        harness = AgentHarness(
            model=self._model,
            session=session,
            registry=registry,
            context=ctx,
            depth=self._parent_depth + 1,
            read_only=True,
        )

        # Pass the section text inline so the worker has everything it needs
        prompt = f"{task.instruction}\n\nContext:\n{task.context}"
        return harness.ask(prompt)
