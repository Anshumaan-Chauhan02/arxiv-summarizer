"""
delegate tool — the orchestrator's way of spawning a read-only worker subagent.

When the model calls delegate(), it is asking the harness to:
  1. Create a SubagentPool backed by the same Ollama client
  2. Wrap the given instruction + context into a SubagentTask
  3. Run it as a read-only AgentHarness at depth+1
  4. Return the worker's final response to the orchestrator

Why a factory (make_delegate) instead of a plain function?
  The delegate tool needs a SubagentPool, which in turn needs the OllamaModelClient.
  But at the time tools/delegate.py is imported, neither the client nor the pool
  exists yet — they are created in main.py at startup. The factory pattern defers
  this wiring: pool_factory() is called lazily at the moment the tool is invoked,
  by which point the client is already initialised.

  pool_factory is a zero-argument callable defined in main.py that returns a fresh
  SubagentPool(model=model, parent_depth=0). A new pool is created each time the
  tool runs — pools are cheap (just a thread pool config), and reusing one across
  calls would risk leaking threads from a previous run.

task_id "delegate_0":
  When delegate() is called it runs exactly one task. The task_id just needs to be
  consistent so we can retrieve the result from the dict returned by run_parallel().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arxiv_summarizer.agent.subagent import SubagentPool, SubagentTask


def make_delegate(pool_factory):
    """
    Return the delegate tool function, closing over `pool_factory`.

    pool_factory is a zero-argument callable that returns a SubagentPool.
    It is called fresh on every invocation so each delegate call gets an
    independent thread pool.
    """
    def delegate(instruction: str, context: str, allowed_tools: list[str] | None = None) -> str:
        """
        Spawn a single read-only worker to execute `instruction` against `context`.

        `context` should be the pre-loaded section text from the orchestrator.
        Workers cannot make network calls or write files — they only process
        the text they are given.
        """
        from arxiv_summarizer.agent.subagent import SubagentTask
        pool = pool_factory()
        task = SubagentTask(
            task_id="delegate_0",
            instruction=instruction,
            context=context,
            allowed_tools=allowed_tools or ["read_section"],
        )
        results = pool.run_parallel([task])
        return results.get("delegate_0", "(no result)")
    return delegate
