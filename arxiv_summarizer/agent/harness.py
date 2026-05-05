"""
AgentHarness — the core agentic loop.

How it works:
  The model is called in a loop. Each response is scanned for <tool>...</tool> tags.
  If tags are found, the harness executes those tools, appends the results to the session
  history, and calls the model again with the updated context. This repeats until the
  model produces a response with no tool tags — that is the final answer.

  The model never directly runs code. It only *requests* tool calls by outputting JSON
  inside <tool> tags. The harness is the one that actually executes them. This design
  works with any Ollama model, including those without native function-calling support.

Depth and read_only:
  `depth` tracks how deeply nested we are in a delegation chain (main agent = 0,
  its workers = 1, their workers = 2, etc.). `read_only=True` blocks write/destructive
  tools — used for subagent workers that should only read and summarize, never save.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from arxiv_summarizer.agent.context import WorkspaceContext
from arxiv_summarizer.agent.prompt import build_prefix, build_prompt
from arxiv_summarizer.agent.session import Session, SessionStore
from arxiv_summarizer.model.ollama_client import OllamaModelClient
from arxiv_summarizer.tools.registry import ToolCall, ToolRegistry

console = Console()

# DOTALL so a tool call can span multiple lines (e.g. long summary args)
_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


class AgentHarness:
    """
    Wraps a model + tools into an agentic ask() loop.

    The harness owns:
    - The conversation session (history + working memory)
    - The tool registry (what the model is allowed to call)
    - The static prompt prefix (built once, reused every turn)
    - Approval gates (asks user before executing risky tools)
    """

    def __init__(
        self,
        model: OllamaModelClient,
        session: Session,
        registry: ToolRegistry,
        context: WorkspaceContext,
        depth: int = 0,
        max_depth: int = 3,
        read_only: bool = False,
        max_steps: int = 20,
        max_attempts: int = 3,
        store: SessionStore | None = None,
        interactive: bool = True,
    ) -> None:
        self._model = model
        self._session = session
        self._registry = registry
        self._context = context
        self._depth = depth
        self._max_depth = max_depth
        self._read_only = read_only
        self._max_steps = max_steps
        self._max_attempts = max_attempts
        self._store = store
        self._interactive = interactive

        # Build once — contains system role, tool schemas, workspace state.
        # Rebuilt only when tools or workspace change (via refresh_prefix).
        self._static_prefix = build_prefix(context, registry.get_prompt_schemas())

        # Tracks exact (tool_name, args) pairs seen this session to prevent
        # the model from issuing the same call twice and wasting steps.
        self._seen_tool_calls: set[str] = set()

    # ── public API ────────────────────────────────────────────────────────────

    def ask(self, user_input: str) -> str:
        """
        Run the agentic loop for one user request.

        Keeps calling the model until it produces a response with no tool tags.
        Tool results are appended to session history so the model sees them on
        the next iteration. Persists the session after every final answer.
        """
        self._session.add("user", user_input)
        steps = 0

        while steps < self._max_steps:
            # We always pass user_input to build_prompt even though it's in history,
            # so the prompt ends with "User: <request>" as a clear call-to-action.
            prompt = build_prompt(self._static_prefix, self._session, user_input)
            raw = self._model.generate(prompt)

            tool_calls = self._parse_tool_calls(raw)

            if not tool_calls:
                # No tool tags → model is done, this is the final answer
                self._session.add("assistant", raw)
                self._distill_memory(raw)
                self._persist()
                return raw

            # Model wants to call tools — record what it said, then execute
            self._session.add("assistant", raw)

            for call in tool_calls:
                result = self._execute_tool(call)
                self._session.add("tool_result", result, tool_name=call.name)
                steps += 1
                if steps >= self._max_steps:
                    break

        self._persist()
        return "Max steps reached. Partial work saved to session history."

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Extract tool calls from raw model output.

        The model signals tool use by embedding JSON inside <tool>...</tool> tags.
        We scan for all such tags in one response — the model may request multiple
        tools in a single turn. Invalid JSON is silently skipped (model retry would
        cost an extra step, so we just ignore malformed tags).
        """
        calls = []
        for match in _TOOL_RE.finditer(text):
            raw_json = match.group(1).strip()
            try:
                data = json.loads(raw_json)
                calls.append(ToolCall(name=data["name"], args=data.get("args", {})))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    # ── execution ─────────────────────────────────────────────────────────────

    def _execute_tool(self, call: ToolCall) -> str:
        """
        Run one tool call through the full safety pipeline:
          1. Dedup check — skip if we've already run this exact call
          2. Validate — correct tool name and required args present
          3. Read-only guard — block write/destructive tools in worker agents
          4. Depth guard — prevent infinite delegation chains
          5. Approval gate — ask user before running risky tools
          6. Execute and return clipped output
        """
        # Sort keys so {"a":1,"b":2} and {"b":2,"a":1} are treated as the same call
        call_key = f"{call.name}:{json.dumps(call.args, sort_keys=True)}"
        if call_key in self._seen_tool_calls:
            return f"[skipped duplicate call to {call.name}]"
        self._seen_tool_calls.add(call_key)

        ok, err = self._registry.validate(call.name, call.args)
        if not ok:
            return f"[tool error] {err}"

        # Worker agents (depth > 0) must never write files or delete data
        if self._read_only:
            spec = self._registry._specs.get(call.name)
            if spec and spec.risk in ("write", "destructive"):
                return f"[blocked] tool '{call.name}' is not allowed in read-only mode"

        # delegate spawns another harness at depth+1; cap to prevent infinite recursion
        if call.name == "delegate" and self._depth >= self._max_depth:
            return f"[blocked] max delegation depth {self._max_depth} reached"

        if self._registry.requires_approval(call.name):
            approved = self._registry.approve_interactively(
                call.name, call.args, interactive=self._interactive
            )
            if not approved:
                return f"[blocked] user denied approval for '{call.name}'"

        console.print(f"  [dim]→ {call.name}({_fmt_args(call.args)})[/dim]")
        result = self._registry.execute(call.name, call.args)
        self._session.add("tool_call", f"{call.name}({_fmt_args(call.args)})", tool_name=call.name)
        return result

    # ── memory distillation ───────────────────────────────────────────────────

    def _distill_memory(self, last_turn: str) -> None:
        """
        After each final answer, make a short secondary model call to update
        WorkingMemory with what was just learned.

        temperature=0.0 for deterministic extraction — we want structured JSON,
        not creative text. This is best-effort: if the model outputs invalid JSON
        or the call fails, we silently keep the old memory rather than crashing.

        We cap last_turn at 600 chars to keep the distillation prompt small.
        """
        current = self._session.memory
        mem_schema = (
            '{"task_description": "...", "current_paper_id": "..." or null, '
            '"current_paper_title": "..." or null, '
            '"key_findings": ["...", "..."], "notes": "..."}'
        )
        distill_prompt = (
            f"Update this working memory JSON based on the assistant turn below.\n"
            f"Current memory: {json.dumps(current.__dict__)}\n"
            f"Last assistant turn (excerpt): {last_turn[:600]}\n"
            f"Output ONLY valid JSON matching this schema: {mem_schema}"
        )
        try:
            raw = self._model.generate(distill_prompt, temperature=0.0)
            # The model may wrap the JSON in markdown fences — search for the object
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                mem = self._session.memory
                mem.task_description = data.get("task_description", mem.task_description)
                mem.current_paper_id = data.get("current_paper_id", mem.current_paper_id)
                mem.current_paper_title = data.get("current_paper_title", mem.current_paper_title)
                mem.key_findings = data.get("key_findings", mem.key_findings)[:8]
                mem.notes = data.get("notes", mem.notes)
        except Exception:
            pass

    # ── persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Write the session to disk so it can be resumed with --session <id>."""
        if self._store:
            self._store.save(self._session)

    # ── prefix refresh ────────────────────────────────────────────────────────

    def refresh_prefix(self) -> None:
        """
        Rebuild the static prompt prefix.

        Call this after registering new tools or after a new summary is saved
        (which changes the workspace paper list). Rebuilding is cheap — it's just
        string concatenation — so it's fine to call it from main.py at startup.
        """
        self._static_prefix = build_prefix(self._context, self._registry.get_prompt_schemas())


def _fmt_args(args: dict) -> str:
    """Compact repr of tool args for the terminal progress line. Shows at most 3 keys."""
    parts = []
    for k, v in list(args.items())[:3]:
        v_str = repr(v)
        if len(v_str) > 40:
            v_str = v_str[:40] + "…"
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)
