"""
ToolRegistry — the central hub for everything tool-related.

Responsibilities:
  1. Registration: maps a tool name → (ToolSpec schema, Python callable)
  2. Prompt injection: serialises all tool schemas as text for the static prefix
  3. Validation: checks tool name exists and all required args are present
  4. Approval: decides whether user confirmation is needed before execution
  5. Execution: calls the Python function and clips the output

Risk levels and their approval behaviour:
  "safe"        — never requires approval (read-only, local operations)
  "network"     — approved once per session; approval is then persisted to
                  tool_allowlist.json so the user isn't re-prompted next run
  "write"       — approved per invocation the first time a new path is written;
                  NOT persisted to disk (user stays in control of writes)
  "destructive" — always requires approval, never cached

Two approval caches exist:
  _session_approved   — in-memory, cleared when the process exits
  _persistent_approved — loaded from tool_allowlist.json, survives restarts
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from rich.console import Console
from rich.prompt import Confirm

from arxiv_summarizer.tools.sandbox import output_clip

console = Console()


@dataclass
class ToolSpec:
    """
    Metadata about one tool: what it's called, what it does, what args it takes,
    and how risky it is. The `parameters` field is a JSON Schema object — the same
    format used by OpenAI function calling — so it doubles as documentation and
    gets injected into the model's prompt verbatim.
    """
    name: str
    description: str
    parameters: dict  # JSON Schema {"type": "object", "properties": {...}, "required": [...]}
    risk: Literal["safe", "network", "write", "destructive"]
    enabled: bool = True


@dataclass
class ToolCall:
    """A parsed tool invocation extracted from raw model output."""
    name: str
    args: dict


class ToolRegistry:
    def __init__(self, allowlist_path: Path) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._fns: dict[str, Callable[..., str]] = {}
        self._allowlist_path = allowlist_path
        # Load persisted approvals from previous sessions
        self._session_approved: set[str] = set()
        self._persistent_approved: set[str] = self._load_allowlist()

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, spec: ToolSpec, fn: Callable[..., str]) -> None:
        """
        Register a tool. Both the schema (for prompt injection and validation)
        and the callable (for execution) are stored by name.
        """
        self._specs[spec.name] = spec
        self._fns[spec.name] = fn

    # ── prompt injection ──────────────────────────────────────────────────────

    def get_prompt_schemas(self) -> str:
        """
        Render all enabled tool schemas as a plain-text block.
        This is what gets embedded in the static prompt prefix so the model
        knows what tools are available and what arguments each one expects.
        """
        lines = []
        for spec in self._specs.values():
            if not spec.enabled:
                continue
            lines.append(f"Tool: {spec.name}")
            lines.append(f"  Description: {spec.description}")
            lines.append(f"  Parameters: {json.dumps(spec.parameters)}")
            lines.append("")
        return "\n".join(lines)

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, name: str, args: dict) -> tuple[bool, str]:
        """
        Check that the tool exists and the model provided all required arguments.
        Returns (True, "") on success or (False, error_message) on failure.
        The error message is returned to the model as a tool_result so it can
        correct itself on the next iteration.
        """
        if name not in self._specs:
            known = ", ".join(self._specs.keys())
            return False, f"Unknown tool '{name}'. Known tools: {known}"
        spec = self._specs[name]
        if not spec.enabled:
            return False, f"Tool '{name}' is disabled"
        required = spec.parameters.get("required", [])
        props = spec.parameters.get("properties", {})
        for req in required:
            if req not in args:
                return False, f"Missing required argument '{req}' for tool '{name}'"
        for key in args:
            if props and key not in props:
                return False, f"Unknown argument '{key}' for tool '{name}'"
        return True, ""

    # ── approval ──────────────────────────────────────────────────────────────

    def requires_approval(self, name: str) -> bool:
        """
        Return True if the user must be asked before this tool runs.

        Safe tools never need approval. For everything else, check both caches —
        if the user already approved this tool (in this session or a previous one),
        skip the prompt.
        """
        spec = self._specs.get(name)
        if spec is None or spec.risk == "safe":
            return False
        approval_key = name
        if approval_key in self._persistent_approved or approval_key in self._session_approved:
            return False
        return True

    def approve_interactively(self, name: str, args: dict, interactive: bool = True) -> bool:
        """
        Show the tool name, risk level, and args, then ask the user y/N.

        Network tools: approval is persisted to tool_allowlist.json after first yes
          — the user doesn't need to approve the same network tool every run.
        Write tools: approval is only session-scoped — never persisted.
          The user sees every write before it happens.

        In non-interactive mode (piped stdin, subagents): write tools are always
        blocked; network tools proceed silently and are added to the session cache.
        """
        spec = self._specs[name]
        console.print(f"\n[yellow]⚠ Tool approval required[/yellow]")
        console.print(f"  Tool: [bold]{name}[/bold]  Risk: [red]{spec.risk}[/red]")
        console.print(f"  Args: {json.dumps(args, indent=2)}")

        if not interactive:
            if spec.risk == "write":
                console.print("[red]  Blocked: non-interactive session, write tools require explicit approval[/red]")
                return False
            self._session_approved.add(name)
            return True

        approved = Confirm.ask(f"  Allow [bold]{name}[/bold]?", default=False)
        if approved:
            self._session_approved.add(name)
            if spec.risk == "network":
                # Network approvals persist — same URL pattern is safe to reuse
                self._persistent_approved.add(name)
                self._save_allowlist()
        return approved

    # ── execution ─────────────────────────────────────────────────────────────

    def execute(self, name: str, args: dict, max_output: int = 4000) -> str:
        """
        Call the registered Python function with the model's args as kwargs.

        Tool functions are registered with signatures that match their JSON Schema
        (e.g. search_arxiv(query, max_results)), so we can unpack args directly
        as **kwargs. The result is clipped to max_output chars before being returned
        to the model — prevents large paper texts from flooding the context window.
        """
        fn = self._fns[name]
        try:
            result = fn(**args)
        except TypeError as e:
            return f"[tool error] Bad arguments for '{name}': {e}"
        except Exception as e:
            return f"[tool error] {name} failed: {e}"
        return output_clip(str(result), max_output)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_allowlist(self) -> set[str]:
        if self._allowlist_path.exists():
            return set(json.loads(self._allowlist_path.read_text()))
        return set()

    def _save_allowlist(self) -> None:
        self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        self._allowlist_path.write_text(json.dumps(sorted(self._persistent_approved), indent=2))

    def list_tools(self) -> list[str]:
        return [n for n, s in self._specs.items() if s.enabled]
