"""
Session persistence — full transcript + distilled working memory.

Two layers of memory exist side by side:

  Full transcript (Session.history):
    Every single exchange in the conversation — user messages, assistant responses,
    tool calls, and tool results — stored as a list of HistoryEntry objects and
    persisted to .sessions/{session_id}.json. This is the source of truth for
    resuming a session with --session <id>.

  Working memory (Session.memory / WorkingMemory):
    A small, distilled summary of the current state: what paper is being worked on,
    key findings discovered so far, and any notes. This is injected into every prompt
    as Layer 2 (after the static prefix). It is rebuilt after each final answer via a
    secondary model call in AgentHarness._distill_memory().

Why two layers?
  The full transcript grows long and must be compressed before being sent to the model
  (see prompt.py history_text()). Working memory is always short and always readable —
  it keeps the model oriented without burning context on old raw exchanges.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class WorkingMemory:
    """
    Distilled state of the current session — what the model should remember right now.

    Updated after each final answer by a secondary model call (AgentHarness._distill_memory).
    Injected into the prompt as Layer 2 so the model stays oriented across turns.
    Capped at 8 key_findings to prevent this block from growing unbounded.
    """
    task_description: str = ""
    current_paper_id: str | None = None
    current_paper_title: str | None = None
    key_findings: list[str] = field(default_factory=list)
    notes: str = ""

    def to_text(self) -> str:
        """Render as a plain-text block for prompt injection."""
        lines = []
        if self.task_description:
            lines.append(f"Task: {self.task_description}")
        if self.current_paper_id:
            lines.append(f"Active paper: {self.current_paper_id} — {self.current_paper_title or 'unknown title'}")
        if self.key_findings:
            lines.append("Key findings so far:")
            for f in self.key_findings[:8]:
                lines.append(f"  - {f}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines) if lines else "(no working memory yet)"


@dataclass
class HistoryEntry:
    """
    One item in the conversation transcript.

    role is one of:
      "user"         — the human's message
      "assistant"    — the model's response (may contain <tool> tags)
      "tool_call"    — a formatted string showing which tool was invoked and with what args
      "tool_result"  — the output returned by that tool

    tool_name is set for tool_call and tool_result entries so history_text() can
    identify and deduplicate repeated read_section calls.
    """
    role: str
    content: str
    timestamp: str
    tool_name: str | None = None


@dataclass
class Session:
    """
    The full state of one conversation.

    history is the complete ordered transcript — used for display and compression.
    memory is the distilled view — used for prompt injection.
    """
    session_id: str
    created_at: str
    history: list[HistoryEntry] = field(default_factory=list)
    memory: WorkingMemory = field(default_factory=WorkingMemory)

    def add(self, role: str, content: str, tool_name: str | None = None) -> None:
        """Append one entry to the transcript with the current UTC timestamp."""
        self.history.append(HistoryEntry(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
        ))


class SessionStore:
    """
    Reads and writes sessions as JSON files under .sessions/.

    Also owns the tool_allowlist.json file that lives in the same directory —
    that file persists which tool approvals the user has already granted so they
    don't get re-prompted for network tools on every run.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = sessions_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._allowlist_path = sessions_dir / "tool_allowlist.json"

    def new_session(self) -> Session:
        """Create a fresh session with a random 8-character hex ID."""
        session_id = uuid.uuid4().hex[:8]
        return Session(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def load(self, session_id: str) -> Session:
        """Load a previous session by ID. Raises FileNotFoundError if not found."""
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        data = json.loads(path.read_text())
        history = [HistoryEntry(**e) for e in data["history"]]
        memory = WorkingMemory(**data["memory"])
        return Session(
            session_id=data["session_id"],
            created_at=data["created_at"],
            history=history,
            memory=memory,
        )

    def save(self, session: Session) -> None:
        """Write the session to .sessions/{session_id}.json (overwrites on update)."""
        path = self._dir / f"{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "history": [asdict(e) for e in session.history],
            "memory": asdict(session.memory),
        }
        path.write_text(json.dumps(data, indent=2))

    def list_sessions(self) -> list[str]:
        """List session IDs, excluding the tool_allowlist file that lives in the same dir."""
        return [p.stem for p in sorted(self._dir.glob("*.json"))
                if p.stem != "tool_allowlist"]

    def load_allowlist(self) -> set[str]:
        if not self._allowlist_path.exists():
            return set()
        return set(json.loads(self._allowlist_path.read_text()))

    def save_allowlist(self, allowlist: set[str]) -> None:
        self._allowlist_path.write_text(json.dumps(sorted(allowlist), indent=2))
