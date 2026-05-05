"""
Prompt assembly — 4-layer structure with history compression.

Every call to the model sends a prompt built from four layers stacked in order:

  Layer 1 — Static prefix (built ONCE per session, reused every turn):
    System role + summary format template + tool schemas + workspace state.
    Expensive to rebuild so it's cached on AgentHarness and only refreshed
    when something structural changes (new tool registered, new paper saved).

  Layer 2 — Working memory (rebuilt after each final answer):
    Distilled state: current paper, key findings, notes.
    Short by design — a few hundred tokens at most.

  Layer 3 — Compressed history (rolling window):
    The full conversation transcript, but compressed to fit in context.
    Old entries are truncated to 180 chars; the most recent 4 exchanges keep
    900 chars each. Repeated read_section calls for the same section are
    collapsed to a single "[already read]" marker.

  Layer 4 — Current request (changes every call):
    The user's input, always placed last so it's the model's immediate focus.
"""

from __future__ import annotations

from arxiv_summarizer.agent.context import WorkspaceContext
from arxiv_summarizer.agent.session import Session

# Injected into the static prefix. The model must follow this structure exactly.
SUMMARY_FORMAT_TEMPLATE = """\
Every summary you produce MUST follow this exact structure:

# {Paper Title} ({arxiv_id})

## TL;DR
3-4 sentences. Explain the entire paper as if to a 5-year-old.
No jargon. Use simple, everyday words.

## The Analogy
One concrete real-world analogy that maps directly to what the paper does.
Make it intuitive and memorable.

## Background: What You Need to Know First
For each prerequisite topic you identify from the paper's introduction or related work,
research it using the web_search tool and write a thorough explanation:

### {Topic Name}
A deep explanation — minimum 150 words per topic.
Cover: what it is, why it matters, how it works conceptually, key terms.
Source: {URL you searched}

(2-5 prerequisite topics total)

## The Paper

### What Problem Does This Solve?
Minimum 200 words. What pain point exists in the world today?
Why haven't existing solutions worked? Who is affected?

### What Does It Do and How?
Minimum 200 words. Explain the approach and the core mechanism step by step.

### What's Unique About This Approach?
Minimum 150 words. What's novel? What prior work does it improve on?

## Why Does This Paper Matter?
Minimum 150 words. Real-world impact, what it enables, significance to the field.
"""

SYSTEM_ROLE = """\
You are an expert academic paper summarizer. Your goal is to produce rich, \
deep summaries of arxiv papers that are accessible to a curious reader who \
may not be a specialist in this field. You have access to tools to fetch papers, \
search the web, and save your work.

When you need to call a tool, output it like this:
<tool>{"name": "tool_name", "args": {"key": "value"}}</tool>

Wait for the tool result before continuing. When you have gathered all the \
information you need, write the final summary using the required format below. \
Do not use one-liners — every section should be detailed and substantive.
"""


def build_prefix(context: WorkspaceContext, tool_schemas: str) -> str:
    """
    Build the static Layer 1 prefix.

    Called once at AgentHarness init and again via refresh_prefix() when the
    tool list or workspace changes. The result is a single string concatenating
    system role, summary format, tool schemas, workspace state, and rules.
    """
    return "\n\n".join([
        SYSTEM_ROLE,
        "## Required Summary Format\n" + SUMMARY_FORMAT_TEMPLATE,
        "## Available Tools\n" + tool_schemas,
        "## Workspace\n" + context.to_text(),
        "## Rules\n"
        "- Never read the same paper section twice in one session\n"
        "- All file writes must be inside data/summaries/\n"
        "- Clip large tool outputs before reasoning about them\n"
        "- For papers longer than 6000 tokens, use delegate() to process sections in parallel\n"
        "- Always web_search prerequisites before writing the Background section\n",
    ])


def build_prompt(
    static_prefix: str,
    session: Session,
    current_request: str,
) -> str:
    """
    Assemble the full 4-layer prompt for one model call.

    static_prefix is passed in (not recomputed here) because it was built once
    at harness init and is reused unchanged across all turns in the session.
    """
    memory_block = "## Working Memory\n" + session.memory.to_text()
    history_block = "## Conversation History\n" + history_text(session)
    user_block = f"User: {current_request}"
    return "\n\n".join([static_prefix, memory_block, history_block, user_block])


def history_text(session: Session, recent_n: int = 4, recent_chars: int = 900, old_chars: int = 180) -> str:
    """
    Compress the session history to fit in the model's context window.

    Strategy:
      - Old entries (beyond the last `recent_n` exchanges): truncated to 180 chars.
        180 is enough to know "this happened" without consuming context.
      - Recent entries (last `recent_n` exchanges): kept at 900 chars.
        900 is enough for the model to reason about the most recent tool results.
      - Repeated read_section calls for the same section: collapsed to one marker.
        The model doesn't need to re-read content it already processed; the marker
        reminds it that the section was read without repeating the full text.

    Each exchange = 2 entries (assistant response + tool result), hence `recent_n * 2`.
    """
    history = session.history
    if not history:
        return "(no history)"

    seen_reads: set[str] = set()
    lines = []
    total = len(history)

    for i, entry in enumerate(history):
        is_recent = i >= total - recent_n * 2
        limit = recent_chars if is_recent else old_chars

        # Deduplicate read_section calls: use the first 80 chars of the call content
        # as the key (contains the tool name + args, which identifies the section).
        if entry.role == "tool_call" and entry.tool_name == "read_section":
            dedup_key = entry.content[:80]
            if dedup_key in seen_reads:
                lines.append(f"[{entry.role}] [already read: {dedup_key[:60]}]")
                continue
            seen_reads.add(dedup_key)

        content = entry.content
        if len(content) > limit:
            content = content[:limit] + "…"

        prefix = f"[{entry.role}]"
        if entry.tool_name:
            prefix += f"[{entry.tool_name}]"
        lines.append(f"{prefix} {content}")

    return "\n".join(lines)
