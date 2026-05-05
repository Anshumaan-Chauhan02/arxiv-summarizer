"""
Sandbox utilities — path safety and output clipping.

These two functions are the harness's primary defences against two common
failure modes in agentic systems:

path_is_within_root:
  Prevents a model-generated path like "../../etc/passwd" from escaping the
  project directory. We use .resolve() on both paths so symlinks and ".." segments
  are fully expanded before comparison — a relative check alone can be bypassed.

output_clip:
  Tool results can be arbitrarily large (a 100-page PDF, a long HTML page).
  If the full output went into the prompt, it would exceed the model's context
  window and either error or cause the model to lose track of earlier context.
  We clip to max_chars and preserve both the head AND tail of the output —
  the head gives the model the opening context (e.g. section title, abstract),
  the tail gives it the conclusion. The middle can usually be inferred.
"""

from __future__ import annotations

from pathlib import Path


def path_is_within_root(path: str | Path, root: Path) -> bool:
    """
    Return True if `path` is inside `root` after resolving symlinks and `..`.
    Used to block file writes outside data/summaries/.
    """
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def output_clip(text: str, max_chars: int = 4000) -> str:
    """
    Truncate text to max_chars, keeping equal portions from head and tail.

    The clipped middle is replaced with a marker showing how many chars were
    removed so the model can tell it's looking at a partial result.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n...[clipped {len(text) - max_chars} chars]...\n" + text[-half:]
