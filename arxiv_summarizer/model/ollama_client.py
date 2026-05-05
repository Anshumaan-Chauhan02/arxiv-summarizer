"""
OllamaModelClient — thin wrapper around Ollama's /api/generate REST endpoint.

Why a custom client instead of a library?
  Ollama has an official Python SDK, but it adds an extra dependency and hides
  the request/response cycle. Using httpx directly keeps the code auditable —
  you can see exactly what is sent and received — and avoids version mismatch
  issues between the SDK and the local Ollama server.

stream=False:
  We request the complete response in one shot rather than streaming tokens.
  Streaming would complicate the agentic loop: the harness needs the full
  response text before it can scan for <tool> tags and decide what to do next.

num_ctx:
  This is Ollama's parameter for the model's context window size. We pass it
  explicitly on every call so the model doesn't silently truncate long prompts.
  Set via --context-limit (default 8192 for gemma4:e2b).

temperature override:
  Most calls use the instance default (0.7 for summaries). The evaluator and
  memory distillation call generate() with temperature=0.0 to get deterministic
  structured output. Passing None falls back to the instance default.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaModelClient:
    """
    Sends prompts to a locally running Ollama server and returns the response text.

    One client instance is shared across the entire session (including subagent workers,
    which share the same instance). httpx.Client maintains a connection pool, so
    concurrent calls from multiple threads are safe.
    """

    def __init__(
        self,
        model: str = "gemma4:e2b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        temperature: float = 0.7,
        context_limit: int = 8192,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.context_limit = context_limit
        # A persistent client reuses TCP connections across calls (faster than
        # creating a new connection per request, especially for many tool calls)
        self._client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, prompt: str, temperature: float | None = None) -> str:
        """
        Send a prompt and return the model's response text.

        Retries up to 3 times with exponential backoff on any network error
        or HTTP error — local Ollama can occasionally time out on long generations.
        The temperature parameter is per-call so the evaluator and memory distillation
        can request temperature=0.0 without affecting the main session's default.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.context_limit,
            },
        }
        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    def health_check(self) -> bool:
        """
        Check whether Ollama is running. Called at CLI startup to give a clear
        error message instead of a cryptic connection refused exception mid-session.
        Uses a short 5s timeout — if Ollama is running it responds instantly.
        """
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return the names of all models currently pulled in Ollama."""
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def __del__(self) -> None:
        # Best-effort cleanup of the underlying connection pool on GC.
        # __del__ can be called in unusual states (e.g. during interpreter shutdown)
        # so we swallow all exceptions rather than risk a secondary error.
        try:
            self._client.close()
        except Exception:
            pass
