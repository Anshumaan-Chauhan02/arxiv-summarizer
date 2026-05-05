"""RequestRouter: classify user intent and route to the appropriate handler."""

from __future__ import annotations

import re
from enum import Enum


class Intent(Enum):
    SEARCH = "search"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    LIST = "list"
    UNKNOWN = "unknown"


_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")

_INTENT_KEYWORDS = {
    Intent.LIST: ["list", "show me", "what have i", "my summaries", "saved", "history"],
    Intent.COMPARE: ["compare", "difference between", "vs", "versus", "contrast"],
    Intent.SUMMARIZE: ["summarize", "summary", "explain", "what does", "tell me about", "read"],
    Intent.SEARCH: ["find", "search", "look for", "papers on", "papers about"],
}


class RequestRouter:
    def classify(self, user_input: str) -> Intent:
        lower = user_input.lower()

        # Check for arxiv ID pattern — strongly suggests summarize
        has_id = bool(_ARXIV_ID_RE.search(user_input))

        for intent, keywords in _INTENT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                # If there are multiple IDs and compare keyword, prefer compare
                ids = _ARXIV_ID_RE.findall(user_input)
                if intent == Intent.SUMMARIZE and len(ids) >= 2:
                    return Intent.COMPARE
                return intent

        if has_id:
            return Intent.SUMMARIZE

        return Intent.UNKNOWN

    def extract_arxiv_ids(self, text: str) -> list[str]:
        return _ARXIV_ID_RE.findall(text)

    def route_message(self, user_input: str) -> tuple[Intent, str]:
        """Returns (intent, enriched_instruction) for the harness."""
        intent = self.classify(user_input)
        ids = self.extract_arxiv_ids(user_input)

        if intent == Intent.SUMMARIZE and ids:
            return intent, (
                f"Please summarize the arxiv paper {ids[0]}. "
                "Follow these steps:\n"
                "1. Call fetch_paper to get metadata and available sections\n"
                "2. Identify 2-5 prerequisite topics from the abstract/intro\n"
                "3. For each prerequisite, call web_search to research it\n"
                "4. Read each paper section using read_section\n"
                "5. Write the full summary following the required format\n"
                "6. Call save_summary to persist your work\n"
            )

        if intent == Intent.COMPARE and ids:
            return intent, (
                f"Compare these arxiv papers: {', '.join(ids)}. "
                "Fetch each paper and provide a structured comparison."
            )

        if intent == Intent.SEARCH:
            return intent, f"Search arxiv for: {user_input}"

        if intent == Intent.LIST:
            return intent, "list_summaries"

        return Intent.UNKNOWN, user_input
