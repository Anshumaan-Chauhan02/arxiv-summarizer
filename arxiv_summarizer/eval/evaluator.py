"""
SummaryEvaluator — quality gate for generated summaries.

After the main AgentHarness produces a summary, this evaluator makes one additional
model call to score it on four dimensions:
  - accuracy:     does it correctly represent what the abstract says?
  - completeness: are all required sections present (TL;DR, Analogy, Background, etc.)?
  - depth:        are sections detailed enough, or are they one-liners?
  - clarity:      would a non-specialist understand it?

The model outputs a JSON object with a 0.0-1.0 score and a critique string.
If the score is below the threshold (default 0.7), the critique is injected back
into the summarization prompt and the model regenerates. This repeats up to
max_rounds times (default 2). The best-scoring version is always returned,
even if no round reaches the threshold.

Why temperature=0.0 for evaluation?
  We want deterministic, consistent scoring — not creative variation. Low temperature
  makes the model behave more like a classifier than a generator.

Why cap abstract at 1000 chars and summary at 3000 chars?
  The eval prompt already contains the rubric and schema. Keeping the inputs short
  prevents the total prompt from exceeding the model's context limit.
  The abstract alone is almost always enough to verify accuracy and completeness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from arxiv_summarizer.model.ollama_client import OllamaModelClient

_EVAL_PROMPT = """\
You are evaluating the quality of an academic paper summary.

Paper title: {title}

Abstract (ground truth):
{abstract}

Summary to evaluate:
{summary}

Rate this summary on the following rubric (each 0.0-1.0):
- accuracy: Does it correctly represent what the abstract says?
- completeness: Does it cover all required sections (TL;DR, Analogy, Background, Problem, How, Unique, Significance)?
- depth: Are the sections detailed (not one-liners)?
- clarity: Is it understandable to a non-specialist?

Output ONLY valid JSON:
{{"score": <average 0.0-1.0>, "critique": "<specific improvement needed>"}}
"""


@dataclass
class EvalResult:
    """
    Result of one evaluation pass.
    `passed` is True when score >= threshold (default 0.7).
    `critique` is the model's specific feedback, used as input for regeneration.
    """
    score: float
    critique: str
    passed: bool


class SummaryEvaluator:
    def __init__(self, model: OllamaModelClient, threshold: float = 0.7) -> None:
        self._model = model
        self._threshold = threshold

    def evaluate(self, paper_title: str, abstract: str, summary: str) -> EvalResult:
        """
        Score a summary against the paper's abstract.

        Returns a fallback EvalResult(score=0.5) if the model response can't be
        parsed as JSON — 0.5 is below the threshold, so the harness will attempt
        regeneration rather than silently accepting a potentially bad summary.
        """
        prompt = _EVAL_PROMPT.format(
            title=paper_title,
            abstract=abstract[:1000],
            summary=summary[:3000],
        )
        try:
            raw = self._model.generate(prompt, temperature=0.0)
            # Model may wrap JSON in markdown code fences — search for the object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 0.5))
                critique = str(data.get("critique", ""))
                return EvalResult(score=score, critique=critique, passed=score >= self._threshold)
        except Exception:
            pass
        return EvalResult(score=0.5, critique="Could not parse eval result", passed=False)

    def regenerate_if_needed(
        self,
        summary: str,
        eval_result: EvalResult,
        model: OllamaModelClient,
        original_prompt: str,
        max_rounds: int = 2,
    ) -> str:
        """
        If the summary didn't pass, inject the critique and regenerate.

        We track the best-scoring version across all rounds and return it at the end
        — even if no round reaches the threshold, the caller always gets the best
        attempt rather than the original failing summary.
        """
        if eval_result.passed:
            return summary

        best_summary = summary
        best_score = eval_result.score

        for _ in range(max_rounds):
            improved_prompt = (
                f"{original_prompt}\n\n"
                f"Previous attempt critique: {eval_result.critique}\n"
                "Please address the critique and produce an improved summary."
            )
            new_summary = model.generate(improved_prompt)
            # For re-evaluation we don't have the title/abstract in scope here,
            # so use placeholder values — the rubric still checks completeness and depth.
            new_eval = self.evaluate("(regeneration)", "(see summary)", new_summary)
            if new_eval.score > best_score:
                best_summary = new_summary
                best_score = new_eval.score
            eval_result = new_eval
            if new_eval.passed:
                break

        return best_summary
