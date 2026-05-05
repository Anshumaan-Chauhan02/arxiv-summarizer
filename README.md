# arxiv-summarizer

A local agent harness for producing deep, accessible summaries of arxiv papers using Ollama. No cloud APIs — everything runs on your machine.

## What It Does

Given an arxiv paper ID or a search query, the agent:

1. Fetches the full paper text (via ar5iv HTML, PDF fallback, or abstract-only)
2. Identifies prerequisite topics from the introduction
3. Web-searches each prerequisite to build background explanations
4. Reads the paper section by section
5. Writes a structured summary that a non-specialist can follow
6. Scores the summary for quality and regenerates if it falls short
7. Saves the result as a markdown file

Every summary follows this format:

```
# Paper Title (arxiv_id)

## TL;DR
3-4 sentences. Explained like you're 5.

## The Analogy
One sticky real-world analogy.

## Background: What You Need to Know First
### Topic 1  (sourced from web search, 150+ words)
### Topic 2  ...

## The Paper
### What Problem Does This Solve?    (200+ words)
### What Does It Do and How?         (200+ words)
### What's Unique About This Approach?  (150+ words)

## Why Does This Paper Matter?       (150+ words)
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com/) running locally with your chosen model pulled

```bash
ollama pull gemma4:e2b   # default model
ollama serve             # start the server
```

## Installation

```bash
git clone <repo>
cd arxiv-summarizer
uv sync
```

## Usage

```bash
# Summarize a paper by arxiv ID
uv run arxiv-summarizer summarize 1706.03762

# Summarize by title/keyword (finds best match, then summarizes)
uv run arxiv-summarizer summarize "attention is all you need"

# Search without summarizing
uv run arxiv-summarizer search "diffusion models 2024" --max 10

# List all saved summaries
uv run arxiv-summarizer list

# Compare two papers side by side
uv run arxiv-summarizer compare 1706.03762 2005.14165

# Resume a previous session
uv run arxiv-summarizer summarize 1706.03762 --session <session_id>

# Use a different model
uv run arxiv-summarizer summarize 1706.03762 --model llama3.2 --context-limit 8192
```

Summaries are saved to `data/summaries/{arxiv_id}.md`. Sessions are saved to `.sessions/`.

## Project Structure

```
arxiv_summarizer/
├── main.py                    # CLI entry point (typer)
│
├── agent/
│   ├── harness.py             # AgentHarness: the ask() loop
│   │                            model → parse <tool> tags → execute → repeat
│   ├── session.py             # Session + WorkingMemory: full transcript + distilled state
│   ├── context.py             # WorkspaceContext: git state + saved papers index
│   ├── prompt.py              # 4-layer prompt assembly + history compression
│   ├── router.py              # RequestRouter: classify intent → enrich instruction
│   └── subagent.py            # SubagentPool: parallel workers via ThreadPoolExecutor
│
├── model/
│   └── ollama_client.py       # OllamaModelClient: wraps POST /api/generate
│
├── tools/
│   ├── registry.py            # ToolRegistry: schemas, validation, approval gates
│   ├── arxiv_search.py        # search_arxiv(): Atom API search
│   ├── arxiv_fetch.py         # fetch_paper(), read_section(): 3-tier text cascade
│   ├── web_search.py          # web_search(): DuckDuckGo for prerequisite research
│   ├── summaries.py           # save_summary(), list_summaries(), compare_papers()
│   ├── delegate.py            # delegate(): spawn a read-only worker subagent
│   └── sandbox.py             # path_is_within_root(), output_clip()
│
├── parsing/
│   ├── html_parser.py         # ar5iv HTML → {section_name: text} (tier 1)
│   ├── pdf_parser.py          # pypdf text extraction (tier 2)
│   └── section_splitter.py    # heuristic section boundary detection for PDF text
│
├── eval/
│   └── evaluator.py           # SummaryEvaluator: score → critique → regenerate
│
└── storage/
    └── markdown_store.py      # saves/loads data/summaries/{id}.md with YAML front matter
```

## How the Agent Loop Works

The harness is not a one-shot prompt — it runs a loop:

```
while not done:
    prompt = [static prefix] + [working memory] + [compressed history] + [user request]
    response = ollama.generate(prompt)

    if response contains <tool>...</tool>:
        execute the tool, append result to history
        continue loop
    else:
        this is the final answer — return it
```

The model signals tool use by embedding JSON in `<tool>` tags. The harness parses these, runs the actual Python functions, and feeds the results back into the next prompt. This works with any Ollama model regardless of native function-calling support.

### Prompt Layers

| Layer | Content | When rebuilt |
|---|---|---|
| Static prefix | System role + tool schemas + workspace state | Once per session (or when tools change) |
| Working memory | Current paper, key findings, notes | After each final answer |
| Compressed history | Recent 4 exchanges at full length; older ones at 180 chars | Every call |
| Current request | The user's input | Every call |

### Tool Approval

Tools have four risk levels:

| Risk | Examples | Approval |
|---|---|---|
| `safe` | list_summaries, read_section | Never required |
| `network` | search_arxiv, fetch_paper, web_search | Once per session, then persisted |
| `write` | save_summary | Each new file path, session-only (not persisted) |
| `destructive` | (reserved, not used in MVP) | Always required |

Approvals for `network` tools are persisted to `.sessions/tool_allowlist.json` so you don't get re-prompted across runs.

### Paper Fetching Cascade

```
fetch_paper(id)
  │
  ├── Tier 1: ar5iv HTML  →  clean section-tagged HTML, covers ~85% of 2020+ papers
  │
  ├── Tier 2: PDF via pypdf  →  raw text + heuristic section splitting
  │
  └── Tier 3: Abstract only  →  always available, produces shorter summary
```

### Parallel Section Processing

For long papers, the orchestrator delegates sections to parallel workers:

```
SubagentPool.run_parallel([
    Task("Summarize Introduction", intro_text),
    Task("Summarize Methods",      methods_text),
    Task("Summarize Results",      results_text),
    Task("Summarize Conclusion",   conclusion_text),
])
```

Workers run in `ThreadPoolExecutor(max_workers=4)` with `read_only=True` — they can read but never write. The orchestrator then synthesizes their outputs into the final summary.

### Quality Gate

After the summary is written, `SummaryEvaluator` makes a second model call to score it. If score < 0.7, the critique is fed back as context and the summary is regenerated (max 2 rounds). The best-scoring version is saved.

## Output Format

Summaries are saved as `data/summaries/{arxiv_id}.md` with YAML front matter:

```yaml
---
id: 1706.03762
title: Attention Is All You Need
authors: Vaswani, Shazeer, Parmar...
date: 2026-05-05
model: gemma4:e2b
eval_score: 0.84
---

# Attention Is All You Need (1706.03762)

## TL;DR
...
```

## Configuration

All options are CLI flags — no config file needed:

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemma4:e2b` | Ollama model name |
| `--context-limit` | `8192` | Model context window (tokens) |
| `--session` | (new) | Resume a previous session by ID |
