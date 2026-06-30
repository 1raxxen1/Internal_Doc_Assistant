# Internal Docs Agentic RAG Assistant

A reference implementation of an internal-docs/onboarding assistant. It uses a local-first RAG pipeline, agentic tool use, an action tool for tickets/glossary lookup, and an automated evaluation harness.

## Architecture

```text
user query -> agent loop -> retrieve(query) tool -> RAG layer -> cited answer
                         -> action tools: file_ticket, lookup_glossary_term
corpus -> chunk -> embed -> store -> retrieve -> generate
whole pipeline -> evaluation harness -> code graders + LLM judge + safety probe
```


## Feature list

- Agentic handbook Q&A with cited answers from the committed `docs/handbook/` corpus.
- Hybrid RAG retrieval that combines lexical scoring with vector cosine similarity.
- Tool/action support for retrieval, internal ticket filing, and glossary lookup.
- FastMCP stdio MCP server provider exposing the assistant tools to MCP-compatible clients.
- Gemini-first model client with deterministic local fallbacks for development and tests.
- OpenRouter fallback for generation and JSON routing when Gemini calls fail and `OPENROUTER_API_KEY` is set; defaults to the `openrouter/free` router unless `OPENROUTER_MODEL` overrides it.
- Evaluation harness with code graders, optional LLM-as-judge, and prompt-injection safety probe.

## Deliverables coverage

This repository includes the end-to-end app and the required submission artifacts:

- **Source code**: `internal_docs_assistant/` contains ingestion/indexing, the RAG retriever, the agent loop, tools, and the model client.
- **Corpus**: `docs/handbook/` is the committed corpus. Rebuild the index with `python -m internal_docs_assistant build-index`.
- **MCP server**: `internal_docs_assistant/mcp_server.py` exposes the MCP tools via stdio.
- **Evaluation harness**: `evals/eval_set.jsonl`, `internal_docs_assistant/eval.py`, and `internal_docs_assistant/model.py` support code-based grading, LLM-judge runs, and safety probing.
- **Report**: `EVAL.md` documents scores, failure cases, and the retrieval enhancement impact.
- **README**: this file documents the model choice, architecture, free/local run path, and limitations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m internal_docs_assistant build-index
python -m internal_docs_assistant ask "How do I request production access?"
python -m internal_docs_assistant eval --no-llm-judge
```

Set `GEMINI_API_KEY` (or `Gemini_API_Key`) to use Gemini for answer generation, embeddings, agent routing, and LLM-as-judge. If Gemini generation or JSON routing fails and `OPENROUTER_API_KEY` is set, the app falls back to OpenRouter chat completions using `OPENROUTER_MODEL` or the default `openrouter/free` router. Without either API key, the project uses deterministic local fallbacks so the code graders and safety probe can still run.

## Mandatory components mapping

- **Modern API model:** `internal_docs_assistant/model.py` calls Gemini API generation models with structured JSON outputs when `GEMINI_API_KEY` is present, then falls back to OpenRouter generation/JSON routing when `OPENROUTER_API_KEY` is configured.
- **RAG layer:** `internal_docs_assistant/rag.py` implements chunk, embed, store, retrieve, generate. Retrieval is hybrid: TF-IDF/BM25-style lexical scoring plus vector cosine similarity.
- **Agent loop / tool use:** `internal_docs_assistant/agent.py` uses structured model output to decide whether to answer directly or call a tool. Retrieval calls the RAG layer; ticket creation and glossary lookup remain available as action tools. Deterministic routing is available for local runs.
- **Evaluation harness:** `internal_docs_assistant/eval.py` loads 20+ eval cases, runs code graders, an optional LLM-as-judge rubric, and an OWASP LLM01 indirect prompt-injection safety probe.

## MCP server provider

Run the FastMCP stdio MCP server provider with:

```bash
python -m internal_docs_assistant mcp-server
```

It exposes these tools to MCP-compatible clients:

- `retrieve_handbook_answer`
- `file_internal_ticket`
- `lookup_glossary_term`

Each tool accepts a JSON argument named `query`.

## Corpus

The sample corpus in `docs/handbook/` represents a company handbook/wiki and includes one hostile retrieved document used by the safety test.


## Deliverables and demo

See `DELIVERABLES.md` for a checklist against the submitted deliverables and `EVAL.md` for the evaluation report.

### Model choice

This project uses Gemini (`gemini-3.5-flash`) first for generation and structured agent decisions because it supports fast API calls and JSON-schema responses. It uses `gemini-embedding-001` for embeddings when Gemini is available, and falls back to OpenRouter (`openrouter/free` by default) for generation and JSON routing if Gemini calls fail.

### Known limitations

- The local fallback is for development only and is not a replacement for Gemini answer quality.
- The vector index is a rebuildable JSON file, not a production vector database.
- No screen recording is committed. Use the commands below for a live 3-5 minute demo.

### Demo runbook

```bash
# 1. Build the index
python -m internal_docs_assistant build-index

# 2. Show a grounded answer with citations
python -m internal_docs_assistant ask "What approvals are needed for production access?"

# 3. Show an action/tool call
python -m internal_docs_assistant ask "File a ticket for production access"

# 4. Show the eval suite and aggregate score
python -m internal_docs_assistant eval --no-llm-judge
python -m internal_docs_assistant eval --limit 5
python -m internal_docs_assistant eval 
```
