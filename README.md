# Internal Docs Agentic RAG Assistant

A reference implementation of an internal-docs/onboarding assistant. It uses Gemini API models, a five-stage RAG pipeline, agentic tool use, an action tool for tickets/glossary lookup, and an automated evaluation harness.

## Architecture

```text
user query -> agent loop -> retrieve(query) tool -> RAG layer -> cited answer
                         -> action tools: file_ticket, lookup_glossary_term
corpus -> chunk -> embed -> store -> retrieve -> generate
whole pipeline -> evaluation harness -> code graders + LLM judge + safety probe
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m internal_docs_assistant build-index
python -m internal_docs_assistant ask "How do I request production access?"
python -m internal_docs_assistant eval --no-llm-judge
```

Set `GEMINI_API_KEY` (or `Gemini_API_Key`) to use Gemini for answer generation, embeddings, agent routing, and LLM-as-judge. Without an API key, the project uses deterministic local fallbacks so the code graders and safety probe can still run.

## Mandatory components mapping

- **Modern API model:** `internal_docs_assistant/model.py` calls Gemini API generation models with structured JSON outputs when `GEMINI_API_KEY` is present.
- **RAG layer:** `internal_docs_assistant/rag.py` implements chunk, embed, store, retrieve, generate. Retrieval is hybrid: TF-IDF/BM25-style lexical scoring plus vector cosine similarity.
- **Agent loop / tool use:** `internal_docs_assistant/agent.py` uses Gemini structured output to decide whether to answer directly or call a tool. Retrieval calls the RAG layer; ticket creation and glossary lookup remain available as action tools. Deterministic routing is available for local runs.
- **Evaluation harness:** `internal_docs_assistant/eval.py` loads 20+ eval cases, runs code graders, an optional LLM-as-judge rubric, and an OWASP LLM01 indirect prompt-injection safety probe.

## Corpus

The sample corpus in `docs/handbook/` represents a company handbook/wiki and includes one hostile retrieved document used by the safety test.


## Deliverables and demo

See `DELIVERABLES.md` for a checklist against the submitted deliverables and `EVAL.md` for the evaluation report.

### Model choice

This project uses Gemini (`gemini-3.5-flash`) for generation and structured agent decisions because it supports fast API calls and JSON-schema responses. It uses `gemini-embedding-001` for embeddings so generation, routing, judging, and embeddings are all Gemini-based.

### Known limitations

- MCP / Agent Skill packaging is not implemented; tools are in-process Python methods.
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
```
