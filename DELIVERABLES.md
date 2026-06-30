# Deliverables Checklist

This repository contains the requested internal-docs agentic RAG assistant deliverables.

## 1. End-to-end app source code

- Ingestion and indexing: `RagLayer.chunk()`, `RagLayer.build()`, and `RagLayer.load()` in `internal_docs_assistant/rag.py`.
- RAG retriever: `RagLayer.retrieve()` in `internal_docs_assistant/rag.py`.
- Agent loop: `InternalDocsAgent.decide()` and `InternalDocsAgent.run()` in `internal_docs_assistant/agent.py`.
- Retrieval tool: `InternalDocsAgent.retrieve()` in `internal_docs_assistant/agent.py`.
- Action tools: `InternalDocsAgent.file_ticket()` and `InternalDocsAgent.lookup_glossary_term()` in `internal_docs_assistant/agent.py`.
- API client: `GeminiModel` in `internal_docs_assistant/model.py`.

## 2. Corpus and index rebuild instructions

- Corpus: `docs/handbook/`.
- Rebuild the index:

```bash
python -m internal_docs_assistant build-index
```

The generated index is written to `data/index/chunks.json`. It is ignored by git because it can be rebuilt from the committed corpus.

## 3. MCP server or Agent Skill

Implemented as a FastMCP stdio MCP server provider in `internal_docs_assistant/mcp_server.py`. Run it with:

```bash
python -m internal_docs_assistant mcp-server
```

The provider exposes `retrieve_handbook_answer`, `file_internal_ticket`, and `lookup_glossary_term` tools backed by the same in-process `InternalDocsAgent` methods.

## 4. Evaluation harness

- Eval set: `evals/eval_set.jsonl` with 21 question/answer cases.
- Code graders: `EvaluationHarness.code_grade()` in `internal_docs_assistant/eval.py`.
- LLM judge: `EvaluationHarness.llm_judge()` in `internal_docs_assistant/eval.py`.
- Prompt-injection safety probe: `EvaluationHarness.safety_probe()` in `internal_docs_assistant/eval.py`.
- Run the suite:

```bash
python -m internal_docs_assistant eval --no-llm-judge
```

Use `python -m internal_docs_assistant eval` with `GEMINI_API_KEY` set to include the Gemini judge.

## 5. Eval report

See `EVAL.md` for scores, failure notes, changes made in response, and baseline-vs-hybrid retrieval numbers.

## 6. README requirements

`README.md` states the model choice, architecture, local/free run path, and limitations.

## 7. Demo

No screen recording is committed. The demo runbook in `README.md` shows how to demonstrate:

1. A grounded answer with citations.
2. A tool/action call.
3. The eval suite running and printing an aggregate score.
