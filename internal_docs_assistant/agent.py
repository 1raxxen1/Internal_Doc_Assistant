from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import GeminiModel
from .rag import RagLayer


@dataclass
class ToolResult:
    tool: str
    payload: dict


class InternalDocsAgent:
    """Simple agent loop: answer directly, or call one tool."""

    def __init__(
        self,
        rag: RagLayer,
        ticket_path: Path = Path("data/tickets/tickets.jsonl"),
        model: GeminiModel | None = None,
    ) -> None:
        self.rag = rag
        self.ticket_path = ticket_path
        self.model = model or rag.model

    def decide(self, query: str) -> dict:
        decision = self._ask_gemini_for_decision(query)
        if decision:
            return decision
        return self._local_decision(query)

    def _ask_gemini_for_decision(self, query: str) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["answer_direct", "retrieve", "file_ticket", "lookup_glossary_term"],
                },
                "answer": {"type": "string"},
                "tool_query": {"type": "string"},
            },
            "required": ["action", "answer", "tool_query"],
            "additionalProperties": False,
        }
        prompt = "Decide if the user can be answered directly or if a tool is needed. Use retrieve for handbook facts."
        decision = self.model.json_chat(prompt, query, schema)
        if decision.get("action"):
            return decision
        return {}

    def _local_decision(self, query: str) -> dict:
        words = query.lower().split()

        if "ticket" in words or "escalate" in words:
            return {"action": "file_ticket", "answer": "", "tool_query": query}

        is_short_greeting = len(words) <= 4 and ("hi" in words or "hello" in words or "thanks" in words)
        if is_short_greeting:
            return {
                "action": "answer_direct",
                "answer": "Hi! Ask me about the handbook, onboarding, access, incidents, expenses, or glossary terms.",
                "tool_query": "",
            }

        return {"action": "retrieve", "answer": "", "tool_query": query}

    def retrieve(self, query: str) -> ToolResult:
        chunks = self.rag.retrieve(query)
        answer = self.rag.generate(query, chunks)
        citations = []
        sources = []
        for chunk in chunks:
            citations.append(chunk.id)
            sources.append(chunk.source)
        return ToolResult("retrieve", {"answer": answer, "citations": citations, "sources": sources})

    def file_ticket(self, query: str) -> ToolResult:
        self.ticket_path.parent.mkdir(parents=True, exist_ok=True)
        next_id = 1
        if self.ticket_path.exists():
            with self.ticket_path.open(encoding="utf-8") as file:
                next_id = sum(1 for _ in file) + 1

        ticket = {"id": f"TICKET-{next_id:04d}", "summary": query, "status": "filed"}
        with self.ticket_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(ticket) + "\n")
        return ToolResult("file_ticket", ticket)

    def lookup_glossary_term(self, query: str) -> ToolResult:
        chunks = self.rag.retrieve(f"Glossary {query}", k=2)
        matches = []
        for chunk in chunks:
            matches.append(asdict(chunk))
        return ToolResult("lookup_glossary_term", {"matches": matches})

    def run(self, query: str) -> ToolResult:
        decision = self.decide(query)
        action = decision["action"]
        tool_query = decision.get("tool_query") or query

        if action == "answer_direct":
            return ToolResult("answer_direct", {"answer": decision["answer"]})
        if action == "file_ticket":
            return self.file_ticket(tool_query)
        if action == "lookup_glossary_term":
            return self.lookup_glossary_term(tool_query)
        return self.retrieve(tool_query)
