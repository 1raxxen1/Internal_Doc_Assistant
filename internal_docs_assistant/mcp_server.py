from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .agent import InternalDocsAgent
from .rag import RagLayer


def build_agent() -> InternalDocsAgent:
    rag = RagLayer(Path("docs/handbook"), Path("data/index/chunks.json"))
    rag.load()
    return InternalDocsAgent(rag)


def create_server() -> FastMCP:
    """Create the FastMCP server for the internal docs assistant."""

    mcp = FastMCP(
        "internal-docs-assistant",
        instructions=(
            "Use these tools to answer questions from the internal handbook, "
            "file internal tickets, and look up glossary terms."
        ),
    )
    agent = build_agent()

    @mcp.tool()
    def retrieve_handbook_answer(query: str) -> dict[str, Any]:
        """Answer a handbook/internal-docs question with citations."""

        return agent.retrieve(query).payload

    @mcp.tool()
    def file_internal_ticket(query: str) -> dict[str, Any]:
        """File an internal follow-up ticket for the user's request."""

        return agent.file_ticket(query).payload

    @mcp.tool()
    def lookup_glossary_term(query: str) -> dict[str, Any]:
        """Look up a glossary term from the handbook corpus."""

        return agent.lookup_glossary_term(query).payload

    return mcp


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
