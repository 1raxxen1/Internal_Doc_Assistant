from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import InternalDocsAgent
from .eval import EvaluationHarness
from .rag import RagLayer


def build_stack() -> tuple[RagLayer, InternalDocsAgent]:
    rag = RagLayer(Path("docs/handbook"), Path("data/index/chunks.json"))
    rag.load()
    return rag, InternalDocsAgent(rag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal docs agentic RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-index")
    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ev = sub.add_parser("eval")
    ev.add_argument("--no-llm-judge", action="store_true")
    ev.add_argument("--limit", type=int, default=None, help="Run only the first N eval cases for a faster evaluation")
    sub.add_parser("mcp-server")
    args = parser.parse_args()

    if args.command == "mcp-server":
        from .mcp_server import main as run_mcp_server

        run_mcp_server()
        return

    if args.command == "build-index":
        rag = RagLayer(Path("docs/handbook"), Path("data/index/chunks.json"))
        rag.build()
        print("built index")
        return

    rag, agent = build_stack()
    if args.command == "ask":
        print(json.dumps(agent.run(args.query).payload, indent=2))
    elif args.command == "eval":
        harness = EvaluationHarness(agent, Path("evals/eval_set.jsonl"))
        print(json.dumps(harness.run(llm_judge=not args.no_llm_judge, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
