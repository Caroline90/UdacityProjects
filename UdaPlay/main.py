from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from udaplay.agent import GameResearchAgent
from udaplay.config import load_config
from udaplay.ingestion import load_games
from udaplay.vector_store import GameVectorStore


def build_store(config):
    return GameVectorStore(
        persist_directory=config.chroma_path,
        collection_name=config.chroma_collection,
        embedding_model=config.embedding_model,
        openai_api_key=config.openai_api_key,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="UdaPlay AI Game Research Agent")
    parser.add_argument("--index", action="store_true", help="Index JSON files and exit")
    parser.add_argument("--ask", type=str, help="Ask one question and exit")
    parser.add_argument("--local-only", action="store_true", help="Do not use web search")
    args = parser.parse_args()

    config = load_config()
    store = build_store(config)

    if args.index:
        games = load_games(config.games_dir)
        if not games:
            print(f"No game records found in {config.games_dir}. Add JSON files first.")
            return 1
        count = store.upsert_games(games)
        print(f"Indexed {count} game records into '{config.chroma_collection}'.")
        return 0

    if store.count() == 0:
        games = load_games(config.games_dir)
        if games:
            count = store.upsert_games(games)
            print(f"Indexed {count} sample game records.")
        else:
            print(f"No indexed games and no JSON records found in {config.games_dir}. Run with --index after adding data.")
            return 1

    agent = GameResearchAgent(config=config, store=store, enable_web=not args.local_only)

    if args.ask:
        result = agent.ask(args.ask)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return 0

    print("UdaPlay Game Research Agent. Ask a question, or type 'exit' to quit.")
    print("Tip: use --local-only to disable Tavily web fallback.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            answer = agent.ask(question)
            print("\nUdaPlay:")
            print(answer.answer)
            if answer.sources:
                print("\nSources:")
                for source in answer.sources:
                    print(f"- {source.title}: {source.url}")
            print(f"\nUsed web search: {answer.used_web_search}")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
