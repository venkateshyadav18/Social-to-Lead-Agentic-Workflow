"""
AutoStream AI Sales Agent — CLI entry point.

Run:
    python main.py

The conversation runs in a loop. Type 'exit' or 'quit' to end the session.
LangGraph's MemorySaver persists state across all turns of the session.
"""

import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

# Load .env if present (optional convenience for local dev)
load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Sales Assistant  🎬              ║
║        Powered by LangGraph + Gemini 1.5 Flash       ║
╚══════════════════════════════════════════════════════╝
  Type 'exit' or 'quit' at any time to end the session.
"""


def validate_env() -> None:
    """Abort early if the API key is missing."""
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "\n[ERROR] GOOGLE_API_KEY is not set.\n"
            "Export it before running:\n"
            "  export GOOGLE_API_KEY=your-gemini-api-key\n"
            "Or add it to a .env file in the project root."
        )
        sys.exit(1)


def run_conversation(agent, thread_id: str) -> None:
    """
    Main conversation loop.

    Each user message is passed to the LangGraph agent via `agent.invoke()`.
    The `thread_id` in the config ties all invocations together so LangGraph
    loads and saves state from the same checkpoint slot.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [],
        "intent": "",
        "retrieved_context": "",
        "collecting_lead": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "response": "",
    }

    # Seed the checkpoint with the initial state
    agent.update_state(config, initial_state)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAssistant: Thanks for chatting! See you soon. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("\nAssistant: Thanks for dropping by! Feel free to come back anytime. 🎬\n")
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        response = result.get(
            "response",
            "Sorry, I didn't quite catch that — could you rephrase?",
        )
        print(f"\nAssistant: {response}\n")

        if result.get("lead_captured"):
            print("─" * 54)
            print("Session complete. Thank you for signing up for AutoStream!")
            print("─" * 54 + "\n")
            break


def main() -> None:
    validate_env()
    print(BANNER)

    print("Initializing agent (first run downloads the embedding model ~90 MB)...\n")
    agent = build_graph()
    print("Agent ready.\n")

    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    run_conversation(agent, thread_id)


if __name__ == "__main__":
    main()
