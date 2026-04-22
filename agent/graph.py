"""
LangGraph graph definition for the AutoStream AI sales agent.

Graph topology:
  START
    └─► classify_intent
          ├─► [greeting]         → generate_greeting        → END
          ├─► [product_inquiry]  → rag_retrieval
          │                          └─► generate_product_response → END
          ├─► [high_intent]      → start_lead_collection    → END
          └─► [collecting_lead]  → extract_lead_info
                                      ├─► [incomplete] → ask_for_lead_info → END
                                      └─► [complete]   → capture_lead      → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    classify_intent_node,
    rag_retrieval_node,
    generate_greeting_node,
    generate_product_response_node,
    start_lead_collection_node,
    extract_lead_info_node,
    ask_for_lead_info_node,
    capture_lead_node,
)


def _route_intent(state: AgentState) -> str:
    """Route after intent classification."""
    intent = state.get("intent", "product_inquiry")
    if intent == "greeting":
        return "generate_greeting"
    elif intent == "high_intent":
        return "start_lead_collection"
    elif intent == "collecting_lead":
        return "extract_lead_info"
    else:
        return "rag_retrieval"


def _route_lead_collection(state: AgentState) -> str:
    """Route after lead extraction — go to capture if all fields present."""
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    if name and email and platform:
        return "capture_lead"
    return "ask_for_lead_info"


def build_graph():
    """Build and compile the AutoStream agent graph with in-memory checkpointing."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("rag_retrieval", rag_retrieval_node)
    builder.add_node("generate_greeting", generate_greeting_node)
    builder.add_node("generate_product_response", generate_product_response_node)
    builder.add_node("start_lead_collection", start_lead_collection_node)
    builder.add_node("extract_lead_info", extract_lead_info_node)
    builder.add_node("ask_for_lead_info", ask_for_lead_info_node)
    builder.add_node("capture_lead", capture_lead_node)

    # Entry point
    builder.add_edge(START, "classify_intent")

    # Intent router
    builder.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "generate_greeting": "generate_greeting",
            "rag_retrieval": "rag_retrieval",
            "start_lead_collection": "start_lead_collection",
            "extract_lead_info": "extract_lead_info",
        },
    )

    # RAG → product response
    builder.add_edge("rag_retrieval", "generate_product_response")

    # Terminal response nodes
    builder.add_edge("generate_greeting", END)
    builder.add_edge("generate_product_response", END)
    builder.add_edge("start_lead_collection", END)

    # Lead collection router
    builder.add_conditional_edges(
        "extract_lead_info",
        _route_lead_collection,
        {
            "ask_for_lead_info": "ask_for_lead_info",
            "capture_lead": "capture_lead",
        },
    )

    builder.add_edge("ask_for_lead_info", END)
    builder.add_edge("capture_lead", END)

    # Compile with in-memory persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
