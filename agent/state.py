"""
Agent state definition for the AutoStream LangGraph agent.

LangGraph persists this state across conversation turns using MemorySaver.
The `add_messages` annotation tells LangGraph to APPEND new messages to the
list rather than overwrite it — which is essential for multi-turn memory.
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full conversation history — new messages are appended automatically
    messages: Annotated[List[BaseMessage], add_messages]

    # Classified intent for the current turn
    # Values: "greeting" | "product_inquiry" | "high_intent" | "collecting_lead"
    intent: str

    # Top-k relevant chunks retrieved from the knowledge base
    retrieved_context: str

    # Flag: are we currently in the lead collection loop?
    collecting_lead: bool

    # Lead fields — populated incrementally across turns
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Set to True after mock_lead_capture() is called successfully
    lead_captured: bool

    # The agent's latest response text (used by main.py to print output)
    response: str
