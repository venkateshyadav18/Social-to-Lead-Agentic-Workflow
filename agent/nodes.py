"""
LangGraph node functions for the AutoStream agent.

Each node receives the full AgentState and returns a PARTIAL state update
(a dict containing only the keys that changed). LangGraph merges this
partial update into the persisted state automatically.

Node responsibilities:
  classify_intent       → detect what the user wants this turn
  rag_retrieval         → fetch relevant knowledge base chunks
  generate_greeting     → respond to casual openers
  generate_product_resp → answer product/pricing questions with RAG context
  start_lead_collection → transition to sign-up flow, ask for first field
  extract_lead_info     → parse name / email / platform from user message
  ask_for_lead_info     → ask for the next missing field
  capture_lead          → call mock_lead_capture() and confirm success
"""

import json
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AgentState
from agent.prompts import (
    SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    LEAD_EXTRACTOR_PROMPT,
    GREETING_RESPONSE_PROMPT,
    PRODUCT_RESPONSE_PROMPT,
    LEAD_COLLECTION_PROMPT,
    LEAD_CAPTURE_SUCCESS_PROMPT,
)
from rag.retriever import KnowledgeRetriever
from tools.lead_capture import mock_lead_capture


# ------------------------------------------------------------------ #
#  Singletons — initialised once and reused across all turns           #
# ------------------------------------------------------------------ #

_llm: Optional[ChatGoogleGenerativeAI] = None
_retriever: Optional[KnowledgeRetriever] = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return _llm


def _get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


# ------------------------------------------------------------------ #
#  Utility                                                             #
# ------------------------------------------------------------------ #

def _format_history(messages: list) -> str:
    """Format all messages except the latest into a readable string."""
    if len(messages) <= 1:
        return "No prior conversation."

    lines = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _field_display(value: Optional[str]) -> str:
    """Return the value or a placeholder for prompt formatting."""
    return value if value else "Not yet collected"


# ------------------------------------------------------------------ #
#  Nodes                                                               #
# ------------------------------------------------------------------ #

def classify_intent_node(state: AgentState) -> dict:
    """
    Classify the user's current intent.

    If we're already mid-collection, skip re-classification and return
    'collecting_lead' so the graph routes to the extraction node.
    """
    if state.get("collecting_lead", False):
        return {"intent": "collecting_lead"}

    llm = _get_llm()
    latest_message = state["messages"][-1].content
    history = _format_history(state["messages"])

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        history=history,
        message=latest_message,
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )

    raw_intent = response.content.strip().lower()

    if raw_intent not in {"greeting", "product_inquiry", "high_intent"}:
        raw_intent = "product_inquiry"

    return {"intent": raw_intent}


def rag_retrieval_node(state: AgentState) -> dict:
    """Retrieve the most relevant knowledge base chunks for the user's query."""
    retriever = _get_retriever()
    query = state["messages"][-1].content
    context = retriever.retrieve(query, top_k=2)
    return {"retrieved_context": context}


def generate_greeting_node(state: AgentState) -> dict:
    """Generate a short, welcoming response to a greeting or small-talk message."""
    llm = _get_llm()
    latest_message = state["messages"][-1].content
    history = _format_history(state["messages"])

    prompt = GREETING_RESPONSE_PROMPT.format(
        history=history,
        message=latest_message,
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    ai_msg = AIMessage(content=response.content)

    return {
        "response": response.content,
        "messages": [ai_msg],
    }


def generate_product_response_node(state: AgentState) -> dict:
    """Generate a grounded product/pricing response using RAG context."""
    llm = _get_llm()
    latest_message = state["messages"][-1].content
    history = _format_history(state["messages"])
    context = state.get("retrieved_context", "")

    prompt = PRODUCT_RESPONSE_PROMPT.format(
        context=context,
        history=history,
        message=latest_message,
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    ai_msg = AIMessage(content=response.content)

    return {
        "response": response.content,
        "messages": [ai_msg],
    }


def start_lead_collection_node(state: AgentState) -> dict:
    """
    Transition into lead-collection mode.

    Sets `collecting_lead = True` and asks for the first missing field.
    This node fires once when high intent is first detected.
    """
    llm = _get_llm()

    prompt = LEAD_COLLECTION_PROMPT.format(
        name=_field_display(state.get("lead_name")),
        email=_field_display(state.get("lead_email")),
        platform=_field_display(state.get("lead_platform")),
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    ai_msg = AIMessage(content=response.content)

    return {
        "collecting_lead": True,
        "response": response.content,
        "messages": [ai_msg],
    }


def extract_lead_info_node(state: AgentState) -> dict:
    """
    Parse name / email / platform out of the user's latest message.

    Uses the LLM to extract structured JSON and merges new values with
    whatever was already collected in previous turns.
    """
    llm = _get_llm()
    latest_message = state["messages"][-1].content

    prompt = LEAD_EXTRACTOR_PROMPT.format(
        name=_field_display(state.get("lead_name")),
        email=_field_display(state.get("lead_email")),
        platform=_field_display(state.get("lead_platform")),
        message=latest_message,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    new_name = state.get("lead_name")
    new_email = state.get("lead_email")
    new_platform = state.get("lead_platform")

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", response.content).strip()
        extracted = json.loads(cleaned)

        if extracted.get("name") and extracted["name"] not in (None, "null", ""):
            new_name = str(extracted["name"]).strip()
        if extracted.get("email") and extracted["email"] not in (None, "null", ""):
            new_email = str(extracted["email"]).strip()
        if extracted.get("platform") and extracted["platform"] not in (None, "null", ""):
            new_platform = str(extracted["platform"]).strip()

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"[WARN] Lead extraction JSON parse failed: {e}")

    return {
        "lead_name": new_name,
        "lead_email": new_email,
        "lead_platform": new_platform,
    }


def ask_for_lead_info_node(state: AgentState) -> dict:
    """Ask for the next missing lead field in a natural, conversational way."""
    llm = _get_llm()

    prompt = LEAD_COLLECTION_PROMPT.format(
        name=_field_display(state.get("lead_name")),
        email=_field_display(state.get("lead_email")),
        platform=_field_display(state.get("lead_platform")),
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    ai_msg = AIMessage(content=response.content)

    return {
        "response": response.content,
        "messages": [ai_msg],
    }


def capture_lead_node(state: AgentState) -> dict:
    """
    Call mock_lead_capture() once all three fields are confirmed,
    then generate a warm success confirmation for the user.
    """
    llm = _get_llm()

    name = state["lead_name"]
    email = state["lead_email"]
    platform = state["lead_platform"]

    mock_lead_capture(name=name, email=email, platform=platform)

    prompt = LEAD_CAPTURE_SUCCESS_PROMPT.format(
        name=name,
        email=email,
        platform=platform,
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    ai_msg = AIMessage(content=response.content)

    return {
        "lead_captured": True,
        "collecting_lead": False,
        "response": response.content,
        "messages": [ai_msg],
    }
