"""LangGraph agent builder — single graph with configurable CausalArmor toggle.

Uses Gemini (via ``langchain-google-genai``) as the agent LLM.  All model
names are configurable via the LangGraph ``configurable`` dict so they can
be changed at invocation time or through LangGraph Studio's UI.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from demo.guard import causal_armor_guard
from demo.state import AgentState, GraphConfig
from demo.tools import book_flight, read_travel_plan, send_money

ALL_TOOLS = [read_travel_plan, book_flight, send_money]

AGENT_MODEL_DEFAULT = "gemini-3-pro-preview"


# ---------------------------------------------------------------------------
# LLM node — reads agent_model from configurable
# ---------------------------------------------------------------------------


def llm_node(state: AgentState, config: RunnableConfig) -> dict:
    """Invoke the agent LLM with tool bindings."""
    configurable = config.get("configurable", {})
    model_name = configurable.get("agent_model", AGENT_MODEL_DEFAULT)

    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
    ).bind_tools(ALL_TOOLS)

    response = model.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> Literal["guard", "__end__"]:
    """Route after LLM: if tool calls exist go to guard, else end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "guard"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent():
    """Build and compile the travel agent graph.

    The graph always includes the guard node.  All model choices are
    read from the ``configurable`` dict at runtime:

    - ``agent_model``           — Gemini model for the agent LLM
    - ``proxy_model``           — vLLM model for LOO scoring
    - ``sanitizer_model``       — Gemini model for sanitisation
    - ``causal_armor_enabled``  — toggle guard on/off
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState, config_schema=GraphConfig)

    graph.add_node("llm", llm_node)
    graph.add_node("guard", causal_armor_guard)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {"guard": "guard", "__end__": END},
    )
    graph.add_edge("guard", "tools")
    graph.add_edge("tools", "llm")

    return graph.compile()
