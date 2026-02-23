"""Configurable LangGraph agent builder.

Unlike :mod:`demo.agent` (hardcoded for the travel demo), this module
accepts arbitrary tools, a guard node callable, a system prompt, and an
agent model name so any consumer can build a guarded agent graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from demo.state import AgentState, GraphConfig

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools available to you to "
    "complete the user's request. Be precise and follow instructions."
)

_DEFAULT_AGENT_MODEL = "gemini-2.0-flash"


def build_configurable_agent(
    tools: list[BaseTool],
    guard_node: Callable,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    agent_model: str = _DEFAULT_AGENT_MODEL,
) -> StateGraph:
    """Build and compile a LangGraph agent with an arbitrary guard node.

    Parameters
    ----------
    tools:
        LangChain tools to bind to the agent LLM.
    guard_node:
        Async callable with signature ``(state, config) -> dict``.
        Typically :meth:`GuardNodeFactory.guard_node`.
    system_prompt:
        System prompt prepended to every conversation.
    agent_model:
        Gemini model name for the agent LLM.

    Returns
    -------
    Compiled :class:`~langgraph.graph.StateGraph` with
    ``llm -> should_continue -> guard -> tools -> llm`` loop.
    """

    def llm_node(state: AgentState, config: RunnableConfig) -> dict:
        configurable = config.get("configurable", {})
        model_name = configurable.get("agent_model", agent_model)

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
        ).bind_tools(tools)

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = model.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["guard", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "guard"
        return "__end__"

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState, config_schema=GraphConfig)

    graph.add_node("llm", llm_node)
    graph.add_node("guard", guard_node)
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
