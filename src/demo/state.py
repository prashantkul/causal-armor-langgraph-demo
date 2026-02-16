"""Agent state and config definitions for the LangGraph travel agent."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphConfig(TypedDict, total=False):
    """Configurable settings surfaced in LangGraph Studio UI."""

    causal_armor_enabled: bool
    agent_model: str


class AgentState(TypedDict):
    """Standard LangGraph message-accumulating state."""

    messages: Annotated[list[AnyMessage], add_messages]
