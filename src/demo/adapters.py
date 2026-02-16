"""Converters between LangChain message types and CausalArmor types."""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from causal_armor import Message, MessageRole, ToolCall

_LC_ROLE_MAP = {
    HumanMessage: MessageRole.USER,
    SystemMessage: MessageRole.SYSTEM,
    AIMessage: MessageRole.ASSISTANT,
    ToolMessage: MessageRole.TOOL,
}


def langchain_to_causal_armor(messages: Sequence[AnyMessage]) -> list[Message]:
    """Convert a sequence of LangChain messages to CausalArmor Messages."""
    result: list[Message] = []
    for msg in messages:
        role = _LC_ROLE_MAP.get(type(msg))
        if role is None:
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        tool_name: str | None = None
        tool_call_id: str | None = None
        if isinstance(msg, ToolMessage):
            tool_name = msg.name
            tool_call_id = msg.tool_call_id

        result.append(
            Message(
                role=role,
                content=content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
        )
    return result


def lc_tool_call_to_causal_armor(tc: dict) -> ToolCall:
    """Convert a single LangChain tool_call dict to a CausalArmor ToolCall."""
    return ToolCall(
        name=tc["name"],
        arguments=tc["args"],
        raw_text=json.dumps({"name": tc["name"], "arguments": tc["args"]}),
    )


def causal_armor_to_lc_tool_call(ca_tc: ToolCall) -> dict:
    """Convert a CausalArmor ToolCall back to a LangChain tool_call dict."""
    return {
        "name": ca_tc.name,
        "args": ca_tc.arguments,
        "id": f"call_{ca_tc.name}",
        "type": "tool_call",
    }
