"""CausalArmor guard node for LangGraph.

Sits between the LLM and tool execution.  When ``causal_armor_enabled`` is
``True`` (read from the LangGraph configurable), it intercepts proposed tool
calls, runs LOO attribution via a real vLLM proxy, and replaces malicious
actions.  When the toggle is ``False`` the node is a transparent pass-through.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from causal_armor import (
    CausalArmorConfig,
    CausalArmorMiddleware,
    DefenseResult,
    Message,
    MessageRole,
    ToolCall,
)
from causal_armor.providers.gemini import GeminiSanitizerProvider
from causal_armor.providers.vllm import VLLMProxyProvider

from demo.adapters import (
    causal_armor_to_lc_tool_call,
    langchain_to_causal_armor,
    lc_tool_call_to_causal_armor,
)
from demo.state import AgentState

# ---------------------------------------------------------------------------
# Mock action provider — sufficient for demo (sanitizer is real Gemini)
# ---------------------------------------------------------------------------


class MockActionProvider:
    """Regenerates a safe action after sanitisation."""

    async def generate(
        self, messages: Sequence[Message]
    ) -> tuple[str, list[ToolCall]]:
        action = ToolCall(
            name="book_flight",
            arguments={"flight_id": "AA1742", "passenger": "Alex Johnson"},
            raw_text=json.dumps(
                {
                    "name": "book_flight",
                    "arguments": {"flight_id": "AA1742", "passenger": "Alex Johnson"},
                }
            ),
        )
        return (action.raw_text, [action])


# ---------------------------------------------------------------------------
# Middleware factory
# ---------------------------------------------------------------------------

_UNTRUSTED_TOOLS = frozenset({"read_travel_plan"})


def _build_middleware() -> CausalArmorMiddleware:
    """Build CausalArmor middleware.

    Model names and vLLM base URL are read from CAUSAL_ARMOR_* env vars
    (loaded from .env by the causal_armor package).
    """
    return CausalArmorMiddleware(
        action_provider=MockActionProvider(),
        proxy_provider=VLLMProxyProvider(),
        sanitizer_provider=GeminiSanitizerProvider(),
        config=CausalArmorConfig(margin_tau=0.0),
    )


# ---------------------------------------------------------------------------
# Guard node
# ---------------------------------------------------------------------------


async def causal_armor_guard(state: AgentState, config: RunnableConfig) -> dict:
    """LangGraph node: conditionally run CausalArmor defence on tool calls.

    Reads ``causal_armor_enabled`` from the configurable namespace.  When
    disabled the node passes tool calls through unchanged.
    """
    configurable = config.get("configurable", {})
    enabled = configurable.get("causal_armor_enabled", True)

    messages = state["messages"]
    last_msg = messages[-1]

    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": []}

    # ---- Pass-through when disabled ----
    if not enabled:
        print("    [GUARD] CausalArmor DISABLED — passing through")
        return {"messages": []}

    # ---- Run CausalArmor defence ----
    print("    [GUARD] CausalArmor ENABLED — running LOO attribution")

    ca_messages = langchain_to_causal_armor(messages)
    middleware = _build_middleware()

    defended_tool_calls: list[dict] = []
    results: list[DefenseResult] = []

    try:
        for tc in last_msg.tool_calls:
            ca_tc = lc_tool_call_to_causal_armor(tc)
            result = await middleware.guard(
                ca_messages,
                ca_tc,
                untrusted_tool_names=_UNTRUSTED_TOOLS,
            )
            results.append(result)

            if result.was_defended:
                defended_tool_calls.append(
                    causal_armor_to_lc_tool_call(result.final_action)
                )
            else:
                defended_tool_calls.append(tc)
    finally:
        await middleware.close()

    # Build a replacement AIMessage with defended tool calls
    new_msg = AIMessage(
        content=last_msg.content,
        tool_calls=defended_tool_calls,
        id=last_msg.id,
    )

    for result in results:
        orig = result.original_action
        final = result.final_action
        if result.was_defended:
            print(f"    [GUARD] BLOCKED: {orig.name}({orig.arguments})")
            print(f"    [GUARD] REPLACED WITH: {final.name}({final.arguments})")
            if result.detection:
                attr = result.detection.attribution
                print(f"    [GUARD] LOO scores — delta_user: {attr.delta_user_normalized:.3f}, "
                      f"span deltas: {attr.span_attributions_normalized}")
        else:
            print(f"    [GUARD] PASSED: {orig.name}({orig.arguments})")

    return {"messages": [new_msg]}
