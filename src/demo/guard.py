"""CausalArmor guard node for LangGraph.

Sits between the LLM and tool execution.  When ``causal_armor_enabled`` is
``True`` (read from the LangGraph configurable), it intercepts proposed tool
calls, runs LOO attribution via a real vLLM proxy, and replaces malicious
actions.  When the toggle is ``False`` the node is a transparent pass-through.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langsmith import trace as ls_trace

from causal_armor import (
    CausalArmorConfig,
    CausalArmorMiddleware,
    DefenseResult,
)
from causal_armor.providers.gemini import GeminiActionProvider, GeminiSanitizerProvider
from causal_armor.providers.vllm import VLLMProxyProvider

from demo.adapters import (
    causal_armor_to_lc_tool_call,
    langchain_to_causal_armor,
    lc_tool_call_to_causal_armor,
)
from demo.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool declarations for the Gemini action provider (function calling)
# ---------------------------------------------------------------------------

_GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "book_flight",
                "description": "Book a specific flight for a passenger.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {"type": "string", "description": "Flight identifier"},
                        "passenger": {"type": "string", "description": "Passenger name"},
                    },
                    "required": ["flight_id", "passenger"],
                },
            },
            {
                "name": "send_money",
                "description": "Transfer money to an external account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "Amount to transfer"},
                        "account": {"type": "string", "description": "Destination account"},
                    },
                    "required": ["amount", "account"],
                },
            },
            {
                "name": "read_travel_plan",
                "description": "Read and extract text from a travel-plan PDF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to PDF file"},
                    },
                    "required": ["file_path"],
                },
            },
        ]
    }
]

# ---------------------------------------------------------------------------
# Middleware factory
# ---------------------------------------------------------------------------

_UNTRUSTED_TOOLS = frozenset({"read_travel_plan"})


def _build_middleware() -> CausalArmorMiddleware:
    """Build CausalArmor middleware with real providers.

    All model names and vLLM base URL are read from CAUSAL_ARMOR_* env vars
    (loaded from .env by the causal_armor package).
    """
    return CausalArmorMiddleware(
        action_provider=GeminiActionProvider(tools=_GEMINI_TOOLS),
        proxy_provider=VLLMProxyProvider(),
        sanitizer_provider=GeminiSanitizerProvider(),
        config=CausalArmorConfig.from_env(),
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
        logger.info("[GUARD] CausalArmor DISABLED — passing through")
        return {"messages": []}

    # ---- Run CausalArmor defence ----
    logger.info("[GUARD] CausalArmor ENABLED — running LOO attribution")

    ca_messages = langchain_to_causal_armor(messages)

    # Debug: log converted messages so we can verify untrusted spans
    tool_msgs = [m for m in ca_messages if m.role.value == "tool"]
    logger.info(f"[GUARD] Context has {len(ca_messages)} messages, {len(tool_msgs)} tool results")
    for tm in tool_msgs:
        logger.info(f"[GUARD]   tool_name={tm.tool_name} content_len={len(tm.content)}")

    middleware = _build_middleware()

    defended_tool_calls: list[dict] = []
    results: list[DefenseResult] = []

    try:
        for tc in last_msg.tool_calls:
            ca_tc = lc_tool_call_to_causal_armor(tc)

            with ls_trace(
                name=f"loo_attribution:{tc['name']}",
                run_type="chain",
                inputs={
                    "tool_name": tc["name"],
                    "tool_args": tc["args"],
                },
            ) as rt:
                result = await middleware.guard(
                    ca_messages,
                    ca_tc,
                    untrusted_tool_names=_UNTRUSTED_TOOLS,
                )

                # Build trace output metadata
                trace_output: dict = {
                    "was_defended": result.was_defended,
                    "original_action": {
                        "name": result.original_action.name,
                        "arguments": result.original_action.arguments,
                    },
                    "final_action": {
                        "name": result.final_action.name,
                        "arguments": result.final_action.arguments,
                    },
                }
                if result.detection:
                    attr = result.detection.attribution
                    trace_output["detection"] = {
                        "delta_user_normalized": round(attr.delta_user_normalized, 4),
                        "span_attributions_normalized": {
                            k: round(v, 4)
                            for k, v in attr.span_attributions_normalized.items()
                        },
                        "is_attack_detected": result.detection.is_attack_detected,
                        "flagged_spans": list(result.detection.flagged_spans),
                        "margin_tau": result.detection.margin_tau,
                    }
                else:
                    trace_output["detection"] = None
                    trace_output["skipped_reason"] = "no untrusted spans in context"
                rt.outputs = trace_output

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
            logger.info(f"[GUARD] BLOCKED: {orig.name}({orig.arguments})")
            logger.info(f"[GUARD] REPLACED WITH: {final.name}({final.arguments})")
            if result.detection:
                attr = result.detection.attribution
                logger.info(
                    f"[GUARD] LOO scores — delta_user: {attr.delta_user_normalized:.3f}, "
                    f"span deltas: {attr.span_attributions_normalized}"
                )
        else:
            logger.info(f"[GUARD] PASSED: {orig.name}({orig.arguments})")

    return {"messages": [new_msg]}
