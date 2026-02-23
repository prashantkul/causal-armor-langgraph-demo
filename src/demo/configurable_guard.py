"""Configurable CausalArmor guard node factory for LangGraph.

Unlike :mod:`demo.guard` (hardcoded for the travel demo), this module
accepts dynamic tool declarations and untrusted-tool sets so any consumer
can build a guard node for arbitrary tool sets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

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


@dataclass
class GuardMetrics:
    """Per-call metrics collected by the guard node."""

    tool_name: str
    was_defended: bool
    is_attack_detected: bool
    latency_seconds: float
    delta_user_normalized: float | None = None
    span_attributions: dict[str, float] = field(default_factory=dict)


class GuardNodeFactory:
    """Factory that produces a configurable CausalArmor guard node.

    Parameters
    ----------
    gemini_tool_declarations:
        Gemini ``function_declarations`` for the action regenerator.
    untrusted_tool_names:
        Tools whose results may contain prompt injections.
    config:
        Optional :class:`CausalArmorConfig` override. Falls back to
        ``CausalArmorConfig.from_env()`` when *None*.
    """

    def __init__(
        self,
        gemini_tool_declarations: list[dict[str, Any]],
        untrusted_tool_names: frozenset[str],
        config: CausalArmorConfig | None = None,
    ) -> None:
        self._tool_declarations = gemini_tool_declarations
        self._untrusted_tool_names = untrusted_tool_names
        self._config = config
        self.metrics: list[GuardMetrics] = []

    def _build_middleware(self) -> CausalArmorMiddleware:
        cfg = self._config or CausalArmorConfig.from_env()
        return CausalArmorMiddleware(
            action_provider=GeminiActionProvider(tools=self._tool_declarations),
            proxy_provider=VLLMProxyProvider(),
            sanitizer_provider=GeminiSanitizerProvider(),
            config=cfg,
        )

    async def guard_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """LangGraph node: run CausalArmor defence on proposed tool calls.

        Reads ``causal_armor_enabled`` from the configurable namespace.
        When disabled the node passes tool calls through unchanged.
        """
        configurable = config.get("configurable", {})
        enabled = configurable.get("causal_armor_enabled", True)

        messages = state["messages"]
        last_msg = messages[-1]

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": []}

        if not enabled:
            logger.info("[GUARD] CausalArmor DISABLED — passing through")
            return {"messages": []}

        logger.info("[GUARD] CausalArmor ENABLED — running LOO attribution")

        ca_messages = langchain_to_causal_armor(messages)

        tool_msgs = [m for m in ca_messages if m.role.value == "tool"]
        logger.info(
            f"[GUARD] Context has {len(ca_messages)} messages, "
            f"{len(tool_msgs)} tool results"
        )

        middleware = self._build_middleware()

        defended_tool_calls: list[dict] = []
        results: list[DefenseResult] = []

        try:
            for tc in last_msg.tool_calls:
                ca_tc = lc_tool_call_to_causal_armor(tc)

                t0 = time.monotonic()
                result = await middleware.guard(
                    ca_messages,
                    ca_tc,
                    untrusted_tool_names=self._untrusted_tool_names,
                )
                elapsed = time.monotonic() - t0

                # Collect metrics
                metric = GuardMetrics(
                    tool_name=tc["name"],
                    was_defended=result.was_defended,
                    is_attack_detected=(
                        result.detection.is_attack_detected
                        if result.detection
                        else False
                    ),
                    latency_seconds=elapsed,
                )
                if result.detection:
                    attr = result.detection.attribution
                    metric.delta_user_normalized = attr.delta_user_normalized
                    metric.span_attributions = dict(
                        attr.span_attributions_normalized
                    )
                self.metrics.append(metric)

                results.append(result)

                if result.was_defended:
                    defended_tool_calls.append(
                        causal_armor_to_lc_tool_call(result.final_action)
                    )
                else:
                    defended_tool_calls.append(tc)
        finally:
            await middleware.close()

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
                logger.info(
                    f"[GUARD] REPLACED WITH: {final.name}({final.arguments})"
                )
            else:
                logger.info(f"[GUARD] PASSED: {orig.name}({orig.arguments})")

        return {"messages": [new_msg]}
