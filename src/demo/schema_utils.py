"""Utilities to convert LangChain tool schemas to Gemini function declarations."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

# JSON Schema type → Gemini parameter type
_JSON_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _convert_schema_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Convert a single JSON Schema property to Gemini parameter format."""
    result: dict[str, Any] = {}

    json_type = prop.get("type", "string")
    # Handle anyOf (e.g. Optional types from Pydantic)
    if "anyOf" in prop:
        for variant in prop["anyOf"]:
            if variant.get("type") != "null":
                json_type = variant.get("type", "string")
                break

    result["type"] = _JSON_TYPE_MAP.get(json_type, "string")

    if "description" in prop:
        result["description"] = prop["description"]
    if "title" in prop and "description" not in result:
        result["description"] = prop["title"]
    if "enum" in prop:
        result["enum"] = prop["enum"]

    # Recurse for array items
    if result["type"] == "array" and "items" in prop:
        result["items"] = _convert_schema_property(prop["items"])

    # Recurse for nested objects
    if result["type"] == "object" and "properties" in prop:
        result["properties"] = {
            k: _convert_schema_property(v)
            for k, v in prop["properties"].items()
        }
        if "required" in prop:
            result["required"] = prop["required"]

    return result


def langchain_tools_to_gemini_declarations(
    tools: list[BaseTool],
) -> list[dict[str, Any]]:
    """Convert LangChain tools to Gemini ``function_declarations`` format.

    Returns a list with a single dict containing a ``function_declarations``
    key, matching the structure expected by
    :class:`causal_armor.providers.gemini.GeminiActionProvider`.

    Example output::

        [{"function_declarations": [
            {"name": "tool_name", "description": "...", "parameters": {...}},
            ...
        ]}]
    """
    declarations: list[dict[str, Any]] = []

    for tool in tools:
        schema = tool.get_input_schema().model_json_schema()

        properties: dict[str, Any] = {}
        for name, prop in schema.get("properties", {}).items():
            properties[name] = _convert_schema_property(prop)

        declaration: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description or tool.name,
        }

        if properties:
            declaration["parameters"] = {
                "type": "object",
                "properties": properties,
            }
            required = schema.get("required", [])
            if required:
                declaration["parameters"]["required"] = required

        declarations.append(declaration)

    return [{"function_declarations": declarations}]
