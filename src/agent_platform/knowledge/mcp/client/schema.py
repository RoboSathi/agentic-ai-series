"""A remote tool's JSON Schema, as the pydantic model the registry validates calls against.

The registry validates a model's arguments with pydantic (`ToolDefinition.args_schema`),
so an MCP tool's `inputSchema` becomes a model with one field per property. Only the
JSON types a tool argument plausibly has are mapped; anything else is accepted as-is
and left to the server, which validates its own arguments anyway.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


class _Arguments(BaseModel):
    """Every generated argument model forbids what the schema does not name."""

    model_config = ConfigDict(extra="forbid")


def arguments_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """A pydantic model with one field per property in `schema`."""
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {
        prop: _field(spec, prop in required)
        for prop, spec in (schema.get("properties") or {}).items()
    }
    return create_model(f"{name}_arguments", __base__=_Arguments, **fields)


def _field(spec: dict[str, Any], required: bool) -> tuple[Any, Any]:
    kind = _JSON_TYPES.get(str(spec.get("type")), Any)
    description = spec.get("description")
    if required:
        return kind, Field(..., description=description)
    return kind | None, Field(default=spec.get("default"), description=description)


__all__ = ["arguments_model"]
