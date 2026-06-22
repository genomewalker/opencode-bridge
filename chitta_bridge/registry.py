from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mcp.types import Tool


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable
    hidden: bool = False
    category: str = "core"

    def as_mcp_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, inputSchema=self.input_schema)


REGISTRY: dict[str, ToolDef] = {}


def register(name: str, schema: dict, *, hidden: bool = False, category: str = "core"):
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = ToolDef(
            name=name,
            description=(fn.__doc__ or "").strip(),
            input_schema=schema,
            handler=fn,
            hidden=hidden,
            category=category,
        )
        return fn

    return deco
