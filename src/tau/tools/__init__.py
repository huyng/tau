"""Built-in tool set: read, bash, edit, write — modeled after pi's defaults."""

from .base import Tool, ToolResult
from .bash import BashTool
from .edit import EditTool, WriteTool
from .read import ReadTool
from .validation import validate_arguments

__all__ = ["Tool", "ToolResult", "default_tools", "validate_arguments"]


def default_tools(cwd: str | None = None) -> list[Tool]:
    return [
        ReadTool(cwd=cwd),
        BashTool(cwd=cwd),
        EditTool(cwd=cwd),
        WriteTool(cwd=cwd),
    ]
