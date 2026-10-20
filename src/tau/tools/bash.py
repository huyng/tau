from __future__ import annotations

import os
import subprocess

from .base import Tool, ToolResult

MAX_OUTPUT_LINES = 2000


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns "
        "stdout and stderr; output is truncated to the last 2000 lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, command: str, timeout: float | None = None, **_) -> ToolResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=int(timeout) if timeout else 120,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Error: command timed out after {int(timeout or 120)}s", is_error=True)

        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr
        output = output.strip()
        output += f"\n[exit code: {proc.returncode}]"

        lines = output.splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            skipped = len(lines) - MAX_OUTPUT_LINES
            output = f"... {skipped} lines skipped ...\n" + "\n".join(lines[-MAX_OUTPUT_LINES:])

        return ToolResult(output, is_error=proc.returncode != 0)
