import os
import re
import subprocess
from typing import Any

from .base import Tool, ToolResult

MAX_OUTPUT_LINES = 2000
DEFAULT_TIMEOUT = 120

#: Variables a child process never needs, and that must not reach its output.
#: The harness reads its own provider key from the environment
#: (``provider.py:211-214``), so an unscrubbed child leaks it with plain `env`.
_SCRUB = re.compile(r"KEY|TOKEN|SECRET|PASSW|CRED", re.IGNORECASE)


def _child_env() -> dict[str, str]:
    """Every variable except credentials. Deliberately not an allowlist: a
    missing PATH/GOPATH/CARGO_HOME breaks toolchains silently, which is worse
    than passing a variable the child did not need."""
    return {k: v for k, v in os.environ.items() if not _SCRUB.search(k)}


def _text(value: str | bytes | None) -> str:
    """Decode command output leniently.

    `text=True` decodes strictly, so a single non-UTF-8 byte (binary output, a
    latin-1 log, a file with junk in it) raises `UnicodeDecodeError` and throws
    away the entire result. Real subprocess output is arbitrary bytes, so
    replacement characters beat losing the command's output.
    """
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _combined(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    out, err = _text(stdout).strip(), _text(stderr).strip()
    if out and err:
        return f"{out}\n{err}"
    return out or err


def _keep_last_lines(output: str) -> str:
    lines = output.splitlines()
    if len(lines) <= MAX_OUTPUT_LINES:
        return output
    skipped = len(lines) - MAX_OUTPUT_LINES
    return f"... {skipped} lines skipped ...\n" + "\n".join(lines[-MAX_OUTPUT_LINES:])


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
            "timeout": {"type": "integer", "description": "Timeout in seconds (optional)"},
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(
        self, command: str, timeout: int | None = None, **_: Any
    ) -> ToolResult:
        timeout_seconds = timeout or DEFAULT_TIMEOUT
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                env=_child_env(),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as expired:
            # Whatever the command produced before the deadline is still
            # valuable — a build that fails at 95% has already said why.
            # (These attributes arrive as bytes even when text=True.)
            partial = _combined(expired.stdout, expired.stderr)
            header = f"Error: command timed out after {timeout_seconds}s"
            if not partial:
                return ToolResult(header, is_error=True)
            return ToolResult(
                _keep_last_lines(f"{header}\nPartial output:\n{partial}"),
                is_error=True,
            )

        output = _combined(proc.stdout, proc.stderr)
        status = f"[exit code: {proc.returncode}]"
        output = f"{output}\n{status}" if output else status
        return ToolResult(_keep_last_lines(output), is_error=proc.returncode != 0)
