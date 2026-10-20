import os
import stat

from .base import Tool, ToolResult

MAX_LINES = 2000
MAX_BYTES = 50 * 1024


class ReadTool(Tool):
    name="read"

    description = (
        "Read the contents of a text file. Output is truncated to 2000 lines "
        "or 50KB (whichever is hit first). Use offset/limit for large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file (relative or absolute)"},
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, path: str, offset: float | None = None, limit: float | None = None,
                **_) -> ToolResult:
        path = os.path.expanduser(path)
        if not os.path.isabs(path) and self.cwd:
            path = os.path.join(self.cwd, path)
        try:
            if not os.path.isfile(path):
                return ToolResult(f"Error: not a file: {path}", is_error=True)
            if stat.S_ISFIFO(os.stat(path).st_mode) or _looks_binary(path):
                return ToolResult(f"Error: {path} is not a text file", is_error=True)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            return ToolResult(f"Error reading {path}: {exc}", is_error=True)

        start = max(int(offset or 1), 1) - 1
        end = start + int(limit) if limit else len(lines)
        selected = lines[start:end]

        total_bytes = sum(len(line.encode("utf-8")) for line in selected)
        truncated = ""
        if total_bytes > MAX_BYTES:
            kept: list[str] = []
            used = 0
            for line in selected:
                used += len(line.encode("utf-8"))
                if used > MAX_BYTES:
                    break
                kept.append(line)
            selected = kept
            truncated = f"\n... output truncated at {MAX_BYTES // 1024}KB"
        if end < len(lines) and not truncated:
            truncated = f"\n... ({len(lines) - end} more lines)"

        return ToolResult("".join(selected) + truncated)


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\x00" in chunk
    except OSError:
        return False
