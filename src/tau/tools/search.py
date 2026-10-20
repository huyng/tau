"""Search tools: `grep` for file contents and `glob` for file paths.

Both walk the tree in plain Python, so tau needs no ripgrep dependency. Two
properties matter more than raw speed:

- **Bounded output.** A broad query must never flood the transcript. Every
  result is capped (`head_limit`, `MAX_LINE_CHARS`) and the tool says so when
  it truncated, instead of silently handing back megabytes.
- **Low noise.** Directories that dominate real repositories (`.git`,
  `node_modules`, `.venv`, `__pycache__`, build output, ...) are pruned, and
  binary files are skipped by content sniff rather than by extension guess.

Paths come back relative to the working directory so results stay short and
copy-pasteable into `read` / `edit`.
"""

import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path, PurePath
from typing import Any

from .base import Tool, ToolResult

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".jj",
    "node_modules", "bower_components",
    ".venv", "venv", "env", ".env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".cache", ".gradle", ".next", ".nuxt", ".svelte-kit", ".turbo",
    "dist", "build", "target", "out", ".idea", ".vscode",
})

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_LINE_CHARS = 400
DEFAULT_HEAD_LIMIT = 250
DEFAULT_GLOB_LIMIT = 200

_MODES = ("content", "files_with_matches", "count")


def _resolve(path: str | None, cwd: str | None) -> Path:
    text = os.path.expanduser(path) if path else "."
    if not os.path.isabs(text) and cwd:
        text = os.path.join(cwd, text)
    return Path(text)


def _display(path: Path, cwd: str | None) -> str:
    """Prefer a cwd-relative path, with forward slashes on every platform."""
    shown = str(path)
    if cwd:
        try:
            relative = os.path.relpath(path, cwd)
        except ValueError:  # different drive on Windows
            relative = shown
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            shown = relative
    return shown.replace(os.sep, "/")


def _matches_pattern(relative: str, pattern: str) -> bool:
    """Match a repo-relative path against a glob.

    Patterns with a slash are matched against the whole path with
    `PurePath.full_match`, so `**/*.py` also matches top-level `a.py`.
    Patterns without one match the basename, so `*.py` does what users expect.
    """
    if "/" in pattern:
        return PurePath(relative).full_match(pattern)
    return fnmatch.fnmatchcase(PurePath(relative).name, pattern)


def _iter_files(root: Path, pattern: str | None = None) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Hidden directories are skipped, matching ripgrep's default; hidden
        # *files* stay visible so `.gitignore`-style config is searchable.
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in SKIP_DIRS and not name.startswith(".")
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if pattern is not None:
                relative = path.relative_to(root).as_posix()
                if not _matches_pattern(relative, pattern):
                    continue
            yield path


def _read_text(path: Path) -> str | None:
    """File contents, or None for binaries, oversized files, and IO errors."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:1024]:
        return None
    return data.decode("utf-8", errors="replace")


def _content_lines(relative: str, lines: list[str], hits: list[int], context: int) -> list[str]:
    """`path:line:text` for matches, `path-line-text` for context lines."""
    out: list[str] = []
    matched = set(hits)
    emitted_to = -1
    for index in hits:
        start = max(index - context, 0)
        end = min(index + context, len(lines) - 1)
        if emitted_to >= 0 and start > emitted_to + 1:
            out.append("--")
        for line_no in range(max(start, emitted_to + 1), end + 1):
            marker = ":" if line_no in matched else "-"
            body = lines[line_no].rstrip()
            if len(body) > MAX_LINE_CHARS:
                body = body[:MAX_LINE_CHARS] + "…"
            out.append(f"{relative}{marker}{line_no + 1}{marker} {body}")
        emitted_to = max(emitted_to, end)
    return out


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression (Python `re` syntax). "
        "Output lines look like `path:line: text`; context lines use `-` "
        "instead of `:`. Binary files, hidden directories and noise like "
        ".git / node_modules / .venv are skipped. Prefer this over running "
        "rg or grep through `bash`: results are compact and cannot flood the "
        "transcript. Narrow with `glob` or `path` when output is truncated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Python regular expression to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: working directory)",
            },
            "glob": {
                "type": "string",
                "description": "Only search files matching this glob, e.g. '**/*.py'",
            },
            "output_mode": {
                "type": "string",
                "description": (
                    "'content' (default) shows matching lines, "
                    "'files_with_matches' lists paths only, 'count' shows "
                    "per-file match counts"
                ),
            },
            "context": {
                "type": "integer",
                "description": "Lines of context to show before and after each match (default 0)",
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive search (default false)",
            },
            "head_limit": {
                "type": "integer",
                "description": f"Maximum output lines (default {DEFAULT_HEAD_LIMIT})",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "content",
        context: int | None = None,
        ignore_case: bool = False,
        head_limit: int | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return ToolResult(f"Error: invalid regular expression {pattern!r}: {exc}", is_error=True)

        root = _resolve(path, self.cwd)
        shown_root = _display(root, self.cwd)
        if not root.exists():
            return ToolResult(f"Error: no such path: {shown_root}", is_error=True)

        mode = output_mode if output_mode in _MODES else "content"
        context_lines = max(context or 0, 0)
        limit = max(head_limit or DEFAULT_HEAD_LIMIT, 1)

        files: Iterator[Path] = (
            iter([root]) if root.is_file() else _iter_files(root, glob or None)
        )

        records: list[str] = []
        matches = 0
        files_hit = 0
        truncated = False
        for file in files:
            text = _read_text(file)
            if not text:
                continue
            lines = text.splitlines()
            hits = [index for index, line in enumerate(lines) if regex.search(line)]
            if not hits:
                continue
            matches += len(hits)
            files_hit += 1
            relative = _display(file, self.cwd)
            if mode == "files_with_matches":
                records.append(relative)
            elif mode == "count":
                records.append(f"{relative}:{len(hits)}")
            else:
                records.extend(_content_lines(relative, lines, hits, context_lines))
            if len(records) > limit:
                truncated = True
                records = records[:limit]
                break

        if not matches:
            scope = f" matching {glob}" if glob else ""
            return ToolResult(f"No matches for /{pattern}/ in {shown_root}{scope}")

        body = "\n".join(records)
        if truncated:
            body += (
                f"\n... truncated at {limit} lines; narrow the search with `glob`, "
                "`path`, or a more specific pattern"
            )
        elif mode == "files_with_matches" and files_hit > 1:
            body += f"\n... {files_hit} files, {matches} matches"
        return ToolResult(body)


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by glob pattern, newest first. Supports `**` for any "
        "number of directories (`**/*.py`), and matches the basename when the "
        "pattern has no slash (`*.md`). Same skips as `grep`: hidden "
        "directories, .git, node_modules, .venv, build output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: working directory)",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of paths (default {DEFAULT_GLOB_LIMIT})",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(
        self,
        pattern: str,
        path: str | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> ToolResult:
        root = _resolve(path, self.cwd)
        shown_root = _display(root, self.cwd)
        if not root.exists():
            return ToolResult(f"Error: no such path: {shown_root}", is_error=True)

        limit_count = max(limit or DEFAULT_GLOB_LIMIT, 1)
        found = (
            [root] if root.is_file()
            else list(_iter_files(root, pattern))
        )
        if not found:
            return ToolResult(f"No files match {pattern!r} in {shown_root}")

        found.sort(key=_mtime, reverse=True)  # newest first
        shown = found[:limit_count]
        body = "\n".join(_display(file, self.cwd) for file in shown)
        if len(found) > limit_count:
            body += f"\n... showing {limit_count} of {len(found)} files (newest first)"
        return ToolResult(body)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
