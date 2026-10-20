"""System prompt assembly: base instructions plus discovered context files.

Like pi, tau builds the system prompt from base instructions and context
files (AGENTS.md) discovered in the working directory and its ancestors.
"""

import os

BASE_SYSTEM_PROMPT = """\
You are tau, an expert coding assistant operating inside tau, a coding agent \
harness. You help users by reading files, executing commands, editing code, \
and writing new files.

# Tools

- read: Read the contents of a file. Use offset/limit for large files.
- bash: Execute a bash command in the current working directory.
- edit: Make precise changes to a file using exact text replacement.
- write: Create a new file or completely rewrite an existing one.

# Rules

- Use bash for file operations like ls, rg, find.
- Use read to examine files instead of cat or sed.
- Use edit for precise changes. The text to replace must match exactly and be
  unique in the file. When changing multiple separate locations in one file,
  batch them into a single edit call instead of multiple edit calls.
- Use write only for new files or complete rewrites.
- Be concise in your responses.
- Show file paths clearly when working with files.

# Environment

Working directory: {cwd}
"""


def find_context_file(cwd: str | None) -> str | None:
    """Find AGENTS.md in cwd or the nearest ancestor, like pi."""
    directory = os.path.abspath(cwd or os.getcwd())
    while True:
        candidate = os.path.join(directory, "AGENTS.md")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def build_system_prompt(cwd: str | None = None) -> str:
    prompt = BASE_SYSTEM_PROMPT.format(cwd=os.path.abspath(cwd or os.getcwd()))
    context_file = find_context_file(cwd)
    if context_file:
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                instructions = f.read().strip()
            if instructions:
                prompt += f"\n# Project instructions\n\nFrom {context_file}:\n\n{instructions}\n"
        except OSError:
            pass
    return prompt
