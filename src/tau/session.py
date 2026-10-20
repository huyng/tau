"""Session persistence: append-only JSONL with a tree-shaped entry history.

Modeled after pi's session format: each entry has an id and refers to its
parent, so messages form a tree whose active branch supplies history for the
next model request. The active leaf is recorded with pointer lines; only the
last pointer in the file matters, which keeps the format append-only.
"""


import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from .messages import AgentMessage, deserialize_message, new_id, serialize_message

SESSIONS_DIR = Path(
    os.environ.get("TAU_SESSIONS_DIR", Path.home() / ".tau" / "sessions")
)

# The sessions directory is shared with other tools (notably the full tau CLI),
# which lay their sessions out as `<cwd-slug>-<hash>/<uuid4>.jsonl` alongside an
# `index.jsonl`. Only files this module itself writes are ours: a 12-hex id in a
# YYYY-MM-DD directory. Anything else uses a different entry schema and would
# load as an empty session, so it must not show up in the listing.
SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SessionManager:
    def __init__(self, directory: Path | None = None, *, in_memory: bool = False):
        self.directory = directory if directory is not None else SESSIONS_DIR
        self.in_memory = in_memory
        self.file: Path | None = None
        self.entries: dict[str, dict] = {}
        self.current_id: str | None = None

    # -- construction ----------------------------------------------------------

    def new_session(self) -> None:
        self.entries = {}
        self.current_id = None
        if self.in_memory:
            return
        date = datetime.now().strftime("%Y-%m-%d")
        self.directory.joinpath(date).mkdir(parents=True, exist_ok=True)
        self.file = self.directory / date / f"{uuid.uuid4().hex[:12]}.jsonl"

    def load(self, path: Path) -> None:
        self.file = path
        self.entries = {}
        self.current_id = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "current" in data:
                    self.current_id = data["current"]
                else:
                    self.entries[data["id"]] = data

    # -- queries ---------------------------------------------------------------

    def active_branch(self) -> list[AgentMessage]:
        """Messages from the root down to the current entry."""
        branch: list[AgentMessage] = []
        entry_id = self.current_id
        while entry_id is not None:
            entry = self.entries[entry_id]
            branch.append(deserialize_message(entry["message"]))
            entry_id = entry["parent"]
        branch.reverse()
        return branch

    @property
    def title(self) -> str:
        for message in self.active_branch():
            data = serialize_message(message)
            if data["role"] == "user":
                text = data["content"].replace("\n", " ")
                return text[:60] + ("…" if len(text) > 60 else "")
        return "(empty session)"

    @staticmethod
    def list_sessions(directory: Path | None = None) -> list[Path]:
        directory = directory if directory is not None else SESSIONS_DIR
        if not directory.is_dir():
            return []
        sessions = [
            p
            for p in directory.glob("*/*.jsonl")
            if p.is_file()
            and DATE_DIR_RE.match(p.parent.name)
            and SESSION_ID_RE.match(p.stem)
        ]
        return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)

    # -- writes ----------------------------------------------------------------

    def compact(self, summary: AgentMessage, keep: list[AgentMessage]) -> None:
        """Re-root the active branch at `summary`, then replay `keep`.

        Compaction is a *branch* operation, not a delete: the previous entries
        stay in `self.entries` and in the JSONL file, so the full transcript
        survives on disk. Only the active leaf moves.
        """
        self.current_id = None
        self.append(summary)
        for message in keep:
            self.append(message)

    def append(self, message: AgentMessage) -> str:
        entry_id = new_id()
        entry = {
            "id": entry_id,
            "parent": self.current_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": serialize_message(message),
        }
        self.entries[entry_id] = entry
        self.current_id = entry_id
        if not self.in_memory and self.file is not None:
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.write(json.dumps({"current": entry_id}) + "\n")
        return entry_id
