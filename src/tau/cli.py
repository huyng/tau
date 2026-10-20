"""Interactive terminal interface, modeled after pi's REPL.

Modes:
- interactive (default): stream responses, use slash commands
- print: `tau -p \"prompt\"` runs once and writes the final response
"""

import argparse
import os
import sys

from . import __version__
from .agent import AgentSession
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolNameEvent,
)
from .prompts import build_system_prompt
from .provider import OpenAIProvider
from .session import SessionManager
from .spinner import Spinner
from .tools import default_tools

# ANSI styling
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

HELP = """\
/help              Show this help
/model [name]      Show or switch the model
/tools             List active tools
/system            Show the current system prompt
/new               Start a new session
/sessions          List recent sessions
/resume <n>        Resume the n-th session from /sessions
/usage             Show cumulative token usage
/quit              Exit (also Ctrl-D)
"""


def make_session(args: argparse.Namespace) -> AgentSession:
    provider = OpenAIProvider(model=args.model, base_url=args.base_url, api_key=args.api_key)
    manager = SessionManager()
    if args.resume:
        sessions = SessionManager.list_sessions()
        if not sessions:
            sys.exit("No sessions found.")
        manager.load(sessions[args.resume - 1])
    else:
        manager.new_session()
    return AgentSession(
        provider=provider,
        tools=default_tools(cwd=os.getcwd()),
        session_manager=manager,
        system_prompt=build_system_prompt(os.getcwd()),
        cwd=os.getcwd(),
    )


class Repl:
    def __init__(self, session: AgentSession):
        self.session = session
        self.session.subscribe(self._on_event)
        self._tool_line_open = False
        self._text_started = False
        self._reasoning_started = False
        self._spinner = Spinner()

    # -- event rendering ---------------------------------------------------------

    def _on_event(self, event) -> None:
        if isinstance(event, MessageStartEvent):
            self._text_started = False
            self._reasoning_started = False
            self._spinner.start(self.session.provider.model)
            return
        if isinstance(event, (AgentStartEvent, ToolNameEvent)):
            return
        # every remaining event can produce output: retire the spinner first
        self._spinner.stop()
        if isinstance(event, ReasoningDeltaEvent):
            print(f"{DIM}{event.delta}{RESET}", end="", flush=True)
            self._reasoning_started = True
        elif isinstance(event, TextDeltaEvent):
            if self._reasoning_started and not self._text_started:
                print()  # separate reasoning from the answer
            self._text_started = True
            print(event.delta, end="", flush=True)
        elif isinstance(event, MessageEndEvent):
            if self._text_started and not event.message.text.endswith("\n"):
                print()
        elif isinstance(event, ToolExecutionStartEvent):
            call = event.tool_call
            args_preview = _format_args(call.arguments)
            self._close_tool_line()
            print(f"{DIM}→ {call.name}({args_preview}){RESET}", flush=True)
            self._tool_line_open = True
        elif isinstance(event, ToolExecutionEndEvent):
            status = "✗" if event.result.is_error else "✓"
            first_line = event.result.output.strip().splitlines()
            preview = first_line[0][:100] if first_line else ""
            if self._tool_line_open:
                print(f"{DIM}  {status} {preview}{RESET}", flush=True)
                self._tool_line_open = False
        elif isinstance(event, ErrorEvent):
            print(f"{YELLOW}{event.message}{RESET}", file=sys.stderr, flush=True)
        elif isinstance(event, AgentEndEvent):
            if event.aborted:
                print(f"{YELLOW}[aborted]{RESET}", flush=True)

    def _close_tool_line(self) -> None:
        self._tool_line_open = False

    # -- main loop ----------------------------------------------------------------

    def run(self) -> None:
        print(
            f"{BOLD}tau{RESET} {DIM}v{__version__} · {self.session.provider.model} · "
            f"{self.session.session_manager.file}{RESET}"
        )
        while True:
            try:
                line = input(f"{CYAN}» {RESET}")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(f"{DIM}(use /quit to exit){RESET}")
                continue

            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    break
                continue

            self._submit(line)
            self._print_usage()

    def _submit(self, text: str) -> None:
        try:
            self.session.prompt(text)
        except KeyboardInterrupt:
            pass  # abort path emits [aborted]
        finally:
            self._spinner.stop()  # e.g. Ctrl-C surfaced outside event handling
        print()  # blank line after each run

    def _print_usage(self) -> None:
        usage = self.session.usage
        if usage.total_tokens:
            print(
                f"{DIM}⟨ {usage.input:,} in · {usage.output:,} out · "
                f"{usage.total_tokens:,} total ⟩{RESET}"
            )

    # -- slash commands -------------------------------------------------------------

    def _command(self, line: str) -> bool:
        """Handle a slash command. Returns True to exit."""
        parts = line.split(maxsplit=1)
        command, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

        if command in ("/quit", "/exit"):
            return True
        if command == "/help":
            print(HELP, end="")
        elif command == "/model":
            if arg:
                self.session.set_model(arg)
            print(f"model: {self.session.provider.model}")
        elif command == "/tools":
            for tool in self.session.tools.values():
                first = tool.description.split(". ")[0]
                print(f"  {BOLD}{tool.name}{RESET}: {first}")
        elif command == "/system":
            print(self.session.system_prompt)
        elif command == "/new":
            self.session.session_manager.new_session()
            print(f"{DIM}new session: {self.session.session_manager.file}{RESET}")
        elif command == "/sessions":
            for i, path in enumerate(SessionManager.list_sessions(), 1):
                marker = "→" if path == self.session.session_manager.file else " "
                print(f"{marker} {i}. {path.stem}  {DIM}{path.parent.name}{RESET}")
        elif command == "/resume":
            if not arg.isdigit():
                print("usage: /resume <n>")
                return False
            sessions = SessionManager.list_sessions()
            index = int(arg) - 1
            if not 0 <= index < len(sessions):
                print("no such session")
                return False
            self.session.session_manager.load(sessions[index])
            print(f"{DIM}resumed {sessions[index]}{RESET}")
        elif command == "/usage":
            self._print_usage()
        else:
            print(f"unknown command: {command} (try /help)")
        return     False


def _format_args(arguments: dict) -> str:
    pieces = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "…"
        text = text.replace("\n", "\\n")
        pieces.append(f"{key}={text!r}")
    return ", ".join(pieces)


def print_mode(args: argparse.Namespace) -> None:
    """Like `pi -p`: run one prompt, write the final response to stdout."""
    session = make_session(args)
    unsubscribe = session.subscribe(
        lambda event: (
            print(event.delta, end="", flush=True)
            if isinstance(event, TextDeltaEvent)
            else None
        )
    )
    try:
        session.prompt(args.prompt)
    finally:
        unsubscribe()
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tau", description="A minimal coding agent, modeled after pi")
    parser.add_argument("prompt", nargs="?", help="starting prompt (interactive mode)")
    parser.add_argument("-p", "--print", action="store_true", help="print mode: run once, write final response")
    parser.add_argument("-m", "--model", help="model to use (default: $TAU_MODEL or gpt-4.1)")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--resume", type=int, metavar="N", help="resume the n-th recent session")
    parser.add_argument("--version", action="version", version=f"tau {__version__}")
    args = parser.parse_args(argv)

    try:
        import openai  # noqa: F401
    except ImportError:
        sys.exit("tau requires the openai package: pip install openai")

    try:
        if args.print:
            if not args.prompt:
                sys.exit("-p requires a prompt")
            print_mode(args)
            return

        session = make_session(args)
    except Exception as exc:
        if type(exc).__name__ == "OpenAIError":
            sys.exit(
                f"Missing credentials. Set OPENAI_API_KEY (or TAU_API_KEY),\n"
                f"or pass --api-key / --base-url for an OpenAI-compatible endpoint."
            )
        raise

    repl = Repl(session)
    if args.prompt:
        repl._submit(args.prompt)
        repl._print_usage()
    repl.run()


if __name__ == "__main__":
    main()
