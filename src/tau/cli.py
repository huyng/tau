"""Interactive terminal interface, modeled after pi's REPL.

Modes:
- interactive (default): stream responses, use slash commands
- print: `tau -p "prompt"` runs once and writes the final response
"""

import argparse
import os
import sys
import time

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
from .render import (
    GutterWriter,
    MarkdownRenderer,
    format_duration,
    render_tool_result,
    summarize_tool_call,
)
from .session import SessionManager
from .spinner import Spinner
from .theme import DIM, Theme, get_theme, terminal_width
from .tools import default_tools

HELP = """\
/help              Show this help
/model [name]      Show or switch the model
/tools             List active tools
/system            Show the current system prompt
/plain             Toggle Markdown rendering
/thinking          Toggle streamed model reasoning
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
    def __init__(self, session: AgentSession, theme: Theme | None = None):
        self.session = session
        self.theme = theme if theme is not None else get_theme()
        self.width = terminal_width()
        self.markdown = True
        self.show_thinking = True

        self.session.subscribe(self._on_event)
        self._tool_line_open = False
        self._text_started = False
        self._text_ends_newline = False
        self._reasoning_started = False
        self._renderer = MarkdownRenderer(self.theme, self.width)
        self._reasoning = GutterWriter(self.theme, style=DIM)
        self._tool_started: dict[str, float] = {}
        self._spinner = Spinner()

    # -- event rendering ---------------------------------------------------------

    def _on_event(self, event) -> None:
        if isinstance(event, MessageStartEvent):
            self._text_started = False
            self._text_ends_newline = False
            self._reasoning_started = False
            self._renderer = MarkdownRenderer(self.theme, self.width)
            self._reasoning = GutterWriter(self.theme, style=DIM)
            self._spinner.start(f"{self.session.provider.model} · thinking")
            return
        if isinstance(event, (AgentStartEvent, ToolNameEvent)):
            return

        # Every remaining event can produce output: retire the spinner first.
        self._spinner.stop()

        if isinstance(event, ReasoningDeltaEvent):
            if self.show_thinking:
                print(self._reasoning.feed(event.delta), end="", flush=True)
                self._reasoning_started = True
        elif isinstance(event, TextDeltaEvent):
            self._finish_reasoning()
            self._text_started = True
            self._text_ends_newline = event.delta.endswith("\n")
            if self.markdown:
                print(self._renderer.feed(event.delta), end="", flush=True)
            else:
                print(event.delta, end="", flush=True)
        elif isinstance(event, MessageEndEvent):
            self._finish_reasoning()
            self._finish_text(event.message.text)
        elif isinstance(event, ToolExecutionStartEvent):
            self._finish_reasoning()
            self._finish_text()
            call = event.tool_call
            summary = summarize_tool_call(
                self.theme, call.name, call.arguments, self.width
            )
            self._close_tool_line()
            head = self.theme.dim("╭─ ") + self.theme.bold(call.name)
            if summary:
                head += " " + summary
            print(head, flush=True)
            self._tool_line_open = True
            self._tool_started[call.id] = time.monotonic()
            self._spinner.start(f"{call.name} …")
        elif isinstance(event, ToolExecutionEndEvent):
            started = self._tool_started.pop(event.tool_call.id, None)
            elapsed = time.monotonic() - started if started is not None else 0.0
            print(
                render_tool_result(
                    self.theme,
                    event.tool_call.name,
                    event.result.output,
                    event.result.is_error,
                    elapsed,
                    self.width,
                ),
                flush=True,
            )
            self._tool_line_open = False
        elif isinstance(event, ErrorEvent):
            print(self.theme.warning(event.message), file=sys.stderr, flush=True)
        elif isinstance(event, AgentEndEvent):
            self._finish_reasoning()
            self._finish_text()
            if event.aborted:
                print(self.theme.warning("[aborted]"), flush=True)

    def _finish_reasoning(self) -> None:
        if self._reasoning_started:
            print(self._reasoning.flush(), end="", flush=True)
            self._reasoning_started = False

    def _finish_text(self, text: str | None = None) -> None:
        if not self._text_started:
            return
        if self.markdown:
            print(self._renderer.flush(), end="", flush=True)
        else:
            ended = text.endswith("\n") if text is not None else self._text_ends_newline
            if not ended:
                print(flush=True)
        self._text_started = False

    def _close_tool_line(self) -> None:
        self._tool_line_open = False

    # -- main loop ----------------------------------------------------------------

    def run(self) -> None:
        theme = self.theme
        print(
            f"{theme.bold('tau')} {theme.dim('v' + __version__)} {theme.dim('·')} "
            f"{theme.model(self.session.provider.model)} {theme.dim('·')} "
            f"{theme.dim(str(self.session.session_manager.file))}"
        )
        while True:
            try:
                line = input(theme.accent("» "))
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(theme.dim("(use /quit to exit)"))
                continue

            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    break
                continue

            self._submit(line)

    def _submit(self, text: str) -> None:
        started = time.monotonic()
        try:
            self.session.prompt(text)
        except KeyboardInterrupt:
            pass  # abort path emits [aborted]
        finally:
            self._spinner.stop()  # e.g. Ctrl-C surfaced outside event handling
        print()  # blank line after each run
        self._print_usage(time.monotonic() - started)

    def _print_usage(self, elapsed: float | None = None) -> None:
        usage = self.session.usage
        if usage.total_tokens:
            parts = [self.session.provider.model]
            if elapsed is not None:
                parts.append(format_duration(elapsed))
            parts += [
                f"{usage.input:,} in",
                f"{usage.output:,} out",
            ]
            print(
                self.theme.dim("⟨ " + " · ".join(parts) + " ⟩"),
                flush=True,
            )

    # -- slash commands -------------------------------------------------------------

    def _command(self, line: str) -> bool:
        """Handle a slash command. Returns True to exit."""
        parts = line.split(maxsplit=1)
        command, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        theme = self.theme

        if command in ("/quit", "/exit"):
            return True
        if command == "/help":
            print(HELP, end="")
        elif command == "/model":
            if arg:
                self.session.set_model(arg)
            print(f"{theme.dim('model:')} {theme.model(self.session.provider.model)}")
        elif command == "/tools":
            for tool in self.session.tools.values():
                first = tool.description.split(". ")[0]
                print(f"  {theme.bold(tool.name)}{theme.dim(': ' + first)}")
        elif command == "/system":
            print(self.session.system_prompt)
        elif command == "/plain":
            self.markdown = not self.markdown
            state = "on" if self.markdown else "off"
            print(theme.dim(f"markdown rendering: {state}"))
        elif command == "/thinking":
            self.show_thinking = not self.show_thinking
            state = "shown" if self.show_thinking else "hidden"
            print(theme.dim(f"reasoning: {state}"))
        elif command == "/new":
            self.session.session_manager.new_session()
            print(theme.dim(f"new session: {self.session.session_manager.file}"))
        elif command == "/sessions":
            for i, path in enumerate(SessionManager.list_sessions(), 1):
                marker = "→" if path == self.session.session_manager.file else " "
                marker = theme.accent(marker)
                print(f"{marker} {i}. {path.stem}  {theme.dim(path.parent.name)}")
        elif command == "/resume":
            if not arg.isdigit():
                print("usage: /resume <n>")
                return False
            sessions = SessionManager.list_sessions()
            index = int(arg) - 1
            if not 0 <= index < len(sessions):
                print(theme.error("no such session"))
                return False
            self.session.session_manager.load(sessions[index])
            print(theme.dim(f"resumed {sessions[index]}"))
        elif command == "/usage":
            self._print_usage()
        else:
            print(
                theme.error(f"unknown command: {command}")
                + theme.dim(" (try /help)")
            )
        return False


def print_mode(args: argparse.Namespace) -> None:
    """Like `pi -p`: run one prompt, write the final response to stdout.

    Deliberately plain: no styling, no tool chatter, no spinners.
    """
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
    repl.run()


if __name__ == "__main__":
    main()
