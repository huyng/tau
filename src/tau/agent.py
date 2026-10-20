"""The agent loop, modeled after pi.

A submitted message is added to the active branch. The agent builds a model
request from the system prompt, active branch, and available tools, then
sends it through the provider. The provider streams an assistant response,
which can contain text and tool calls. The agent records the response,
executes each tool call, and records the results. That completes one turn.
If tool results require another model request, the agent starts another
turn; otherwise the run ends.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolNameEvent,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .provider import OpenAIProvider
from .session import SessionManager
from .tools import Tool, ToolResult, validate_arguments

MAX_RETRIES = 3
RETRY_DELAYS = (1, 2, 4)


class AgentSession:
    def __init__(
        self,
        provider: OpenAIProvider,
        tools: list[Tool],
        session_manager: SessionManager,
        system_prompt: str,
        cwd: str | None = None,
    ):
        self.provider = provider
        self.tools = {tool.name: tool for tool in tools}
        self.session_manager = session_manager
        self.system_prompt = system_prompt
        self.cwd = cwd
        self._subscribers: list[Callable] = []
        self._abort_event = threading.Event()
        self._busy = False
        self.usage = Usage()  # cumulative totals across runs

        if session_manager.file is None:
            session_manager.new_session()

    # -- event subscription ----------------------------------------------------

    def subscribe(self, callback: Callable) -> Callable[[], None]:
        """Subscribe to session events. Returns an unsubscribe function."""
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _emit(self, event) -> None:
        for callback in list(self._subscribers):
            callback(event)

    # -- state -----------------------------------------------------------------

    @property
    def messages(self) -> list[AgentMessage]:
        """The active branch: conversation history for the next request."""
        return self.session_manager.active_branch()

    @property
    def busy(self) -> bool:
        return self._busy

    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, AssistantMessage):
                return message.text
        return ""

    def set_model(self, model: str) -> None:
        self.provider.model = model

    # -- prompting -------------------------------------------------------------

    def prompt(self, text: str) -> AssistantMessage:
        """Run the agent loop for a user message. Resolves when the run ends."""
        self._abort_event.clear()
        self._busy = True
        self._emit(AgentStartEvent())
        self.session_manager.append(UserMessage(content=text))

        assistant: AssistantMessage | None = None
        aborted = False
        try:
            while True:
                assistant = self._run_turn()
                if assistant is None or self._abort_event.is_set():
                    aborted = True
                    break
                if assistant.stop_reason == "toolUse" and assistant.tool_calls:
                    if not self._execute_tool_calls(assistant):
                        aborted = True  # aborted between tool calls
                        break
                    continue  # tool results require another model request
                break
        except KeyboardInterrupt:
            aborted = True
        finally:
            self._busy = False
            self._emit(AgentEndEvent(aborted=aborted))

        if assistant is None:
            assistant = AssistantMessage(
                content=[], model=self.provider.model, provider=self.provider.name,
                stop_reason="aborted",
            )
            self.session_manager.append(assistant)
        return assistant

    def abort(self) -> None:
        """Stop the active run after the current provider chunk / tool call."""
        self._abort_event.set()

    def dispose(self) -> None:
        self.abort()

    # -- turns -------------------------------------------------------------------

    def _run_turn(self) -> AssistantMessage | None:
        """One model request: stream a response, record it, return it."""
        self._emit(MessageStartEvent())
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.provider.stream(
                    self.system_prompt,
                    self.messages,
                    list(self.tools.values()),
                    abort_event=self._abort_event,
                )
                while True:
                    try:
                        event = next(stream)
                    except StopIteration as stop:
                        assistant: AssistantMessage = stop.value
                        break
                    self._emit(event)
                self.usage.add(assistant.usage)
                if assistant.stop_reason == "error":
                    # Retry without recording the failed attempt; the final
                    # error message is appended once all retries are exhausted.
                    raise RuntimeError(assistant.error_message or "provider error")
                self.session_manager.append(assistant)
                self._emit(MessageEndEvent(message=assistant))
                return assistant
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # transient provider failures
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    self._emit(ErrorEvent(message=f"{exc}; retrying in {RETRY_DELAYS[attempt]}s"))
                    time.sleep(RETRY_DELAYS[attempt])
        error = AssistantMessage(
            content=[],
            model=self.provider.model,
            provider=self.provider.name,
            stop_reason="error",
            error_message=str(last_error),
        )
        self._emit(ErrorEvent(message=str(last_error)))
        self.session_manager.append(error)
        self._emit(MessageEndEvent(message=error))
        return error

    def _execute_tool_calls(self, assistant: AssistantMessage) -> bool:
        """Execute each tool call and record the results. False if aborted."""
        for call in assistant.tool_calls:
            if self._abort_event.is_set():
                return False
            self._emit(ToolExecutionStartEvent(tool_call=call))
            result = self._execute_one(call)
            message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                output=result.output,
                is_error=result.is_error,
            )
            self.session_manager.append(message)
            self._emit(ToolExecutionEndEvent(tool_call=call, result=message))
        return True

    def _execute_one(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(output=f"Error: unknown tool: {call.name}", is_error=True)
        if not isinstance(call.arguments, dict):
            return ToolResult(
                output=(
                    f"Error: bad arguments for {call.name}: expected a JSON object, "
                    f"got {type(call.arguments).__name__}"
                ),
                is_error=True,
            )
        problems = validate_arguments(tool.parameters, call.arguments)
        if problems:
            return ToolResult(
                output=(
                    f"Error: invalid arguments for {call.name}: "
                    + "; ".join(problems)
                    + ". Fix the arguments and call the tool again."
                ),
                is_error=True,
            )
        try:
            return tool.execute(**call.arguments)
        except TypeError as exc:
            return ToolResult(output=f"Error: bad arguments for {call.name}: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"Error executing {call.name}: {exc}", is_error=True)
