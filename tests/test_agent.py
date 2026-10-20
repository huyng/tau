"""Tests for the tau agent harness using a scripted fake provider."""

import json
from typing import Any, cast

import pytest

from tau.agent import AgentSession
from tau.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageEndEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tau.messages import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from tau.session import SessionManager
from tau.tools import Tool, ToolResult


class FakeProvider:
    """Scripts a sequence of turns: each turn is an AssistantMessage."""

    name = "fake"

    def __init__(self, turns: list[AssistantMessage]):
        self.turns = list(turns)
        self.model = "fake-model"
        self.requests: list[tuple[str, list, list]] = []

    def stream(self, system_prompt, messages, tools, abort_event=None):
        self.requests.append((system_prompt, list(messages), list(tools)))
        turn = self.turns.pop(0)

        def generate():
            for block in turn.content:
                if isinstance(block, TextContent):
                    for word in block.text.split(" "):
                        yield TextDeltaEvent(delta=word + " ")
            return turn

        return generate()


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the input."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    def execute(self, text: str, **_: Any) -> ToolResult:
        self.calls.append({"text": text})
        return ToolResult(output=f"echo: {text}")


def make_session(tmp_path, turns) -> tuple[AgentSession, EchoTool, SessionManager]:
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    tool = EchoTool()
    session = AgentSession(
        provider=FakeProvider(turns),
        tools=[tool],
        session_manager=manager,
        system_prompt="test system prompt",
        cwd=str(tmp_path),
    )
    return session, tool, manager


def test_agent_loop_tool_call_and_final_text(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
        model="fake-model",
        provider="fake",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="all done")],
        model="fake-model",
        provider="fake",
        stop_reason="stop",
    )
    session, tool, manager = make_session(tmp_path, [turn1, turn2])

    events = []
    session.subscribe(events.append)
    final = session.prompt("run the echo tool")

    assert final.text == "all done"
    assert tool.calls == [{"text": "hello"}]

    kinds = [type(e) for e in events]
    assert kinds[0] is AgentStartEvent
    assert kinds[-1] is AgentEndEvent
    assert kinds.count(MessageEndEvent) == 2
    assert kinds.count(ToolExecutionStartEvent) == 1
    assert kinds.count(ToolExecutionEndEvent) == 1
    assert any(isinstance(e, TextDeltaEvent) and "done" in e.delta for e in events)

    # active branch: user, assistant(toolCall), toolResult, assistant(text)
    messages = session.messages
    assert [type(m) for m in messages] == [
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        AssistantMessage,
    ]
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].output == "echo: hello"
    assert session.last_assistant_text() == "all done"

    # the second request included the tool result
    provider = cast(FakeProvider, session.provider)
    _, second_request_messages, _ = provider.requests[1]
    assert isinstance(second_request_messages[-1], ToolResultMessage)
    assert second_request_messages[-1].output == "echo: hello"


def test_session_jsonl_persistence(tmp_path):
    turn = AssistantMessage(
        content=[TextContent(text="hi there")],
        model="fake-model",
        provider="fake",
        stop_reason="stop",
    )
    session, _, manager = make_session(tmp_path, [turn])
    session.prompt("say hi")

    session_file = manager.file
    assert session_file is not None
    lines = [json.loads(l) for l in session_file.read_text().splitlines() if l.strip()]
    entries = [l for l in lines if "id" in l]
    pointers = [l for l in lines if "current" in l]
    assert len(entries) == 2  # user + assistant
    assert entries[0]["parent"] is None
    assert entries[1]["parent"] == entries[0]["id"]
    assert pointers[-1]["current"] == entries[-1]["id"]
    assert entries[0]["message"]["role"] == "user"
    assert entries[1]["message"]["role"] == "assistant"

    # reload and verify the active branch is restored
    restored = SessionManager(directory=tmp_path)
    restored.load(session_file)
    assert [type(m) for m in restored.active_branch()] == [UserMessage, AssistantMessage]


def test_unknown_tool_records_error_result(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="nonexistent", arguments={})],
        model="fake-model",
        provider="fake",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="recovered")],
        model="fake-model",
        provider="fake",
        stop_reason="stop",
    )
    session, _, _ = make_session(tmp_path, [turn1, turn2])
    final = session.prompt("call a missing tool")

    assert final.text == "recovered"
    tool_result = [m for m in session.messages if isinstance(m, ToolResultMessage)][0]
    assert tool_result.is_error
    assert "unknown tool" in tool_result.output


def test_invalid_argument_type_is_rejected_before_execution(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": 42})],
        model="fake-model",
        provider="fake",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="retried")],
        model="fake-model",
        provider="fake",
        stop_reason="stop",
    )
    session, tool, _ = make_session(tmp_path, [turn1, turn2])
    session.prompt("call with a bad argument type")

    assert tool.calls == []  # never reached the tool
    result = [m for m in session.messages if isinstance(m, ToolResultMessage)][0]
    assert result.is_error
    assert "invalid arguments for echo" in result.output
    assert "arguments.text must be a string (got integer)" in result.output


def test_missing_required_argument_is_rejected(tmp_path):
    turn1 = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={})],
        model="fake-model",
        provider="fake",
        stop_reason="toolUse",
    )
    turn2 = AssistantMessage(
        content=[TextContent(text="retried")],
        model="fake-model",
        provider="fake",
        stop_reason="stop",
    )
    session, tool, _ = make_session(tmp_path, [turn1, turn2])
    session.prompt("call without the required argument")

    assert tool.calls == []
    result = [m for m in session.messages if isinstance(m, ToolResultMessage)][0]
    assert result.is_error
    assert "arguments.text is required" in result.output


def test_abort_prevents_tool_execution(tmp_path):
    turn = AssistantMessage(
        content=[ToolCall(id="c1", name="echo", arguments={"text": "nope"})],
        model="fake-model",
        provider="fake",
        stop_reason="toolUse",
    )
    session, tool, _ = make_session(tmp_path, [turn])

    def aborter(event):
        if isinstance(event, MessageEndEvent):
            session.abort()

    session.subscribe(aborter)
    final = session.prompt("abort before tools run")

    assert final.stop_reason == "toolUse"  # message completed
    assert tool.calls == []  # but tools never ran
    last = session.messages[-1]
    assert isinstance(last, AssistantMessage)  # no tool result was appended


def test_abort_during_retry_stops_without_another_request(tmp_path):
    class FailingProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self.calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("boom")

    provider = FailingProvider()
    session = AgentSession(
        provider=provider,
        tools=[],
        session_manager=SessionManager(directory=tmp_path),
        system_prompt="test system prompt",
    )

    def abort_on_error(event):
        if isinstance(event, ErrorEvent):
            session.abort()

    session.subscribe(abort_on_error)
    final = session.prompt("hi")

    assert final.stop_reason == "aborted"
    assert provider.calls == 1


def test_repl_renders_text_and_reasoning_deltas(capsys):
    """Regression: the interactive REPL must print streamed text (it used to
    only render tool events, so replies were invisible)."""
    from tau.cli import Repl
    from tau.events import MessageEndEvent, MessageStartEvent, ReasoningDeltaEvent

    class StubSession:
        provider = type("P", (), {"model": "m"})()
        session_manager = type("S", (), {"file": "x"})

        def subscribe(self, cb):
            return lambda: None

    repl = Repl(cast(AgentSession, StubSession()))
    repl._on_event(MessageStartEvent())
    repl._on_event(ReasoningDeltaEvent(delta="thinking... "))
    repl._on_event(TextDeltaEvent(delta="Hello"))
    repl._on_event(TextDeltaEvent(delta=" there"))
    message = AssistantMessage(
        content=[TextContent(text="Hello there")], model="m", provider="fake"
    )
    repl._on_event(MessageEndEvent(message=message))

    out = capsys.readouterr().out
    assert "thinking... " in out
    assert "Hello there" in out
    assert out.endswith("\n")
