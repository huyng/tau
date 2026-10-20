"""Provider tests, focused on DeepSeek thinking-mode compatibility."""

from types import SimpleNamespace

from tau.events import ReasoningDeltaEvent, TextDeltaEvent
from tau.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    deserialize_message,
    serialize_message,
)
from tau.provider import OpenAIProvider


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.chunks)


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=FakeCompletions(chunks))


def chunk(*, content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(choices=choices, usage=usage)


def collect(provider, messages):
    """Drive provider.stream to completion, returning (events, message)."""
    stream = provider.stream("system", messages, [])
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as stop:
            return events, stop.value


def make_provider(chunks) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(chunks)
    provider = OpenAIProvider(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        client=client,
    )
    return provider, client


def test_reasoning_is_streamed_stored_and_sent_back():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks = [
        chunk(reasoning="Let me "),
        chunk(reasoning="think."),
        chunk(content="Hello"),
        chunk(content=" there"),
        chunk(finish_reason="stop", usage=usage),
    ]
    provider, _ = make_provider(chunks)

    events, message = collect(provider, [UserMessage(content="hi")])

    assert "".join(e.delta for e in events if isinstance(e, ReasoningDeltaEvent)) == "Let me think."
    assert "".join(e.delta for e in events if isinstance(e, TextDeltaEvent)) == "Hello there"
    assert message.reasoning_content == "Let me think."
    assert message.text == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.total_tokens == 15

    # DeepSeek requires reasoning_content on prior assistant turns when tools
    # are present. Verify it survives serialization and request construction.
    data = serialize_message(message)
    assert data["reasoningContent"] == "Let me think."
    assert deserialize_message(data).reasoning_content == "Let me think."

    request = provider._request_messages(
        "system",
        [
            UserMessage(content="hi"),
            message,
            ToolResultMessage(tool_call_id="c1", tool_name="read", output="ok"),
        ],
    )
    assert request[2]["reasoning_content"] == "Let me think."


def test_reasoning_is_not_sent_to_non_deepseek_providers():
    provider = OpenAIProvider(
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key="test-key",
        client=FakeClient([]),
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        provider="openai",
        reasoning_content="private reasoning",
    )

    request = provider._request_messages("system", [message])

    assert "reasoning_content" not in request[1]


def test_reasoning_passthrough_can_be_forced_with_env(monkeypatch):
    monkeypatch.setenv("TAU_PASS_REASONING_CONTENT", "1")
    provider = OpenAIProvider(
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key="test-key",
        client=FakeClient([]),
    )
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        model="glm-4.6",
        provider="openai",
        reasoning_content="private reasoning",
    )

    request = provider._request_messages("system", [message])

    assert request[1]["reasoning_content"] == "private reasoning"


def test_reasoning_is_sent_back_alongside_tool_calls():
    provider, _ = make_provider([])
    message = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
        ],
        model="deepseek-flash",
        provider="openai",
        stop_reason="toolUse",
        reasoning_content="I should read the file.",
    )

    request = provider._request_messages("system", [message])
    assistant = request[1]

    assert assistant["reasoning_content"] == "I should read the file."
    assert assistant["tool_calls"][0]["function"]["name"] == "read"


def test_deepseek_failure_finish_reasons_map_to_error():
    provider, _ = make_provider(
        [chunk(finish_reason="insufficient_system_resource")]
    )

    _, message = collect(provider, [UserMessage(content="hi")])

    assert message.stop_reason == "error"
    assert message.error_message == "insufficient_system_resource"
