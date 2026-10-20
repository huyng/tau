"""Model runtime: streams assistant responses from an OpenAI-compatible API.

The provider converts agent messages and tool definitions into a
chat-completions request, streams the response as events, and returns the
completed AssistantMessage as the generator's return value. The agent loop
consumes it like this:

    stream = provider.stream(system_prompt, messages, tools)
    for event in stream:
        ...  # TextDeltaEvent / ToolNameEvent
    # stream.return_value holds the completed AssistantMessage
"""

from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .messages import AgentMessage
    from .tools import Tool

from .events import ReasoningDeltaEvent, TextDeltaEvent, ToolNameEvent
from .messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
    ToolResultMessage,
)


class OpenAIProvider:
    """Streaming provider for any OpenAI-compatible endpoint.

    Configuration precedence: explicit arguments, then TAU_* environment
    variables, then OPENAI_* environment variables, then defaults.
    """

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, client=None):
        self.model = model or os.environ.get("TAU_MODEL", "gpt-4.1")
        base_url = base_url or os.environ.get("TAU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("TAU_API_KEY")
        if base_url and "deepseek" in base_url:
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI  # lazy so tests/SDK callers can inject a client

            self.client = OpenAI(base_url=base_url, api_key=api_key)

        # DeepSeek thinking mode requires reasoning_content to be echoed back
        # when tools are present. Other providers (e.g. z.ai) should keep the
        # previous behavior unless explicitly opted in.
        self.pass_reasoning_content = bool(base_url and "deepseek" in base_url)
        override = os.environ.get("TAU_PASS_REASONING_CONTENT")
        if override is not None:
            self.pass_reasoning_content = override.strip().lower() not in (
                "0", "false", "no", "off",
            )

    # -- request construction ------------------------------------------------

    def _request_messages(self, system_prompt: str, messages: list[AgentMessage]) -> list[dict]:
        request: list[dict] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if isinstance(message, UserMessage):
                request.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
                entry: dict = {"role": "assistant"}
                if message.text:
                    entry["content"] = message.text
                if message.reasoning_content and self.pass_reasoning_content:
                    entry["reasoning_content"] = message.reasoning_content
                if calls:
                    entry["tool_calls"] = calls
                request.append(entry)
            elif isinstance(message, ToolResultMessage):
                request.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.output,
                    }
                )
        return request

    @staticmethod
    def _tool_schemas(tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    # -- streaming ------------------------------------------------------------

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Iterator[TextDeltaEvent | ToolNameEvent]:
        """Yield streaming events; the completed AssistantMessage is the
        generator's return value (StopIteration.value)."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._request_messages(system_prompt, messages),
            tools=self._tool_schemas(tools) if tools else None,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = Usage()
        stop_reason = "stop"
        failure_reason: str | None = None
        aborted = False

        try:
            for chunk in stream:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    stream.close()
                    break
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        input=chunk.usage.prompt_tokens or 0,
                        output=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    text_parts.append(delta.content)
                    yield TextDeltaEvent(delta=delta.content)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ReasoningDeltaEvent(delta=reasoning)
                for tc in delta.tool_calls or []:
                    acc = tool_calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["name"] += tc.function.name
                            yield ToolNameEvent(name=acc["name"])
                        if tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    stop_reason = {
                        "stop": "stop",
                        "tool_calls": "toolUse",
                        "length": "length",
                        "aborted": "aborted",
                        # DeepSeek-specific failure reasons
                        "content_filter": "error",
                        "insufficient_system_resource": "error",
                    }.get(choice.finish_reason, "stop")
                    if stop_reason == "error":
                        failure_reason = choice.finish_reason
        except KeyboardInterrupt:
            # Ctrl-C surfaced inside the stream: finalize as aborted.
            aborted = True

        content: list = [TextContent(text="".join(text_parts))]
        for index in sorted(tool_calls):
            acc = tool_calls[index]
            try:
                arguments = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": acc["arguments"]}
            content.append(
                ToolCall(id=acc["id"] or new_call_id(), name=acc["name"], arguments=arguments)
            )

        return AssistantMessage(
            content=content,
            model=self.model,
            provider=self.name,
            stop_reason="aborted" if aborted else stop_reason,
            usage=usage,
            error_message=failure_reason,
            reasoning_content="".join(reasoning_parts) or None,
        )


def new_call_id() -> str:
    import uuid

    return f"call_{uuid.uuid4().hex[:24]}"
