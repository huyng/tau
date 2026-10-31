"""Model runtime: streams assistant responses from an OpenAI-compatible API.

The provider converts agent messages and tool definitions into a
chat-completions request, streams the response as events, and returns the
completed AssistantMessage as the generator's return value. The agent loop
consumes it like this:

    stream = provider.stream(system_prompt, messages, tools)
    for event in stream:
        ...  # TextDeltaEvent / ToolNameEvent
    # The completed AssistantMessage is the generator's return value
    # (StopIteration.value).

Transport is plain ``urllib`` + ``http.client``; the wire format is SSE, which
is a line protocol, so one ``for line in response`` covers it. Chunk dicts are
passed around as plain mappings rather than typed models — nothing here wants
attribute access, and ``stream_chunk_transport`` can be swapped out wholesale
(tests do exactly that).
"""

import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Generator, Iterator, Mapping
from contextlib import closing
from typing import Any, Protocol

from .events import ProviderEvent, ReasoningDeltaEvent, TextDeltaEvent, ToolNameEvent
from .messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .tools import Tool

#: A parsed `data:` payload from the event stream.
Chunk = Mapping[str, Any]

#: `(url, headers, body) -> iterator of decoded chunks`, in request order.
class Transport(Protocol):
    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float = ...,
    ) -> Iterator[Chunk]: ...

DEFAULT_TIMEOUT = 600.0

MISSING_CREDENTIALS = (
    "Missing credentials. Set OPENAI_API_KEY (or TAU_API_KEY),\n"
    "or pass --api-key / --base-url for an OpenAI-compatible endpoint."
)


class ProviderError(RuntimeError):
    """A failure talking to the endpoint.

    ``retryable`` distinguishes "try again in a moment" (rate limits, 5xx,
    dropped connections) from "this will never succeed" (bad key, bad request,
    content filtered). The agent loop re-raises the latter immediately instead
    of burning its retry budget and the user's time on it.
    """

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class MissingCredentialsError(ProviderError):
    """No API key was configured for the endpoint."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


#: Status codes worth retrying: rate limits and server-side failures.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})


class Provider(Protocol):
    """What the agent needs from a model runtime; OpenAIProvider implements it.

    Tests and other hosts can satisfy it with a stand-in instead of a real
    transport. ``stream`` yields events and returns the completed message via
    StopIteration.value.
    """

    name: str
    model: str

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]: ...


def stream_chunk_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[Chunk]:
    """POST `body` and yield each SSE event's JSON payload.

    Lines that are not `data:` (blank separators, `:` heartbeat comments,
    `event:` fields) are skipped, as is the `[DONE]` terminator. Multiple
    `data:` lines would form one event per the SSE spec, but OpenAI-style APIs
    send exactly one JSON object per event, so each line is decoded alone.

    The response object is closed when the caller stops iterating, so abandoning
    the stream (Ctrl-C, abort) drops the connection instead of draining it.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ProviderError(
            _describe_http_error(exc), retryable=_is_retryable_status(exc.code)
        ) from exc
    except urllib.error.URLError as exc:
        # DNS failure, refused connection, TLS problem, read timeout.
        raise ProviderError(f"cannot reach {url}: {exc.reason}") from exc

    try:
        with closing(response):
            for raw in response:  # line-buffered by http.client; one SSE line each
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:  # never lose a turn to a stray line
                    raise ProviderError(
                        f"malformed stream event: {payload[:200]}"
                    ) from exc
                if isinstance(chunk, dict):
                    yield chunk
    except (http.client.HTTPException, OSError) as exc:
        # A connection dropped or timed out mid-stream: the turn is incomplete,
        # and a fresh request is the remedy. ProviderErrors raised above pass
        # through untouched.
        raise ProviderError(f"stream interrupted: {exc!r}") from exc


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """Fold an HTTP error into one readable line, preferring the API's message."""
    detail = ""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # the body is a courtesy; never mask the status
        body = ""
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            elif parsed.get("message"):
                detail = str(parsed["message"])
        if not detail:
            detail = " ".join(body.split())[:300]
    return f"HTTP {exc.code} {exc.reason}" + (f": {detail}" if detail else "")


def _is_retryable_status(code: int | None) -> bool:
    """429 and 5xx are transient; other 4xx are the caller's fault."""
    if code is None:
        return True
    return code in RETRYABLE_STATUSES or 500 <= code <= 599


class OpenAIProvider:
    """Streaming provider for any OpenAI-compatible endpoint.

    Configuration precedence: explicit arguments, then TAU_* environment
    variables, then OPENAI_* environment variables, then defaults.
    """

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, transport: Transport | None = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.model = model or os.environ.get("TAU_MODEL", "gpt-4.1")
        self.base_url = (
            base_url or os.environ.get("TAU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        api_key = api_key or os.environ.get("TAU_API_KEY")
        if "deepseek" in self.base_url:
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.transport = transport or stream_chunk_transport
        self.timeout = timeout
        if not self.api_key:
            # Raised here, at construction, so callers fail before any work.
            raise MissingCredentialsError(MISSING_CREDENTIALS)

        # DeepSeek thinking mode requires reasoning_content to be echoed back
        # when tools are present. Other providers (e.g. z.ai) should keep the
        # previous behavior unless explicitly opted in.
        self.pass_reasoning_content = "deepseek" in self.base_url
        override = os.environ.get("TAU_PASS_REASONING_CONTENT")
        if override is not None:
            self.pass_reasoning_content = override.strip().lower() not in (
                "0", "false", "no", "off",
            )

    # -- request construction ------------------------------------------------

    def _request_messages(
        self, system_prompt: str, messages: list[AgentMessage]
    ) -> list[dict[str, Any]]:
        request: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
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
                entry: dict[str, Any] = {"role": "assistant"}
                if message.text:
                    entry["content"] = message.text
                if message.reasoning_content and self.pass_reasoning_content:
                    # DeepSeek-only field: outside the OpenAI request schema.
                    entry["reasoning_content"] = message.reasoning_content
                if calls:
                    entry["tool_calls"] = calls
                if not entry.get("content") and "tool_calls" not in entry:
                    # An aborted or failed turn leaves an assistant message with
                    # nothing in it. Providers reject that shape outright, and it
                    # carries nothing a later request needs.
                    continue
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
    def _tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
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

    def _request_body(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._request_messages(system_prompt, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Omit `tools` entirely when there are none: some endpoints reject an
        # empty array where a missing key is fine.
        if tools:
            body["tools"] = self._tool_schemas(tools)
        return json.dumps(body).encode("utf-8")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise MissingCredentialsError(MISSING_CREDENTIALS)
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            # urllib does not decompress, and SSE must be read line by line.
            "Accept-Encoding": "identity",
        }

    # -- streaming ------------------------------------------------------------

    def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
        abort_event: threading.Event | None = None,
    ) -> Generator[ProviderEvent, None, AssistantMessage]:
        """Yield streaming events; the completed AssistantMessage is the
        generator's return value (StopIteration.value)."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        chunks = self.transport(
            url,
            self._headers(),
            self._request_body(system_prompt, messages, tools),
            timeout=self.timeout,
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = Usage()
        stop_reason = "stop"
        failure_reason: str | None = None
        aborted = False

        try:
            for chunk in chunks:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    break
                parsed_usage = _usage_from(chunk.get("usage"))
                if parsed_usage is not None:
                    usage = parsed_usage
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                # The stream is untrusted JSON, so narrow before use: a
                # malformed event is skipped rather than crashing the turn.
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
                    yield TextDeltaEvent(delta=piece)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    yield ReasoningDeltaEvent(delta=reasoning)
                for call in _as_list(delta.get("tool_calls")):
                    if not isinstance(call, Mapping):
                        continue
                    index = call.get("index", 0)
                    acc = tool_calls.setdefault(
                        index if isinstance(index, int) else 0,
                        {"id": None, "name": "", "arguments": ""},
                    )
                    if isinstance(call.get("id"), str):
                        acc["id"] = call["id"]
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        name = function.get("name")
                        if isinstance(name, str) and name:
                            acc["name"] += name
                            yield ToolNameEvent(name=acc["name"])
                        fragment = function.get("arguments")
                        if isinstance(fragment, str) and fragment:
                            acc["arguments"] += fragment
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    stop_reason = _STOP_REASONS.get(finish_reason, "stop")
                    if stop_reason == "error":
                        failure_reason = finish_reason
        except KeyboardInterrupt:
            # Ctrl-C surfaced inside the stream: finalize as aborted.
            aborted = True
        finally:
            # Deterministic: drop the connection instead of draining it.
            _close_stream(chunks)

        content: list[TextContent | ToolCall] = []
        # No empty text block when nothing was streamed: it would be the whole
        # content of an interrupted turn, which providers reject.
        if text := "".join(text_parts):
            content.append(TextContent(text=text))
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


# OpenAI's documented reasons, plus the DeepSeek-specific failures tau has seen.
_STOP_REASONS = {
    "stop": "stop",
    "tool_calls": "toolUse",
    "length": "length",
    "aborted": "aborted",
    "content_filter": "error",
    "insufficient_system_resource": "error",
}


def _usage_from(payload: Any) -> Usage | None:
    if not isinstance(payload, Mapping):
        return None
    return Usage(
        input=_as_int(payload.get("prompt_tokens")),
        output=_as_int(payload.get("completion_tokens")),
        total_tokens=_as_int(payload.get("total_tokens")),
    )


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _close_stream(stream: Any) -> None:
    """Close a transport generator if it is closable, ignoring the rest."""
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def new_call_id() -> str:
    import uuid

    return f"call_{uuid.uuid4().hex[:24]}"
