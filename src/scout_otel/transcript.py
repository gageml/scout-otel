"""Convert OTEL spans to Scout transcripts."""

import asyncio
import contextlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from inspect_ai.event._event import Event
    from inspect_ai.event._model import ModelEvent
    from inspect_ai.event._tool import ToolEvent
    from inspect_ai.model._chat_message import ChatMessage
    from inspect_scout._transcript.types import Transcript, TranscriptContent


class ScoutTranscriptExporter(SpanExporter):
    """Export OTEL spans to Scout transcript database.

    Collects spans by scout.transcript_id and flushes on shutdown.
    """

    def __init__(self, location: str, content: "TranscriptContent | None" = None):
        """Initialize the exporter.

        Args:
            location: Scout transcript database location (local path,
                S3, etc.).
            content: Which content to extract. Defaults to all messages
                and events.
        """
        from inspect_scout._transcript.types import TranscriptContent

        self.location = location
        self.content = content or TranscriptContent(messages="all", events="all")
        self._spans: dict[str, list[ReadableSpan]] = {}

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Collect spans."""
        for span in spans:
            transcript_id = (span.attributes or {}).get("scout.transcript_id")
            if isinstance(transcript_id, str):
                self._spans.setdefault(transcript_id, []).append(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Flush any remaining transcripts on shutdown."""
        self._flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending transcripts."""
        return True

    def _flush(self) -> None:
        """Flush all pending transcripts."""
        from inspect_scout import transcripts_db

        if not self._spans:
            return

        # Spans to transcript
        transcripts = [
            _spans_to_transcript(tid, spans, self.content)
            for tid, spans in self._spans.items()
        ]

        async def write() -> None:
            async with transcripts_db(self.location) as db:
                await db.insert(transcripts)

        # Write transcript with disabled scout display
        with _no_scout_display():
            asyncio.run(write())

        # Flush clears buffer
        self._spans.clear()


@contextlib.contextmanager
def _no_scout_display() -> Iterator[None]:
    """Temporarily suppress Scout's display output during DB operations."""
    import inspect_scout._display._display as display_module
    from inspect_scout._display.none import DisplayNone

    original = display_module._display
    display_module._display = DisplayNone()
    try:
        yield
    finally:
        display_module._display = original


def _spans_to_transcript(
    transcript_id: str,
    spans: list[ReadableSpan],
    content: "TranscriptContent",
) -> "Transcript":
    """Convert spans to a Scout Transcript."""
    from inspect_scout._transcript.types import Transcript

    summary = _span_summary(spans, content)

    return Transcript(
        transcript_id=transcript_id,
        model=summary.model,
        date=datetime.now(UTC).isoformat(),
        total_tokens=summary.total_tokens or None,
        total_time=summary.total_time,
        messages=summary.messages,
        events=summary.events,
        **summary.scout_attrs,
    )


@dataclass
class _SpanSummary:
    """Summary data extracted from spans."""

    messages: list["ChatMessage"] = field(default_factory=list)
    events: list["Event"] = field(default_factory=list)
    total_tokens: int = 0
    scout_attrs: dict[str, Any] = field(default_factory=dict)
    scout_model: str | None = None
    gen_ai_model: str | None = None
    start: int | None = None
    end: int | None = None

    @property
    def total_time(self) -> float | None:
        if self.start is not None and self.end is not None:
            return (self.end - self.start) / 1e9
        return None

    @property
    def model(self) -> str | None:
        return self.scout_model or self.gen_ai_model


def _span_summary(
    spans: list[ReadableSpan],
    content: "TranscriptContent",
) -> _SpanSummary:
    """Traverse spans top-down to build message and event sequences.

    Algorithm:

    1. Build span tree and traverse in time order (depth-first by start
       time)
    2. For each span:
       - Extract prompts and completions for messages
       - Extract tool events and model events
       - Deduplicate by message/event keys
    3. Use (role, content, tool_calls, tool_call_id) for message key
    4. Use response_id for model event key, tool call id for tool events
    """
    summary = _SpanSummary()

    # First pass: extract metadata from ALL spans (scout attrs, timing, tokens)
    for span in spans:
        attrs = span.attributes or {}

        # Scout attributes (agent name, model, etc.) - only apply from
        # first span with scout attrs
        if not summary.scout_attrs:
            scout_attrs = _scout_attrs(attrs)
            if scout_attrs:
                # Model name (from scout attrs)
                summary.scout_model = scout_attrs.pop("model", None)
                summary.scout_attrs = scout_attrs

        # Timing bounds
        summary.start = _time_bound(min, summary.start, span.start_time)
        summary.end = _time_bound(max, summary.end, span.end_time)

        # Model name (from gen_ai.request.model)
        if summary.gen_ai_model is None:
            model = attrs.get("gen_ai.request.model")
            if isinstance(model, str):
                summary.gen_ai_model = model

        # Token usage (sum from all spans)
        span_tokens = attrs.get("llm.usage.total_tokens") or attrs.get(
            "gen_ai.usage.total_tokens"
        )
        if isinstance(span_tokens, (int, float)):
            summary.total_tokens += int(span_tokens)

    if not content.messages and not content.events:
        return summary

    # Build span tree for traversal
    span_by_id: dict[int, ReadableSpan] = {}
    children: dict[int | None, list[ReadableSpan]] = {}

    for span in spans:
        span_id = span.context.span_id if span.context else None
        if span_id is not None:
            span_by_id[span_id] = span

        parent_id = span.parent.span_id if span.parent else None
        children.setdefault(parent_id, []).append(span)

    # Sort children by start time
    for child_list in children.values():
        child_list.sort(key=lambda s: s.start_time or 0)

    # Track messages by message key
    seen_message_keys: set[tuple] = set()
    messages: list[ChatMessage] = []

    # Track events by message key
    seen_event_keys: set[str] = set()
    events: list[Event] = []

    def _message_key(msg: "ChatMessage") -> tuple:
        """Create key for a message."""
        from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageTool

        role = msg.role
        text = msg.text if hasattr(msg, "text") else ""

        # Include tool_calls for assistant messages (exclude ID since it
        # varies by span)
        tool_calls_key: tuple = ()
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            tool_calls_key = tuple(
                (tc.function, json.dumps(tc.arguments, sort_keys=True))
                for tc in msg.tool_calls
            )

        # Include tool_call_id for tool messages
        tool_call_id = ""
        if isinstance(msg, ChatMessageTool):
            tool_call_id = msg.tool_call_id or ""

        return (role, text, tool_calls_key, tool_call_id)

    def _model_event_key(span: ReadableSpan) -> str | None:
        """Create key for a model event based on span attributes.

        Returns None if this span should not generate a model event
        (e.g., wrapper spans that duplicate content from child spans).
        """
        attrs = span.attributes or {}

        # Use gen_ai.response.id if available - this uniquely identifies
        # the API call. Spans without response.id are typically wrapper
        # spans that duplicate content from child spans, so we skip them
        # for event generation.
        response_id = attrs.get("gen_ai.response.id")
        if isinstance(response_id, str):
            return f"model:{response_id}"

        # No response ID means this is likely a wrapper span - skip it
        return None

    def _tool_event_key(span: ReadableSpan) -> str:
        """Create key for a tool event based on span attributes."""
        attrs = span.attributes or {}

        # Use tool call ID if available
        tool_id = attrs.get("gen_ai.tool.call.id")
        if isinstance(tool_id, str):
            return f"tool:{tool_id}"

        # Fallback to span ID
        span_id = span.context.span_id if span.context else None
        return f"tool:{span_id}"

    def _find_insert_position(span_prompts: list["ChatMessage"], msg_idx: int) -> int:
        """Find insert pos for a new message based on its neighbors."""

        # Look for a neighbor to anchor the position - check messages
        # before this one in the span's prompt array
        for i in range(msg_idx - 1, -1, -1):
            neighbor_key = _message_key(span_prompts[i])
            if neighbor_key in seen_message_keys:
                # Find this neighbor in our list and insert after it
                for j, existing in enumerate(messages):
                    if _message_key(existing) == neighbor_key:
                        return j + 1

        # No earlier neighbor found, check messages after
        for i in range(msg_idx + 1, len(span_prompts)):
            neighbor_key = _message_key(span_prompts[i])
            if neighbor_key in seen_message_keys:
                # Find this neighbor in our list and insert before it
                for j, existing in enumerate(messages):
                    if _message_key(existing) == neighbor_key:
                        return j

        # No neighbors found, append at end
        return len(messages)

    def _process_span(span: ReadableSpan) -> None:
        """Process a single span, extracting messages and events."""

        attrs = span.attributes or {}
        prompts, completions = _gen_ai_messages(attrs)

        # Messages
        if content.messages:
            # Process prompts - these show conversation history
            for i, msg in enumerate(prompts):
                key = _message_key(msg)
                if key not in seen_message_keys:
                    pos = _find_insert_position(prompts, i)
                    messages.insert(pos, msg)
                    seen_message_keys.add(key)

            # Process completions - these are new messages from this
            # call
            for msg in completions:
                key = _message_key(msg)
                if key not in seen_message_keys:
                    messages.append(msg)
                    seen_message_keys.add(key)

        # Events
        if content.events:
            # Check if this is a tool span
            tool_event = _tool_event(span)
            if tool_event:
                key = _tool_event_key(span)
                if key not in seen_event_keys:
                    events.append(tool_event)
                    seen_event_keys.add(key)
            elif prompts or completions:
                # Create a model event for LLM calls (only for spans
                # with response ID)
                key = _model_event_key(span)
                if key is not None and key not in seen_event_keys:
                    model_event = _model_event(span, prompts, completions)
                    if model_event:
                        events.append(model_event)
                        seen_event_keys.add(key)

    def _traverse(parent_id: int | None) -> None:
        """Depth-first traversal of span tree."""
        for span in children.get(parent_id, []):
            _process_span(span)
            span_id = span.context.span_id if span.context else None
            if span_id is not None:
                _traverse(span_id)

    # Traverse from root (parent_id=None) - collects messages and events
    _traverse(None)

    summary.messages = messages
    summary.events = events
    return summary


def _scout_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scout.* attributes, stripping the prefix."""
    return {
        k.removeprefix("scout."): v
        for k, v in attrs.items()
        if k.startswith("scout.") and k != "scout.transcript_id"
    }


def _time_bound(
    fn: Callable[[int, int], int], a: int | None, b: int | None
) -> int | None:
    """Apply fn to a and b if both present, else return non None val."""
    return fn(a, b) if a and b else a or b


def _gen_ai_messages(
    attrs: Mapping[str, Any],
) -> tuple[list["ChatMessage"], list["ChatMessage"]]:
    """Extract gen_ai.prompt.* and gen_ai.completion.* attributes.

    Extracts:
    - role, content, tool_call_id from prompts and completions
    - tool_calls from assistant messages (gen_ai.*.N.tool_calls.M.*)
    """
    prompts: dict[int, dict[str, Any]] = {}
    completions: dict[int, dict[str, Any]] = {}

    for key, value in attrs.items():
        if not isinstance(key, str):
            continue

        if key.startswith("gen_ai.prompt."):
            parts = key.split(".")
            if len(parts) >= 4 and parts[3] in ("role", "content", "tool_call_id"):
                idx = int(parts[2])
                prompts.setdefault(idx, {})[parts[3]] = value
            elif len(parts) >= 6 and parts[3] == "tool_calls":
                # gen_ai.prompt.N.tool_calls.M.field (for assistant
                # messages with tool calls)
                idx = int(parts[2])
                tool_idx = int(parts[4])
                field = parts[5]
                prompts.setdefault(idx, {})
                tool_calls = prompts[idx].setdefault("tool_calls", {})
                tool_calls.setdefault(tool_idx, {})[field] = value

        elif key.startswith("gen_ai.completion."):
            parts = key.split(".")
            if len(parts) < 4:
                continue

            # parts[2] should be a numeric index (e.g.,
            # gen_ai.completion.0.role). Tool spans use
            # gen_ai.completion.tool.* for tool metadata, which is not a
            # chat message—skip those.
            if not parts[2].isdigit():
                continue

            idx = int(parts[2])
            completions.setdefault(idx, {})

            if parts[3] in ("role", "content"):
                completions[idx][parts[3]] = value
            elif parts[3] == "tool_calls" and len(parts) >= 6:
                # gen_ai.completion.N.tool_calls.M.field
                tool_idx = int(parts[4])
                field = parts[5]
                tool_calls = completions[idx].setdefault("tool_calls", {})
                tool_calls.setdefault(tool_idx, {})[field] = value

    prompt_messages = [
        _chat_message(prompts[idx])
        for idx in sorted(prompts)
        if "role" in prompts[idx]
        and ("content" in prompts[idx] or "tool_calls" in prompts[idx])
    ]

    completion_messages = [
        _chat_message(completions[idx])
        for idx in sorted(completions)
        if "role" in completions[idx]
    ]

    return prompt_messages, completion_messages


def _chat_message(msg_data: dict[str, Any]) -> "ChatMessage":
    """Create ChatMessage from dict.

    Has with 'role', 'content', and optional 'tool_calls'.
    """
    from inspect_ai.model._chat_message import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool._tool_call import ToolCall

    role = msg_data["role"]
    content = msg_data.get("content", "")

    # Extract tool_calls if present (for assistant messages)
    tool_calls_data = msg_data.get("tool_calls", {})
    tool_calls: list[ToolCall] | None = None
    if tool_calls_data:
        tool_calls = []
        for tool_idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[tool_idx]
            tool_id = tc.get("id", f"tool_{tool_idx}")
            func_name = tc.get("name", "unknown")
            arguments_str = tc.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(id=tool_id, function=func_name, arguments=arguments)
            )

    if role == "system":
        return ChatMessageSystem(content=content)
    elif role == "user":
        return ChatMessageUser(content=content)
    elif role == "tool":
        # Tool result message
        tool_call_id = msg_data.get("tool_call_id", "")
        return ChatMessageTool(content=content, tool_call_id=tool_call_id)
    else:
        # Default to assistant for completions with unknown/missing role
        return ChatMessageAssistant(content=content, tool_calls=tool_calls)


def _model_event(
    span: ReadableSpan,
    prompts: list["ChatMessage"],
    completions: list["ChatMessage"],
) -> "ModelEvent | None":
    """Create a ModelEvent from span data.

    Extracts request parameters and tool definitions from span
    attributes. Returns None if the span doesn't represent an actual
    model call.
    """
    from inspect_ai.event._model import ModelEvent
    from inspect_ai.model._chat_message import ChatMessageAssistant
    from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput

    attrs = span.attributes or {}

    # Extract usage first to help determine if this is a real model call
    usage = _extract_usage(attrs)

    # Skip spans that don't have actual model content
    # A real model call should have prompts, completions, or usage data
    if not prompts and not completions and not usage:
        return None

    model = attrs.get("gen_ai.request.model") or attrs.get("scout.model") or "unknown"

    output_message = completions[0] if completions else ChatMessageAssistant(content="")
    if not isinstance(output_message, ChatMessageAssistant):
        output_message = ChatMessageAssistant(content=output_message.text)

    output = ModelOutput(
        model=str(model),
        choices=[ChatCompletionChoice(message=output_message, stop_reason="stop")],
        usage=usage,
    )

    start_time = None
    completed = None
    if span.start_time:
        start_time = datetime.fromtimestamp(span.start_time / 1e9, tz=UTC)
    if span.end_time:
        completed = datetime.fromtimestamp(span.end_time / 1e9, tz=UTC)

    # Extract request parameters
    config = _extract_config(attrs)

    # Extract tool definitions
    tools = _extract_tools(attrs)

    return ModelEvent(
        model=str(model),
        input=prompts,
        output=output,
        tools=tools,
        tool_choice="auto",
        config=config,
        timestamp=start_time or datetime.now(UTC),
        completed=completed,
    )


def _extract_config(attrs: Mapping[str, Any]) -> Any:
    """Extract generation config from span attributes."""
    from inspect_ai.model._generate_config import GenerateConfig

    # Request parameters
    temperature = attrs.get("gen_ai.request.temperature")
    top_p = attrs.get("gen_ai.request.top_p")

    # Reasoning parameters (o1/o3 models)
    reasoning_effort = attrs.get("gen_ai.request.reasoning_effort")
    reasoning_summary = attrs.get("gen_ai.request.reasoning_summary")

    # Build config with available parameters
    kwargs: dict[str, Any] = {}

    if isinstance(temperature, (int, float)):
        kwargs["temperature"] = float(temperature)

    if isinstance(top_p, (int, float)):
        kwargs["top_p"] = float(top_p)

    # reasoning_effort can be a string or list (OpenAI Agents SDK uses
    # list)
    if isinstance(reasoning_effort, str) and reasoning_effort in (
        "minimal",
        "low",
        "medium",
        "high",
    ):
        kwargs["reasoning_effort"] = reasoning_effort

    if isinstance(reasoning_summary, str) and reasoning_summary in (
        "concise",
        "detailed",
        "auto",
    ):
        kwargs["reasoning_summary"] = reasoning_summary

    return GenerateConfig(**kwargs)


def _extract_tools(attrs: Mapping[str, Any]) -> list[Any]:
    """Extract tool definitions from span attributes.

    Tool definitions are stored as:
    - llm.request.functions.N.name
    - llm.request.functions.N.description
    - llm.request.functions.N.parameters (JSON string, OpenAI style)
    - llm.request.functions.N.input_schema (JSON string, Anthropic style)
    """
    import json

    from inspect_ai.tool._tool_info import ToolInfo
    from inspect_ai.tool._tool_params import ToolParams

    tools: dict[int, dict[str, str]] = {}

    for key, value in attrs.items():
        if not isinstance(key, str) or not key.startswith("llm.request.functions."):
            continue

        parts = key.split(".")
        if len(parts) < 4:
            continue

        try:
            idx = int(parts[3])
        except ValueError:
            continue

        field = parts[4] if len(parts) > 4 else None
        if field in ("name", "description", "parameters", "input_schema"):
            tools.setdefault(idx, {})[field] = str(value)

    result = []
    for idx in sorted(tools):
        tool_data = tools[idx]
        name = tool_data.get("name")
        description = tool_data.get("description", "")

        if not name:
            continue

        # Parse parameters schema
        params = ToolParams()
        schema_str = tool_data.get("parameters") or tool_data.get("input_schema")
        if schema_str:
            try:
                schema = json.loads(schema_str)
                if isinstance(schema, dict):
                    params = ToolParams(
                        properties=schema.get("properties", {}),
                        required=schema.get("required", []),
                    )
            except json.JSONDecodeError:
                pass

        result.append(ToolInfo(name=name, description=description, parameters=params))

    return result


def _tool_event(span: ReadableSpan) -> "ToolEvent | None":
    """Create a ToolEvent from a tool span.

    Supports multiple span formats:

    1. Traceloop style (Anthropic with @trace_tool, LangChain):
       - traceloop.span.kind=tool
       - traceloop.entity.name: tool function name
       - traceloop.entity.input: JSON with {"args": [...], "kwargs": {...}}
       - traceloop.entity.output: JSON result

    2. PydanticAI style:
       - gen_ai.tool.name: tool function name
       - tool_arguments: JSON string of arguments
       - tool_response: string result

    3. OpenAI Agents SDK style:
       - traceloop.span.kind=tool
       - gen_ai.tool.name: tool function name
       - No input/output captured (tool calls visible in LLM response spans)
    """
    import json
    import uuid

    from inspect_ai.event._tool import ToolEvent

    attrs = span.attributes or {}

    # Determine tool name from various sources
    tool_name: str | None = None
    arguments: dict[str, Any] = {}
    result: str = ""

    span_kind = attrs.get("traceloop.span.kind")

    # Pattern 1: Traceloop style (span.kind=tool with entity.* attrs)
    if span_kind == "tool":
        name_attr = attrs.get("traceloop.entity.name")
        if isinstance(name_attr, str):
            tool_name = name_attr
        else:
            name_attr = attrs.get("gen_ai.tool.name")
            if isinstance(name_attr, str):
                tool_name = name_attr

        # Parse traceloop entity input
        entity_input = attrs.get("traceloop.entity.input")
        if isinstance(entity_input, str):
            try:
                parsed = json.loads(entity_input)
                if isinstance(parsed, dict):
                    # traceloop format: {"args": [...], "kwargs": {...}}
                    args = parsed.get("args", [])
                    kwargs = parsed.get("kwargs", {})
                    for i, arg in enumerate(args):
                        arguments[f"arg{i}"] = arg
                    arguments.update(kwargs)
            except json.JSONDecodeError:
                pass

        # Parse traceloop entity output
        entity_output = attrs.get("traceloop.entity.output")
        if isinstance(entity_output, str):
            try:
                parsed = json.loads(entity_output)
                result = json.dumps(parsed) if not isinstance(parsed, str) else parsed
            except json.JSONDecodeError:
                result = entity_output

    # Pattern 2: PydanticAI style (gen_ai.tool.name with tool_arguments/tool_response)
    elif "gen_ai.tool.name" in attrs and "tool_response" in attrs:
        name_attr = attrs.get("gen_ai.tool.name")
        if isinstance(name_attr, str):
            tool_name = name_attr

        # Parse tool_arguments (JSON string)
        tool_args = attrs.get("tool_arguments")
        if isinstance(tool_args, str):
            try:
                parsed = json.loads(tool_args)
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                pass

        # tool_response is already a string
        tool_resp = attrs.get("tool_response")
        if isinstance(tool_resp, str):
            result = tool_resp

    # No matching pattern
    if not isinstance(tool_name, str):
        return None

    # Timestamps
    start_time = None
    completed = None
    if span.start_time:
        start_time = datetime.fromtimestamp(span.start_time / 1e9, tz=UTC)
    if span.end_time:
        completed = datetime.fromtimestamp(span.end_time / 1e9, tz=UTC)

    # Generate a tool call ID from span ID or tool call ID attribute
    tool_id = attrs.get("gen_ai.tool.call.id")
    if not isinstance(tool_id, str):
        span_id = span.context.span_id if span.context else None
        tool_id = f"tool_{span_id:016x}" if span_id else f"tool_{uuid.uuid4().hex[:12]}"

    return ToolEvent(
        id=tool_id,
        function=tool_name,
        arguments=arguments,
        result=result,
        timestamp=start_time or datetime.now(UTC),
        completed=completed,
    )


def _extract_usage(attrs: Mapping[str, Any]) -> Any:
    """Extract token usage from span attributes.

    Includes cache and reasoning token counts when available.
    """
    from inspect_ai.model._model_output import ModelUsage

    input_tokens = attrs.get("llm.usage.prompt_tokens") or attrs.get(
        "gen_ai.usage.input_tokens"
    )
    output_tokens = attrs.get("llm.usage.completion_tokens") or attrs.get(
        "gen_ai.usage.output_tokens"
    )
    total_tokens = attrs.get("llm.usage.total_tokens") or attrs.get(
        "gen_ai.usage.total_tokens"
    )

    # Cache tokens (OpenAI style)
    cache_read = attrs.get("gen_ai.usage.cache_read_input_tokens")

    # Reasoning tokens (o1/o3 models)
    reasoning_tokens = attrs.get("gen_ai.usage.reasoning_tokens")

    if not any([input_tokens, output_tokens, total_tokens]):
        return None

    assert input_tokens is None or isinstance(input_tokens, (int | float))
    assert output_tokens is None or isinstance(output_tokens, (int | float))
    assert total_tokens is None or isinstance(total_tokens, (int | float))
    assert cache_read is None or isinstance(cache_read, (int | float))
    assert reasoning_tokens is None or isinstance(reasoning_tokens, (int | float))

    return ModelUsage(
        input_tokens=int(input_tokens) if input_tokens else 0,
        output_tokens=int(output_tokens) if output_tokens else 0,
        total_tokens=int(total_tokens) if total_tokens else 0,
        input_tokens_cache_read=int(cache_read) if cache_read else None,
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens else None,
    )
