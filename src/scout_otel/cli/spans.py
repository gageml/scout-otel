"""View span trees from JSON files, grouped by transcript."""

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ._common import (
    CompactTree,
    format_duration,
    format_timestamp,
    highlighted_content,
    pad_y,
    pager,
)
from ._highlight import mark_highlights, render_json

Span = Mapping[str, Any]


@click.command()
@click.argument("spans_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-v", "--verbose", count=True, help="Increase output detail (-vvv for full JSON)"
)
@click.option("--no-pager", is_flag=True, help="Disable paging")
@click.option(
    "-H",
    "--highlight",
    "highlights",
    multiple=True,
    help="JSONPath spec to highlight (implies -vvv)",
)
def spans(
    spans_dir: Path, verbose: int, no_pager: bool, highlights: tuple[str, ...]
) -> None:
    """View span trees from JSON files, grouped by transcript."""
    # -H implies verbose=3 (full JSON)
    if highlights:
        verbose = 3

    console = Console()
    all_spans = _load_spans(spans_dir)

    if not all_spans:
        console.print("[dim]No spans found[/dim]")
        raise SystemExit(0)

    trees = _build_trees(all_spans, verbose, list(highlights))
    with pager(console, no_pager):
        _print_header(console, spans_dir, all_spans, len(trees))
        console.print()
        for tree in trees:
            console.print(tree)
            if verbose == 0:
                console.print()


def _print_header(
    console: Console,
    spans_dir: Path,
    all_spans: list[Span],
    transcript_count: int,
) -> None:
    """Print summary header for spans directory."""
    # Collect unique values from span attributes
    task_sets: set[str] = set()
    agents: set[str] = set()
    for span in all_spans:
        attrs = span.get("attributes", {})
        if ts := attrs.get("scout.task_set"):
            task_sets.add(ts)
        if agent := attrs.get("scout.agent"):
            agents.add(agent)

    table = Table(
        show_header=False,
        box=ROUNDED,
        padding=(0, 1),
        border_style="dim",
    )
    table.add_column(style="dim")
    table.add_column()
    table.add_row("spans_dir", str(spans_dir))
    if task_sets:
        table.add_row("task_set", ", ".join(sorted(task_sets)))
    if agents:
        table.add_row("agents", ", ".join(sorted(agents)))
    table.add_row("transcripts", str(transcript_count))
    table.add_row("spans", str(len(all_spans)))
    console.print(table)


def _load_spans(spans_dir: Path) -> list[Span]:
    """Load all span JSON files from directory."""
    spans = []
    for path in spans_dir.glob("*.json"):
        with open(path) as f:
            spans.append(json.load(f))
    return spans


def _build_trees(
    spans: list[Span], verbose: int, highlights: list[str] | None = None
) -> list[Tree]:
    """Build Rich trees from spans, grouped by transcript_id with hierarchy."""
    transcripts = _group_by_transcript(spans)

    trees = []
    for transcript_id, transcript_spans in transcripts.items():
        if not transcript_spans:
            continue

        # Get metadata from first span
        first_span = transcript_spans[0]
        attrs = first_span.get("attributes", {})
        agent = attrs.get("scout.agent", "")
        model = attrs.get("scout.model", "")
        start_time = first_span.get("start_time", "")

        # Create root label
        label_parts = []
        if agent:
            label_parts.append(f"[green]{agent}[/green]")
        truncated_id = transcript_id[:8]
        label_parts.append(f"[dim]{truncated_id}[/dim]")
        if model:
            label_parts.append(f"[cyan]{model}[/cyan]")
        if start_time:
            label_parts.append(f"[dim]{format_timestamp(start_time)}[/dim]")

        tree = CompactTree(" | ".join(label_parts))

        # Build hierarchy from parent_id relationships
        _build_tree(transcript_spans, verbose, tree, highlights)

        # Use earliest start_time for sorting
        earliest = transcript_spans[0].get("start_time", 0)
        trees.append((earliest, tree))

    # Return trees sorted by start time
    return [tree for _, tree in sorted(trees, key=lambda t: t[0])]


def _group_by_transcript(spans: list[Span]) -> dict[str, list[Span]]:
    """Group spans by scout.transcript_id, sorted by start_time within each group."""
    by_transcript: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        attrs = span.get("attributes", {})
        transcript_id = attrs.get("scout.transcript_id", "unknown")
        by_transcript[transcript_id].append(span)

    # Sort spans within each transcript by start_time
    for transcript_id in by_transcript:
        by_transcript[transcript_id].sort(key=lambda s: s.get("start_time", 0))

    return by_transcript


def _build_tree(
    spans: list[Span],
    verbose: int,
    tree: Tree,
    highlights: list[str] | None = None,
) -> None:
    """Add spans to tree respecting parent_id hierarchy."""
    # Build lookup maps
    by_span_id: dict[str, Span] = {}
    children: dict[str | None, list[Span]] = defaultdict(list)

    for span in spans:
        span_id = span.get("context", {}).get("span_id")
        parent_id = span.get("parent_id")
        if span_id:
            by_span_id[span_id] = span
        children[parent_id].append(span)

    # Find root spans (no parent or parent not in this set)
    root_spans = []
    for span in spans:
        parent_id = span.get("parent_id")
        if parent_id is None or parent_id not in by_span_id:
            root_spans.append(span)

    # Sort roots by start_time
    root_spans.sort(key=lambda s: s.get("start_time", 0))

    # Recursively add spans
    def add_children(parent_tree: Tree, parent_span_id: str | None) -> None:
        child_spans = children.get(parent_span_id, [])
        child_spans.sort(key=lambda s: s.get("start_time", 0))
        for span in child_spans:
            span_id = span.get("context", {}).get("span_id")
            is_leaf = span_id is None or not children.get(span_id)
            subtree = parent_tree.add(_format_span(span, verbose, is_leaf, highlights))
            if span_id:
                add_children(subtree, span_id)

    # Add root spans and their descendants
    for span in root_spans:
        span_id = span.get("context", {}).get("span_id")
        is_leaf = span_id is None or not children.get(span_id)
        subtree = tree.add(_format_span(span, verbose, is_leaf, highlights))
        if span_id:
            add_children(subtree, span_id)


def _leaf_content_blocks(span: Span) -> Sequence[RenderableType]:
    """Get padded content blocks for a leaf span.

    Returns content wrapped with vertical padding. If no content,
    uses a placeholder.
    """
    content = _content_blocks(span)
    if not content:
        # Would like to show content with some signal but we don't have
        # it --- call attention to this with simple msg (intentionally
        # subtle)
        span_id = span.get("context", {}).get("span_id", "unknown")
        content = [Text(f"Nothing to show (span {span_id})", style="dim italic")]
    return pad_y(content)


def _format_span(
    span: Span,
    verbose: int,
    is_leaf: bool,
    highlights: list[str] | None = None,
) -> Text | Group:
    """Format a span for display.

    Format depends on verbose level from 0 (simplest) to 3 (full JSON).
    Leaf spans show content; non-leaf spans show header only (at levels 1-2).
    """
    if verbose == 0:
        return _format_level_0(span)
    elif verbose == 1:
        return _format_level_1(span, is_leaf)
    elif verbose == 2:
        return _format_level_2(span, is_leaf)
    else:
        return _format_v3(span, highlights)


def _format_span_header(span: Span) -> str:
    """Build the one-line header for a span.

    Format varies by span type (traceloop.span.kind):
    - workflow/agent: name | system | duration
    - tool: name | duration
    - LLM calls (no kind): name | model | tokens | duration
    """
    name = span.get("name", "unknown")
    attrs = span.get("attributes", {})
    duration = format_duration(span.get("start_time", ""), span.get("end_time", ""))

    status_code = span.get("status", {}).get("status_code", "")
    is_error = status_code == "ERROR"
    error_type = attrs.get("error.type", "")

    # Format name with error prefix if applicable
    if is_error:
        if error_type:
            name_part = f"[red]\\[ERROR={error_type}][/red] [bold]{name}[/bold]"
        else:
            name_part = f"[red]\\[ERROR][/red] [bold]{name}[/bold]"
    else:
        name_part = f"[bold]{name}[/bold]"

    # Build parts based on span type
    span_kind = attrs.get("traceloop.span.kind")

    if span_kind in ("workflow", "agent"):
        provider = _get_provider(attrs)
        parts = [name_part, f"[dim]{provider}[/dim]"]
        if duration:
            parts.append(f"[cyan]{duration}[/cyan]")
    else:
        # Check for tool execution (has gen_ai.tool.name)
        tool_name = attrs.get("gen_ai.tool.name")
        if tool_name:
            parts = [name_part, f"[dim]{tool_name}[/dim]"]
            if duration:
                parts.append(f"[cyan]{duration}[/cyan]")
        else:
            # Otherwise treat as LLM call
            model = attrs.get(
                "gen_ai.response.model", attrs.get("gen_ai.request.model", "")
            )
            input_tokens = attrs.get("gen_ai.usage.input_tokens", "")
            output_tokens = attrs.get("gen_ai.usage.output_tokens", "")

            parts = [name_part]
            if model:
                parts.append(f"[dim]{model}[/dim]")
            if input_tokens or output_tokens:
                parts.append(f"[yellow]{input_tokens} → {output_tokens}[/yellow]")
            if duration:
                parts.append(f"[cyan]{duration}[/cyan]")

    return " | ".join(parts)


def _get_provider(attrs: Mapping[str, Any]) -> str:
    """Get the GenAI system/provider from span attributes.

    Tries gen_ai.system (OpenLLMetry), then gen_ai.provider.name (OTel spec).
    Returns "?" if neither is present.
    """
    return attrs.get("gen_ai.system") or attrs.get("gen_ai.provider.name") or "?"


def _format_level_0(span: Span) -> Text:
    """Level 0: one-line summary only."""
    header = _format_span_header(span)
    return Text.from_markup(header)


def _format_level_1(span: Span, is_leaf: bool) -> Text | Group:
    """Level 1: header + message/tool content for leaf spans."""
    header = _format_span_header(span)
    elements: list[RenderableType] = [Text.from_markup(header)]
    if is_leaf:
        elements.extend(_leaf_content_blocks(span))
    return Group(*elements)


def _content_blocks(span: Span) -> list[RenderableType]:
    """Generate content blocks from span for display."""
    messages = _message_blocks(span)
    tools = _tool_execution_blocks(span)
    # Expect spans have messages OR tool execution content, not both
    assert not (messages and tools)
    return messages or tools


def _message_blocks(span: Span):
    """Generate message content blocks from span for display."""
    attrs = span.get("attributes", {})
    spec = _spec_message_blocks(attrs)
    legacy = _legacy_message_blocks(attrs)
    # Expect spec or legacy, not both
    assert not (spec and legacy)
    return spec or legacy


def _spec_message_blocks(attrs: Mapping[str, Any]) -> list[RenderableType]:
    """Format spec-format messages.

    OTEL LLM spec attrs:

    `gen_ai.input.messages`
    `gen_ai.output.messages`

    Message structure: [{"role": "...", "parts": [...], "finish_reason": "..."}]
    Part types: text, tool_call, tool_result
    """
    input_messages = attrs.get("gen_ai.input.messages")
    output_messages = attrs.get("gen_ai.output.messages")

    if not input_messages and not output_messages:
        return []

    elements: list[RenderableType] = []

    if input_messages:
        elements.extend(
            _spec_messages_for_attr("gen_ai.input.messages", input_messages)
        )
    if output_messages:
        if input_messages:
            elements.append(Text(""))
        elements.extend(
            _spec_messages_for_attr("gen_ai.output.messages", output_messages)
        )

    return elements


def _spec_messages_for_attr(attr_name: str, messages_json: str) -> list[RenderableType]:
    """Format a single spec-format message attribute."""
    elements: list[RenderableType] = []
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        # If parsing fails, show raw content
        elements.append(Text(attr_name, style="dim"))
        elements.append(highlighted_content(str(messages_json)))
        return elements

    for i, msg in enumerate(messages):
        if i > 0:
            elements.append(Text(""))

        parts = msg.get("parts", [])

        # Heading: attr_name role
        heading = Text(attr_name, style="dim")
        role = msg.get("role")
        if role:
            heading.append(f" {role}", style="yellow")
        elements.append(heading)

        # Message parts
        for part in parts:
            elements.append(Text(""))

            part_type = part.get("type", "")
            # Text -> show in content block
            if part_type == "text":
                content = part.get("content", "")
                if content:
                    elements.append(highlighted_content(content))

            # Tool call -> name and args
            elif part_type == "tool_call":
                tool_name = part.get("name", "unknown")
                arguments = part.get("arguments", {})
                tool_heading = Text(" → tool_call ", style="dim")
                tool_heading.append(tool_name, style="green")
                elements.append(tool_heading)
                if arguments:
                    args_str = (
                        json.dumps(arguments, indent=2)
                        if isinstance(arguments, dict)
                        else str(arguments)
                    )
                    elements.append(
                        Padding(highlighted_content(args_str), (0, 0, 0, 3))
                    )

            # Tool result -> name and result
            elif part_type == "tool_result":
                tool_name = part.get("name", "")
                tool_id = part.get("id", "")
                result = part.get("content", part.get("result", ""))
                tool_heading = Text(" → tool_result ", style="dim")
                if tool_name:
                    tool_heading.append(tool_name, style="green")
                if tool_id:
                    tool_heading.append(f" ({tool_id[:12]}...)", style="dim")
                elements.append(tool_heading)
                if result:
                    elements.append(highlighted_content(str(result)))

    return elements


def _legacy_message_blocks(attrs: Mapping[str, Any]) -> list[RenderableType]:
    """Format legacy-format messages.

    Legacy attr convention:

    `gen_ai.prompt.{N}.*`
    `gen_ai.completion.{N}.*`
    `gen_ai.completion.{N}.tool_calls.{M}.*`

    Extracts prompt and completion content from indexed attributes.
    Prefers user role for prompts, takes first completion.
    """
    elements: list[RenderableType] = []
    prompt_role, prompt, completion_role, completion = _prompt_completion(attrs)

    if prompt:
        heading = Text("gen_ai.prompt", style="dim")
        if prompt_role:
            heading.append(" ")
            heading.append(prompt_role, style="yellow")
        elements.append(heading)
        elements.append(highlighted_content(prompt))

    if completion:
        heading = Text("gen_ai.completion", style="dim")
        if completion_role:
            heading.append(" ")
            heading.append(completion_role, style="yellow")
        if prompt:
            elements.append(Text(""))
        elements.append(heading)
        elements.append(highlighted_content(completion))

    # Tool calls from completion
    tool_calls = _legacy_tool_calls(attrs)
    for tool_name, tool_args in tool_calls:
        elements.append(Text(""))
        tool_heading = Text(" → tool_call ", style="dim")
        tool_heading.append(tool_name, style="green")
        elements.append(tool_heading)
        if tool_args and tool_args != "{}":
            elements.append(Padding(highlighted_content(tool_args), (0, 0, 0, 3)))

    return elements


def _legacy_tool_calls(attrs: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Extract tool calls from legacy completion attributes.

    Parses gen_ai.completion.{N}.tool_calls.{M}.name and .arguments attrs.
    Returns list of (name, arguments) tuples.
    """
    tool_calls: list[tuple[str, str]] = []

    # Look for tool calls in completions (typically just completion 0)
    for completion_idx in range(10):
        for tool_idx in range(10):
            prefix = f"gen_ai.completion.{completion_idx}.tool_calls.{tool_idx}"
            name = attrs.get(f"{prefix}.name", "")
            if not name:
                break
            arguments = attrs.get(f"{prefix}.arguments", "")
            tool_calls.append((name, arguments))
        # Stop if no tool calls found at this completion index
        if not attrs.get(f"gen_ai.completion.{completion_idx}.tool_calls.0.name"):
            break

    return tool_calls


def _tool_execution_blocks(span: Span) -> list[RenderableType]:
    """Extract tool execution content from span for display.

    Dispatches to format-specific handlers based on available attributes.
    """
    attrs = span.get("attributes", {})

    # OpenAI Agents SDK pattern: gen_ai.tool.name + tool_arguments/tool_response
    if attrs.get("gen_ai.tool.name"):
        return _tool_execution_blocks_default(attrs)

    # LangChain/LangGraph pattern: traceloop.span.kind=tool + entity.input/output
    if attrs.get("traceloop.span.kind") == "tool":
        return _tool_execution_blocks_traceloop(attrs)

    return []


def _tool_execution_blocks_default(attrs: Mapping[str, Any]) -> list[RenderableType]:
    """Format tool execution from OpenAI Agents SDK pattern.

    Attributes:
    - gen_ai.tool.name: the tool being called
    - tool_arguments: input to the tool (JSON string)
    - tool_response: output from the tool
    """
    elements: list[RenderableType] = []

    tool_arguments = attrs.get("tool_arguments", "")
    tool_response = attrs.get("tool_response", "")

    if tool_arguments:
        elements.append(Text("tool_arguments", style="dim"))
        elements.append(highlighted_content(tool_arguments))

    if tool_response:
        if tool_arguments:
            elements.append(Text(""))
        elements.append(Text("tool_response", style="dim"))
        elements.append(highlighted_content(tool_response))

    return elements


def _tool_execution_blocks_traceloop(attrs: Mapping[str, Any]) -> list[RenderableType]:
    """Format tool execution from traceloop pattern.

    Handles two variants:
    1. LangChain/LangGraph: entity.input has 'inputs', entity.output has
       nested kwargs.content
    2. @trace_tool decorator: entity.input has direct args, entity.output
       is the return value
    """
    elements: list[RenderableType] = []

    entity_input = attrs.get("traceloop.entity.input", "")
    entity_output = attrs.get("traceloop.entity.output", "")

    # Extract tool arguments from entity.input
    if entity_input:
        try:
            input_data = json.loads(entity_input)
            # LangChain format: has 'inputs' key
            if isinstance(input_data, dict) and "inputs" in input_data:
                inputs = input_data.get("inputs", {})
                if inputs:
                    elements.append(Text("tool_arguments", style="dim"))
                    elements.append(highlighted_content(json.dumps(inputs, indent=2)))
            # @trace_tool format: direct dict of arguments
            elif isinstance(input_data, dict) and input_data:
                elements.append(Text("tool_arguments", style="dim"))
                elements.append(highlighted_content(json.dumps(input_data, indent=2)))
        except json.JSONDecodeError:
            pass

    # Extract tool result from entity.output
    if entity_output:
        try:
            output_data = json.loads(entity_output)
            # LangChain format: nested output.kwargs.content
            if isinstance(output_data, dict) and "output" in output_data:
                content = output_data.get("output", {})
                if isinstance(content, dict):
                    content = content.get("kwargs", {}).get("content", "")
                if content:
                    if elements:
                        elements.append(Text(""))
                    elements.append(Text("tool_response", style="dim"))
                    elements.append(highlighted_content(str(content)))
            # @trace_tool format: direct return value (string, list, etc.)
            else:
                if isinstance(output_data, str):
                    content = output_data
                else:
                    content = json.dumps(output_data)
                if elements:
                    elements.append(Text(""))
                elements.append(Text("tool_response", style="dim"))
                elements.append(highlighted_content(content))
        except json.JSONDecodeError:
            # Raw string output
            if elements:
                elements.append(Text(""))
            elements.append(Text("tool_response", style="dim"))
            elements.append(highlighted_content(entity_output))

    return elements


def _format_level_2(span: Span, is_leaf: bool) -> Group:
    """Level 2: level 1 + select attributes table.

    See _v2_attrs for attr selection.
    """
    header = _format_span_header(span)
    elements: list[RenderableType] = [Text.from_markup(header)]
    attrs = _level_2_attrs(span)
    if attrs:
        attr_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2, 0, 0),
            pad_edge=False,
        )
        attr_table.add_column(style="dim")
        attr_table.add_column()
        for key, value in attrs:
            attr_table.add_row(key, value)
        elements.append(Text(""))
        elements.append(attr_table)

    if is_leaf:
        elements.extend(_leaf_content_blocks(span))
    else:
        elements.append(Text(""))

    return Group(*elements)


def _level_2_attrs(span: Span) -> list[tuple[str, str]]:
    """Extract interesting span info and gen_ai.* attributes for level 2 display.

    List is sorted by status, time, and gen_ai attr keys.

    Includes span timing and filtered gen_ai attributes (excluding prompt/completion).
    """
    result = []

    # Status
    status_code = span.get("status", {}).get("status_code", "")
    if status_code:
        result.append(("status_code", status_code))

    # Span timing
    start_time = span.get("start_time", "")
    end_time = span.get("end_time", "")
    if start_time:
        result.append(("start_time", format_timestamp(start_time)))
    if end_time:
        result.append(("end_time", format_timestamp(end_time)))

    # Filter out message and tool exec content as these are shown in
    # block displays. (NOTE this logic is coupled to _message_blocks()
    # and _tool_execution_blocks() - both sections should be updated
    # with changes to this logic.)
    filter_out = [
        # Legacy message content
        re.compile(r"gen_ai\.(prompt|completion)\.\d+\.(content|role)"),
        # Spec message content
        re.compile(r"gen_ai\.(input|output)\.messages"),
        # Tool exec
        re.compile(r"tool_(arguments|response)"),
    ]
    for key, value in sorted(span.get("attributes", {}).items()):
        if not key.startswith("gen_ai.") or any(
            pattern.match(key) for pattern in filter_out
        ):
            continue
        value = _try_format_json(value) if isinstance(value, str) else str(value)
        result.append((key, value))

    return result


def _try_format_json(s: str):
    try:
        data = json.loads(s)
    except ValueError:
        return s
    else:
        return json.dumps(data, indent=2)


def _format_v3(span: Span, highlights: list[str] | None = None) -> Group:
    """Level 3: full JSON payload."""
    header = _format_span_header(span)
    if highlights:
        marked = mark_highlights(dict(span), highlights)
        json_text = render_json(marked)
        json_block = Padding(json_text, (0, 1))
    else:
        json_body = json.dumps(span, indent=2)
        json_block = highlighted_content(json_body)
    return Group(Text.from_markup(header), Padding(json_block, (1, 0)))


def _prompt_completion(attrs: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Extract prompt and completion text from span attributes.

    Returns (prompt_role, prompt_content, completion_role, completion_content).
    """
    prompt = ""
    prompt_role = ""
    completion = ""
    completion_role = ""

    # Find prompt content (look for user role first, then any)
    for i in range(10):
        role = attrs.get(f"gen_ai.prompt.{i}.role", "")
        content = attrs.get(f"gen_ai.prompt.{i}.content", "")
        if content and role == "user":
            prompt = content
            prompt_role = role
            break
        if content and not prompt:
            prompt = content
            prompt_role = role

    # Find completion content
    for i in range(10):
        role = attrs.get(f"gen_ai.completion.{i}.role", "")
        content = attrs.get(f"gen_ai.completion.{i}.content", "")
        if content:
            completion = content
            completion_role = role
            break

    return prompt_role, prompt, completion_role, completion
