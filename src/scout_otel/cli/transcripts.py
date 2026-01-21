"""View transcripts from Scout transcript database."""

import asyncio

import click
from rich.box import ROUNDED
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from ._common import CompactTree, format_timestamp, highlighted_content, pager


@click.command()
@click.argument("location", type=click.Path(exists=True))
@click.option(
    "-v", "--verbose", count=True, help="Increase output detail (-vvv for full)"
)
@click.option("--no-pager", is_flag=True, help="Disable paging")
def transcripts(location: str, verbose: int, no_pager: bool) -> None:
    """View transcripts from Scout transcript database."""
    console = Console()
    all_transcripts = asyncio.run(_load_transcripts(location))

    if not all_transcripts:
        console.print("[dim]No transcripts found[/dim]")
        raise SystemExit(0)

    trees = _build_trees(all_transcripts, verbose)
    with pager(console, no_pager):
        _print_header(console, location, all_transcripts)
        console.print()
        for tree in trees:
            console.print(tree)
            console.print()


def _print_header(console: Console, location: str, all_transcripts: list) -> None:
    """Print summary header for transcripts directory."""
    task_sets: set[str] = set()
    agents: set[str] = set()
    for transcript in all_transcripts:
        if transcript.task_set:
            task_sets.add(transcript.task_set)
        if transcript.agent:
            agents.add(transcript.agent)

    table = Table(
        show_header=False,
        box=ROUNDED,
        padding=(0, 1),
        border_style="dim",
    )
    table.add_column(style="dim")
    table.add_column()
    table.add_row("transcripts_dir", location)
    if task_sets:
        table.add_row("task_set", ", ".join(sorted(task_sets)))
    if agents:
        table.add_row("agents", ", ".join(sorted(agents)))
    table.add_row("transcripts", str(len(all_transcripts)))
    console.print(table)


async def _load_transcripts(location: str) -> list:
    """Load transcripts from database."""
    from inspect_scout import transcripts_db
    from inspect_scout._transcript.types import TranscriptContent

    result = []
    async with transcripts_db(location) as db:
        async for info in db.select():
            transcript = await db.read(
                info, TranscriptContent(messages="all", events="all")
            )
            result.append(transcript)
    return result


def _build_trees(transcripts: list, verbose: int) -> list[RenderableType]:
    """Build Rich trees from transcripts."""
    trees = []

    for transcript in transcripts:
        if verbose >= 3:
            tree = _format_transcript_v3(transcript)
        else:
            tree = _format_transcript_v0(transcript)

        trees.append((transcript.date or "", tree))

    # Sort by date and return trees
    return [tree for _, tree in sorted(trees, key=lambda t: t[0])]


def _format_transcript_header(transcript) -> str:
    """Build the one-line header for a transcript."""
    label_parts = []
    if transcript.agent:
        label_parts.append(f"[green]{transcript.agent}[/green]")
    truncated_id = transcript.transcript_id[:8]
    label_parts.append(f"[dim]{truncated_id}[/dim]")
    if transcript.model:
        label_parts.append(f"[cyan]{transcript.model}[/cyan]")
    if transcript.date:
        label_parts.append(f"[dim]{format_timestamp(transcript.date)}[/dim]")
    return " | ".join(label_parts)


def _format_transcript_v0(transcript) -> CompactTree:
    """Level 0: tree with messages and events nodes."""
    tree = CompactTree(_format_transcript_header(transcript))

    # Add messages under Messages node
    if transcript.messages:
        messages_node = tree.add("[bold]Messages[/bold]")
        for msg in transcript.messages:
            messages_node.add(_format_message(msg))

    # Add events under Events node
    if transcript.events:
        events_node = tree.add("[bold]Events[/bold]")
        for event in transcript.events:
            events_node.add(_format_event(event))

    return tree


def _format_transcript_v3(transcript) -> CompactTree:
    """Level 3: full JSON payload organized by section."""
    import json

    tree = CompactTree(_format_transcript_header(transcript))

    # Transcript Info section (everything except messages and events)
    info_data = transcript.model_dump(exclude={"messages", "events"})
    info_node = tree.add("[bold]Transcript Info[/bold]")
    info_json = json.dumps(info_data, indent=2, default=str)
    info_node.add(Padding(highlighted_content(info_json), (1, 0)))

    # Messages section
    messages_node = tree.add("[bold]Messages[/bold]")
    messages_data = [msg.model_dump() for msg in transcript.messages]
    messages_json = json.dumps(messages_data, indent=2, default=str)
    messages_node.add(Padding(highlighted_content(messages_json), (1, 0)))

    # Events section
    events_node = tree.add("[bold]Events[/bold]")
    events_data = [event.model_dump() for event in transcript.events]
    events_json = json.dumps(events_data, indent=2, default=str)
    events_node.add(Padding(highlighted_content(events_json), (1, 0)))
    return tree


def _format_message(msg) -> Text:
    """Format a message as one-line header."""
    role = msg.role
    text = msg.text[:60].replace("\n", " ") if msg.text else ""

    if role == "system":
        return Text.from_markup(f"[bold]system[/bold] | [dim]{text}...[/dim]")
    elif role == "user":
        return Text.from_markup(f"[bold]user[/bold] | [dim]{text}...[/dim]")
    elif role == "assistant":
        # Check for tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tool_names = ", ".join(tc.function for tc in tool_calls)
            return Text.from_markup(
                f"[bold]assistant[/bold] | [yellow]tools: {tool_names}[/yellow]"
            )
        return Text.from_markup(f"[bold]assistant[/bold] | [dim]{text}...[/dim]")
    elif role == "tool":
        func = getattr(msg, "function", None) or ""
        return Text.from_markup(f"[bold]tool[/bold] | [green]{func}[/green]")
    else:
        return Text.from_markup(f"[bold]{role}[/bold] | [dim]{text}...[/dim]")


def _format_event(event) -> Text:
    """Format an event as one-line header."""
    event_type = getattr(event, "event", "unknown")

    if event_type == "model":
        model = getattr(event, "model", "")
        output = getattr(event, "output", None)
        usage = getattr(output, "usage", None) if output else None
        if usage:
            input_t = getattr(usage, "input_tokens", "")
            output_t = getattr(usage, "output_tokens", "")
            return Text.from_markup(
                f"[bold magenta]model[/bold magenta] | [dim]{model}[/dim] | "
                f"[yellow]{input_t} → {output_t}[/yellow]"
            )
        return Text.from_markup(
            f"[bold magenta]model[/bold magenta] | [dim]{model}[/dim]"
        )

    elif event_type == "tool":
        func = getattr(event, "function", "")
        return Text.from_markup(
            f"[bold magenta]tool[/bold magenta] | [green]{func}[/green]"
        )

    else:
        return Text.from_markup(f"[bold magenta]{event_type}[/bold magenta]")
