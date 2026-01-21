"""Inspect AI case - creates golden standard transcript via inspect_scout.

Run with: uv run python run_case.py inspect
"""

import os
from pathlib import Path

from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import tool
from inspect_scout import transcripts_db, transcripts_from
from inspect_scout._transcript.types import TranscriptContent

from ._context import transcripts_dir_var


@tool
def get_time():
    async def execute():
        """Get the current time."""
        return "The current time is 3:00 PM."

    return execute


SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked about the time, use the get_time tool."
)


@task
def time_task():
    return Task(
        dataset=[Sample(input="What time is it?", target="3:00 PM")],
        solver=[system_message(SYSTEM_PROMPT), use_tools(get_time()), generate()],
        scorer=match(),
    )


async def write_transcript(log_path: str, output_dir: Path) -> None:
    """Use inspect_scout to convert eval log to transcript and write to database."""
    source = transcripts_from(log_path)

    # Collect transcripts from eval log
    transcripts_list = []
    async with source.reader() as reader:
        async for info in reader.index():
            transcript = await reader.read(
                info, TranscriptContent(messages="all", events="all")
            )
            transcripts_list.append(transcript)

    # Write to Scout transcript database (Parquet format)
    async with transcripts_db(str(output_dir)) as db:
        await db.insert(transcripts_list)

    print(f"Wrote {len(transcripts_list)} transcript(s) to: {output_dir}")


async def main():
    """Run eval and convert to transcript using inspect_scout."""
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

    transcripts_dir = transcripts_dir_var.get()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir = transcripts_dir.parent / "logs"

    # Run the eval asynchronously
    logs = await eval_async(
        time_task(),
        model="openai/gpt-4o-mini",
        log_dir=str(log_dir),
    )
    log = logs[0]

    # Get the log file path
    log_path = log.location
    assert log_path
    if log_path.startswith("file://"):
        log_path = log_path[7:]

    await write_transcript(log_path, transcripts_dir)
