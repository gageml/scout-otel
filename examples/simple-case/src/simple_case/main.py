"""Span-to-message example CLI.

Demonstrates span-to-message inference across different LLM frameworks.

Usage: span-to-message <case>
"""

import asyncio
import importlib
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from scout_otel import init_tracing

from ._context import transcripts_dir_var


@click.command()
@click.argument("case")
@click.option(
    "-d",
    "--runs-dir",
    default="./runs",
    metavar="DIR",
    help="Output directory for runs (runs)",
)
def main(case: str, runs_dir: str) -> None:
    """Run a span-to-message case.

    CASE is the name of the case to run (e.g. agent, langchain, inspect).
    """
    asyncio.run(run_case(case, runs_dir))


async def run_case(case_name: str, runs_dir: str) -> None:
    load_dotenv()

    # Import and run the case module
    module_name = f"simple_case.case_{case_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise click.ClickException(f"Case '{case_name}' not found")

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / f"{timestamp}_{case_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    spans_dir = run_dir / "spans"
    transcripts_dir = run_dir / "transcripts"

    # Make transcript dir available as context (required by inspect,
    # which needs to know where to write it's generated transcripts)
    transcripts_dir_var.set(transcripts_dir)

    # Initialize tracing
    init_tracing(
        transcripts_location=str(transcripts_dir),
        spans_dir=spans_dir,
    )

    # Case module must provide `main` async function
    click.echo(f"Running case: {case_name}")
    if not hasattr(module, "main"):
        raise click.ClickException(f"Module {module} does not export main")

    await module.main()
    click.echo()
    click.echo(f"Run output written to '{run_dir}'")


if __name__ == "__main__":
    main()
