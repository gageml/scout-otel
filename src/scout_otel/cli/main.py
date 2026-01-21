"""Scout OTEL CLI entry point."""

import click

from .spans import spans
from .transcripts import transcripts


@click.group()
def cli() -> None:
    """Scout OTEL tools."""


cli.add_command(spans)
cli.add_command(transcripts)


def main():
    cli()


if __name__ == "__main__":
    main()
