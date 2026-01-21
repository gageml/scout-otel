"""Texas Hold'em poker demo: LLM agents vs bot opponents.

Demonstrates scout-otel context isolation by running LLM-powered poker
agents. All LLM calls are automatically tagged with the agent's context
ID via OTEL spans.

This CLI runs poker hands on tables. A table defines the players,
buy-in, and blinds. Built-in tables are in tables/*.toml; use --table to
select by name (e.g. `heads-up`, `friday-night`) or provide a path to a
custom TOML file.

Run `texas-holdem --help` for usage.
"""

from pathlib import Path

import click


@click.command()
@click.option(
    "--table",
    "table_config",
    default="openai",
    metavar="TABLE",
    help="Table name or path to TOML file (openai)",
)
@click.option(
    "-d",
    "--runs-dir",
    default="./runs",
    metavar="DIR",
    help="Output directory for runs (runs)",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(table_config: str, runs_dir: str, debug: bool) -> None:
    """Run a No-Limit poker hand with LLM agent(s) and/or bots.

    Tables define players, buy-in, and blinds. Built-in tables are in
    tables/*.toml. Specify a table by name (e.g. `friday-night`) or as a
    path to the config file.
    """
    import asyncio
    import logging
    from datetime import datetime

    from dotenv import load_dotenv

    from scout_otel.tracing import init_tracing

    from .config import load_table_config, make_game_id
    from .game import PlayerError, init_players, init_table, print_results, run_hand
    from .logging import logger, setup_logging

    load_dotenv()

    config = load_table_config(table_config)
    if len(config.players) < 2:
        raise click.ClickException(
            f"Invalid table config '{table_config}': must have at least 2 players"
        )

    buyin = config.buyin
    small_blind = config.small_blind
    big_blind = config.big_blind
    game_name = config.game_name
    game_id = make_game_id(game_name)

    # Create run directory: {runs_dir}/{timestamp}_{table_name}/
    table_name = Path(table_config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / f"{timestamp}_{table_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    logs_dir = run_dir / "logs"
    spans_dir = run_dir / "spans"
    transcripts_dir = run_dir / "transcripts"
    logs_dir.mkdir(exist_ok=True)
    spans_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)

    # Init Python logging
    log_file = logs_dir / f"{game_id}.log"
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_file, level)

    # Init table/players
    logger.info("Initializing table")
    try:
        players = init_players(config.players)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    table = init_table(players, buyin, small_blind, big_blind)

    # Init span tracing
    logger.info("Initializing tracing")
    init_tracing(
        transcripts_location=str(transcripts_dir),
        spans_dir=spans_dir,
    )

    # Enable PydanticAI native instrumentation if any pydantic players
    if any(p.type == "pydantic" for p in config.players):
        from pydantic_ai import Agent

        Agent.instrument_all()

    logger.info(f"Game: {game_name or 'Poker'} ({game_id})")
    logger.info(f"Players: {', '.join(p.name for p in players)}")
    logger.info(f"Buy-in: {buyin}, Blinds: {small_blind}/{big_blind}")
    logger.info(f"Output: {run_dir}")

    # Async for OTEL context isolation across concurrent player tasks
    try:
        asyncio.run(run_hand(table, players, game_id, config.response_timeout))
    except PlayerError as e:
        raise click.ClickException(str(e)) from None
    print_results(table, players, buyin=buyin)

    click.echo()
    click.echo(f"Results written to {run_dir}")
    click.echo()
    click.echo(f"Run 'scout-otel spans {spans_dir}' to view spans.")
    click.echo()
    click.echo(f"Run 'scout-otel transcripts {spans_dir}' to view transcripts.")


if __name__ == "__main__":
    main()
