"""Texas Hold'em game runner.

Orchestrates a poker hand: manages turn progression, dispatches to players,
and narrates the action.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from texasholdem import TexasHoldEm
from texasholdem.card import Card
from texasholdem.game.action_type import ActionType
from traceloop.sdk import Traceloop

from scout_otel.agent_task import AgentTask

from .config import PlayerConfig
from .logging import logger
from .player import Bot, Player, create_player, format_action


class PlayerError(Exception):
    """Raised when a player task fails."""

    def __init__(self, player_name: str, cause: BaseException):
        super().__init__(f"Player '{player_name}' failed: {cause}")
        self.player_name = player_name
        self.cause = cause


@dataclass
class TurnRequest:
    """A request for a player to take their turn.

    Attributes:
        table: Current table state.
        hand: Player's hole cards.
    """

    table: TexasHoldEm
    hand: Sequence[Card]


def init_players(player_configs: list[PlayerConfig]) -> list[Player]:
    """Create player instances from configuration.

    Args:
        player_configs: List of player configurations.

    Returns:
        List of Player instances. Index corresponds to seat number.

    Raises:
        ValueError: Invalid config.
    """
    players: list[Player] = []
    for cfg in player_configs:
        if cfg.type == "bot":
            players.append(Bot(cfg.name))
        else:
            try:
                players.append(create_player(cfg.type, cfg.name, cfg.model))
            except ValueError as e:
                raise ValueError(f"Invalid player '{cfg.name}': {e}") from None
    return players


def init_table(
    players: list[Player],
    buyin: int = 500,
    small_blind: int = 5,
    big_blind: int = 10,
) -> TexasHoldEm:
    """Create a new Texas Hold'em table.

    Args:
        players: List of players for the table.
        buyin: Starting chip stack for each player.
        small_blind: Small blind amount.
        big_blind: Big blind amount.

    Returns:
        Configured TexasHoldEm table instance.
    """

    # Currently only uses player count
    max_players = len(players)

    return TexasHoldEm(buyin, big_blind, small_blind, max_players)


async def run_hand(
    table: TexasHoldEm,
    players: list[Player],
    game_id: str,
    response_timeout: float = 30.0,
) -> None:
    """Run a single hand of poker.

    Each player runs as a concurrent async task with isolated OTEL context.
    The game progresses turn-by-turn via queue-based coordination.

    Args:
        table: The TexasHoldEm table instance.
        players: List of players (index is seat number).
        game_id: Unique identifier for grouping OTEL spans.
        response_timeout: Max seconds to wait for a player response.

    Raises:
        PlayerError: If a player times out or their action fails.
    """
    # Deal cards and post blinds
    logger.debug("Starting hand")
    table.start_hand()

    logger.info(f"Hand #{table.num_hands}")
    logger.info(f"Button: {table.btn_loc}, SB: {table.sb_loc}, BB: {table.bb_loc}")

    # Show hole cards
    for seat, player in enumerate(players):
        hand = table.get_hand(seat)
        if hand:
            logger.info(f"{player.name}'s hole cards: {hand[0]} {hand[1]}")

    # Set up player task queues
    player_queues: list[asyncio.Queue[TurnRequest | None]] = [
        asyncio.Queue() for _ in players
    ]
    response_queue: asyncio.Queue[tuple[ActionType, int | None]] = asyncio.Queue()

    # Spawn a task per player. Poker is turn-based, so these tasks don't run
    # concurrently in practice — but we structure it this way to demonstrate
    # OTEL context isolation in a concurrent async environment. Each task gets
    # its own contextvar scope, so agent_task() calls don't leak across players.
    tasks: dict[asyncio.Task[None], Player] = {
        asyncio.create_task(
            _player_task(player, game_id, player_queues[seat], response_queue)
        ): player
        for seat, player in enumerate(players)
    }

    # Current table hand phase (preflop/flop/turn/river) - detects transitions
    current_phase = None

    # Play through betting rounds until hand completes
    while table.is_hand_running():
        # Announce new community cards when phase changes
        if table.hand_phase != current_phase:
            current_phase = table.hand_phase
            if table.board:
                logger.info(f"--- {current_phase.name} ---")
                logger.info(f"Board: {' '.join(str(c) for c in table.board)}")

        # Current player info from table state
        seat = table.current_player
        player = players[seat]
        chips = table.players[seat].chips
        hand = table.get_hand(seat)

        # Request action from player
        logger.debug(f"Requesting action from {player.name} ({player.model})")
        request_time = asyncio.get_event_loop().time()
        await player_queues[seat].put(TurnRequest(table=table, hand=hand))

        # Poll for response, checking for task failures between attempts
        logger.debug(f"Waiting for response from {player.name}")
        timeout_at = request_time + response_timeout
        response = None
        while True:
            try:
                response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=5.0,  # Poll interval
                )
                break
            except TimeoutError:
                # Check if any player task has exited
                for task in tasks:
                    if task.done():
                        try:
                            task.result()  # Re-raises the exception from the task
                        except:
                            raise
                        else:
                            raise RuntimeError("Unexpected task exit")
                # Check overall timeout
                if asyncio.get_event_loop().time() >= timeout_at:
                    logger.warning(f"Timeout waiting for response from {player.name}")
                    raise PlayerError(
                        player.name, TimeoutError("No response from player")
                    )
        response_time = asyncio.get_event_loop().time()
        elapsed_s = response_time - request_time
        logger.debug(f"Got response from {player.name}")

        action, total = response
        action_str = format_action(action, total)
        logger.info(f"{player.name} [{chips} chips]: {action_str} ({elapsed_s:.1f}s)")

        # Apply action (has side effect of setting next player via table.current_player)
        table.take_action(action, total=total)

    # Send None sentinel to each player queue, triggering task exit
    for queue in player_queues:
        await queue.put(None)

    # Wait for all player tasks to complete
    await asyncio.gather(*tasks.keys())


async def _player_task(
    player: Player,
    game_id: str,
    turn_queue: asyncio.Queue[TurnRequest | None],
    response_queue: asyncio.Queue[tuple[ActionType, int | None]],
) -> None:
    """Process turns for a single player.

    Waits for turn requests, executes the player's action within an isolated
    OTEL context, and sends the response back to the game loop.

    Args:
        player: The player instance.
        game_id: Unique identifier for grouping OTEL spans.
        turn_queue: Queue for receiving turn requests. None signals shutdown.
        response_queue: Queue for sending action responses.

    Raises:
        PlayerError: If the player's action fails. At DEBUG log level,
            the original exception is re-raised for full traceback.
    """
    logger.debug(f"Player task started: {player.name}")
    while True:
        turn = await turn_queue.get()
        if turn is None:
            logger.debug(f"Player task exiting: {player.name}")
            break

        # Wrap LLM call in agent_task context so OpenLLMetry spans are tagged
        # with this player's identity (agent name, model, game_id). This is
        # how we correlate LLM telemetry with the logical agent that made it.
        logger.debug(f"Calling LLM for {player.name} ({player.model})")
        Traceloop.set_association_properties(
            {
                "player": player.name,
                "model": player.model,
                "game": game_id,
            }
        )
        with AgentTask(
            agent=player.name,
            model=player.model,
            task_set=game_id,
        ):
            try:
                action, total = await player.get_action(turn.table, turn.hand)
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    raise  # Full traceback at DEBUG level
                raise PlayerError(player.name, e) from None
        logger.debug(f"LLM returned for {player.name}: {action}")

        await response_queue.put((action, total))


def print_results(
    table: TexasHoldEm,
    players: list[Player],
    buyin: int = 500,
) -> None:
    """Print hand results including winners and chip counts.

    Args:
        table: The completed table instance.
        players: List of players.
        buyin: Original buy-in amount for calculating profit/loss.
    """
    assert table.hand_history

    logger.info("=== HAND COMPLETE ===")

    if table.board:
        logger.info(f"Final board: {' '.join(str(c) for c in table.board)}")

    for seat, player in enumerate(players):
        hand = table.get_hand(seat)
        if hand:
            logger.info(f"{player.name}: {hand[0]} {hand[1]}")

    settle = table.hand_history.settle
    if settle is not None and settle.pot_winners:
        for pot_idx, (chips, _, winner_seats) in settle.pot_winners.items():
            winner_names = [players[s].name for s in winner_seats]
            pot_label = "main pot" if pot_idx == 0 else f"side pot {pot_idx}"
            if len(winner_names) == 1:
                logger.info(f"{winner_names[0]} wins {pot_label} ({chips} chips)")
            else:
                names = ", ".join(winner_names)
                logger.info(f"{names} split {pot_label} ({chips} chips)")

    logger.info("Final chip counts:")
    for seat, player in enumerate(players):
        chips = table.players[seat].chips
        diff = chips - buyin
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        logger.info(f"  {player.name}: {chips} ({diff_str})")
