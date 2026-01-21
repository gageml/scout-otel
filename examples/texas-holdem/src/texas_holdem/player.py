"""Poker player base classes and utilities.

Provides the Player base class, BotPlayer, and shared utilities for
LLM-powered players. Framework-specific player implementations are in
separate modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from texasholdem import TexasHoldEm
from texasholdem.card import Card
from texasholdem.game.action_type import ActionType

POKER_SYSTEM_PROMPT = "You are a poker player in a Texas Hold'em no limit game."


class ActionParseError(Exception):
    """LLM response can't be parsed as a valid action."""


class Player(ABC):
    """Base class for a poker player.

    Attributes:
        name: The player's display name at the table.
        model: LLM model identifier, or None for non-LLM players (bots).
    """

    type = None

    def __init__(self, name: str, model: str | None = None):
        self.name = name
        self.model = model

    @abstractmethod
    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        """Get the player's action for the current game state.

        Args:
            table: The current table state including board, pots, and players.
            hand: The player's two hole cards.

        Returns:
            Tuple of (action, total). For RAISE, total is the raise amount.
            For other actions, total is None.
        """


class LLMPlayer(Player):
    """Base class for LLM-powered players.

    Attributes:
        name: The player's display name at the table.
        model: The LLM model identifier used by this player.
        max_retries: Maximum retry attempts if LLM response is invalid.
    """

    def __init__(self, name: str, model: str, max_retries: int = 3):
        super().__init__(name, model)
        self.max_retries = max_retries


class Bot(Player):
    """Simple bot player using the texasholdem call_agent strategy."""

    def __init__(self, name: str):
        super().__init__(name)

    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        from texasholdem.agents.basic import call_agent

        return call_agent(table)


@dataclass
class TableContext:
    """Context for tool calls during a player's turn."""

    table: TexasHoldEm
    hand: Sequence[Card]
    action_taken: tuple[ActionType, int | None] | None = None


@dataclass
class PlayerType:
    """Configuration for a player type.

    Attributes:
        default_model: The model identifier to use when none is specified.
        player_class: The LLMPlayer subclass to instantiate.
    """

    default_model: str
    player_class: type[LLMPlayer]


def create_player(
    player_type: str,
    name: str,
    model: str | None = None,
) -> LLMPlayer:
    """Create an LLM player of the specified type.

    Args:
        player_type: One of 'openai', 'pydantic', 'langchain', 'anthropic'.
        name: The player's display name at the table.
        model: Model identifier to use. If None, uses the default for the
            player type.

    Returns:
        An LLMPlayer instance configured with the given name and model.

    Raises:
        ValueError: If player_type is not recognized.
    """
    if player_type == "openai":
        from .player_openai import OpenAIAgentsPlayer

        return OpenAIAgentsPlayer(name, model or "gpt-4o-mini")
    elif player_type == "pydantic":
        from .player_pydantic_ai import PydanticAIPlayer

        return PydanticAIPlayer(name, model or "openai:gpt-4o-mini")
    elif player_type == "langchain":
        from .player_langchain import LangChainPlayer

        return LangChainPlayer(name, model or "gpt-4o-mini")
    elif player_type == "anthropic":
        from .player_anthropic import AnthropicPlayer

        return AnthropicPlayer(name, model or "claude-sonnet-4-20250514")
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def table_state_prompt(table: TexasHoldEm, hand: Sequence[Card]) -> str:
    """Return table state for display to an agent.

    Args:
        table: The current table state.
        hand: The player's two hole cards.

    Returns:
        A formatted string containing hole cards, board, pot size,
        chip counts, and available actions.
    """
    moves = table.get_available_moves()
    raise_range = moves.raise_range

    board_str = " ".join(str(c) for c in table.board) if table.board else "none"
    pot_total = sum(p.get_total_amount() for p in table.pots)
    chips_to_call = table.chips_to_call(table.current_player)
    player_chips = table.players[table.current_player].chips

    actions_str = ", ".join(a.name for a in moves.action_types)
    raise_info = ""
    if ActionType.RAISE in moves.action_types and raise_range:
        raise_info = (
            f"\nIf raising, specify total amount "
            f"between {raise_range.start} and {raise_range.stop - 1}."
        )

    return f"""Your hole cards: {hand[0]} {hand[1]}
Board: {board_str}
Pot: {pot_total}
Your chips: {player_chips}
Chips to call: {chips_to_call}

Available actions: {actions_str}{raise_info}"""


def action_prompt(table: TexasHoldEm, hand: Sequence[Card]) -> str:
    """Return prompt describing the current table state for the LLM.

    Args:
        table: The current table state.
        hand: The player's two hole cards.

    Returns:
        A formatted prompt string containing hole cards, board, pot size,
        chip counts, and available actions, ending with a question.
    """
    return table_state_prompt(table, hand) + "\n\nWhat is your action?"


def validate_action(
    table: TexasHoldEm,
    action: str,
    amount: int | None = None,
) -> tuple[ActionType, int | None] | str:
    """Validate and normalize a poker action.

    Args:
        table: The current table state.
        action: The action name (FOLD, CHECK, CALL, RAISE).
        amount: The raise amount (required for RAISE).

    Returns:
        On success: tuple of (ActionType, normalized_amount).
        On failure: error message string.
    """
    moves = table.get_available_moves()
    available_actions = set(moves.action_types)
    raise_range = moves.raise_range

    action_map = {
        "FOLD": ActionType.FOLD,
        "CHECK": ActionType.CHECK,
        "CALL": ActionType.CALL,
        "RAISE": ActionType.RAISE,
    }

    action_type = action_map.get(action.upper())
    if action_type is None:
        return f"Unknown action: {action}. Use FOLD, CHECK, CALL, or RAISE."
    if action_type not in available_actions:
        available = ", ".join(a.name for a in moves.action_types)
        return f"Action {action} not available. Available: {available}"

    if action_type == ActionType.RAISE:
        if amount is None:
            return "RAISE requires an amount."
        if raise_range and amount not in raise_range:
            amount = max(raise_range.start, min(amount, raise_range.stop - 1))

    return (action_type, amount)


def parse_action(
    response_text: str, table: TexasHoldEm
) -> tuple[ActionType, int | None]:
    """Parse LLM response into an action tuple.

    Args:
        response_text: Raw text response from the LLM.
        table: Current table state, used to validate available actions.

    Returns:
        Tuple of (action, total). For RAISE, total is the raise amount.
        For other actions, total is None.

    Raises:
        ParseError: If the response cannot be parsed into a valid action.
    """
    moves = table.get_available_moves()
    available_actions = set(moves.action_types)
    raise_range = moves.raise_range

    parts = response_text.strip().upper().split()
    if not parts:
        raise ActionParseError("Empty response")

    action_name = parts[0]

    action_map = {
        "FOLD": ActionType.FOLD,
        "CHECK": ActionType.CHECK,
        "CALL": ActionType.CALL,
        "RAISE": ActionType.RAISE,
        "ALL_IN": ActionType.ALL_IN,
        "ALL-IN": ActionType.ALL_IN,
        "ALLIN": ActionType.ALL_IN,
    }

    action = action_map.get(action_name)
    if action is None:
        raise ActionParseError(f"Unknown action: {action_name}")
    if action not in available_actions and action != ActionType.ALL_IN:
        raise ActionParseError(f"Action not available: {action_name}")

    if action == ActionType.RAISE:
        if len(parts) < 2:
            raise ActionParseError("RAISE requires an amount")
        try:
            total = int(parts[1])
        except ValueError:
            raise ActionParseError(f"Invalid raise amount: {parts[1]}") from None
        else:
            if raise_range and total not in raise_range:
                total = max(raise_range.start, min(total, raise_range.stop - 1))
            return action, total

    return action, None


def fallback_action(table: TexasHoldEm) -> tuple[ActionType, int | None]:
    """Return a safe fallback action when LLM response parsing fails.

    Prefers CALL over CHECK over FOLD, choosing the least aggressive
    valid action to stay in the hand when possible.

    Args:
        table: The current table state.

    Returns:
        Tuple of (action, None). The action is always a non-raise action.
    """
    moves = table.get_available_moves()
    available_actions = list(moves.action_types)

    if ActionType.CALL in available_actions:
        return ActionType.CALL, None
    if ActionType.CHECK in available_actions:
        return ActionType.CHECK, None
    return ActionType.FOLD, None


def format_action(action: ActionType, total: int | None) -> str:
    """Format an action for human-readable display.

    Args:
        action: The action type (FOLD, CHECK, CALL, RAISE, etc.).
        total: The raise amount for RAISE actions, None otherwise.

    Returns:
        A string like "CALL" or "RAISE to 150".
    """
    if total is not None:
        return f"{action.name} to {total}"
    return action.name
