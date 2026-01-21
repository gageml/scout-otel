"""PydanticAI poker player."""

from collections.abc import Sequence
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from texasholdem import TexasHoldEm
from texasholdem.card import Card
from texasholdem.game.action_type import ActionType

from .player import (
    POKER_SYSTEM_PROMPT,
    LLMPlayer,
    TableContext,
    fallback_action,
    table_state_prompt,
    validate_action,
)


class PydanticAIPlayer(LLMPlayer):
    """Poker agent using PydanticAI with tool-based table access."""

    type = "pydantic_ai"

    def __init__(self, name: str, model: str = "openai:gpt-4o-mini"):
        super().__init__(name, model)

        try:
            self.agent: Agent[TableContext, str] = Agent(
                model,
                deps_type=TableContext,
                system_prompt=POKER_SYSTEM_PROMPT,
            )
        except UserError as e:
            raise ValueError(str(e)) from None

        @self.agent.tool
        def get_table_state(ctx: RunContext[TableContext]) -> str:
            """Get the current game state.

            Includes your cards, the board, pot, and available actions.
            """
            return table_state_prompt(ctx.deps.table, ctx.deps.hand)

        @self.agent.tool
        def place_action(
            ctx: RunContext[TableContext],
            action: Literal["FOLD", "CHECK", "CALL", "RAISE"],
            amount: int | None = None,
        ) -> str:
            """Place your poker action.

            Args:
                action: The action to take (FOLD, CHECK, CALL, or RAISE).
                amount: Required for RAISE - the total amount to raise to.
            """
            result = validate_action(ctx.deps.table, action, amount)
            if isinstance(result, str):
                return result  # Error message
            ctx.deps.action_taken = result
            return f"Action {action} placed successfully."

    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        context = TableContext(table=table, hand=hand)

        await self.agent.run("Action's on you", deps=context)

        if context.action_taken is not None:
            return context.action_taken

        return fallback_action(table)
