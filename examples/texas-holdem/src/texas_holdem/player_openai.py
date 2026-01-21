"""OpenAI Agents SDK poker player.

NOTE on traces: OpenAI Agents SDK creates tool spans via its native
TracingProcessor (FunctionSpanData), but these spans don't capture
input/output. Unlike player_anthropic which uses @trace_tool to get
traceloop.entity.input and traceloop.entity.output attributes, the
OpenAI tool spans only record tool name and type. Tool call arguments
and results are still available in the LLM response spans
(gen_ai.completion.*.tool_calls and gen_ai.prompt entries).

This behavior is left as is as it provides some variety in the traces,
as well as reflects the default behavior of OpenAI Agents.
"""

from collections.abc import Sequence
from typing import Literal

from agents import Agent, RunContextWrapper, Runner, function_tool
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


class OpenAIAgentsPlayer(LLMPlayer):
    """Poker agent using OpenAI Agents SDK with tool-based table access."""

    type = "openai"

    def __init__(self, name: str, model: str = "gpt-4o-mini"):
        super().__init__(name, model)

        @function_tool
        def get_table_state(ctx: RunContextWrapper[TableContext]) -> str:
            """Get the current game state.

            This includes your cards, the board, pot, and available actions.
            """
            return table_state_prompt(ctx.context.table, ctx.context.hand)

        @function_tool
        def place_action(
            ctx: RunContextWrapper[TableContext],
            action: Literal["FOLD", "CHECK", "CALL", "RAISE"],
            amount: int | None = None,
        ) -> str:
            """Place your poker action.

            Args:
                action: The action to take (FOLD, CHECK, CALL, or RAISE).
                amount: Required for RAISE - the total amount to raise to.
            """
            result = validate_action(ctx.context.table, action, amount)
            if isinstance(result, str):
                return result  # Error message
            ctx.context.action_taken = result
            return f"Action {action} placed successfully."

        self.agent = Agent(
            name=name,
            instructions=POKER_SYSTEM_PROMPT,
            model=model,
            tools=[get_table_state, place_action],
        )

    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        context = TableContext(table=table, hand=hand)

        await Runner.run(
            self.agent,
            input="Action's on you",
            context=context,
        )

        if context.action_taken is not None:
            return context.action_taken

        return fallback_action(table)
