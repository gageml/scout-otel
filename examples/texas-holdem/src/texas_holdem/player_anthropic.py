"""Anthropic API poker player."""

from collections.abc import Sequence

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock
from texasholdem import TexasHoldEm
from texasholdem.card import Card
from texasholdem.game.action_type import ActionType
from traceloop.sdk.decorators import tool as trace_tool

from .player import (
    POKER_SYSTEM_PROMPT,
    LLMPlayer,
    fallback_action,
    table_state_prompt,
    validate_action,
)

TOOLS: list[ToolParam] = [
    {
        "name": "get_table_state",
        "description": (
            "Get the current game state. "
            "This includes your cards, the board, pot, and available actions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "place_action",
        "description": "Place your poker action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["FOLD", "CHECK", "CALL", "RAISE"],
                    "description": "The action to take.",
                },
                "amount": {
                    "type": "integer",
                    "description": (
                        "Required for RAISE - the total amount to raise to."
                    ),
                },
            },
            "required": ["action"],
        },
    },
]


@trace_tool(name="get_table_state")
def exec_get_table_state(table: TexasHoldEm, hand: Sequence[Card]) -> str:
    """Execute get_table_state tool."""
    return table_state_prompt(table, hand)


@trace_tool(name="place_action")
def exec_place_action(
    table: TexasHoldEm, action: str, amount: int | None
) -> tuple[str, tuple[ActionType, int | None] | None]:
    """Execute place_action tool. Returns (response, action_taken)."""
    validated = validate_action(table, action, amount)
    if isinstance(validated, str):
        return validated, None
    return f"Action {action} placed successfully.", validated


class AnthropicPlayer(LLMPlayer):
    """Poker agent using Anthropic API with tool-based table access."""

    type = "anthropic"

    def __init__(self, name: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(name, model)
        self.client = AsyncAnthropic()

    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        assert self.model is not None

        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": "Action's on you",
            }
        ]

        action_taken: tuple[ActionType, int | None] | None = None

        for _ in range(self.max_retries):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=POKER_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                break

            tool_results: list[ToolResultBlockParam] = []
            for block in response.content:
                if not isinstance(block, ToolUseBlock):
                    continue

                if block.name == "get_table_state":
                    resp = exec_get_table_state(table, hand)
                elif block.name == "place_action":
                    action = str(block.input.get("action", ""))
                    raw_amount = block.input.get("amount")
                    amount: int | None = None
                    if isinstance(raw_amount, int):
                        amount = raw_amount
                    resp, result = exec_place_action(table, action, amount)
                    if result is not None:
                        action_taken = result
                else:
                    resp = f"Unknown tool: {block.name}"

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": resp,
                    }
                )

            if not tool_results:
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if action_taken is not None:
                return action_taken

        if action_taken is not None:
            return action_taken

        return fallback_action(table)
