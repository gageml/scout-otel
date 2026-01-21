"""LangChain poker player."""

from collections.abc import Sequence
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
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


class LangChainPlayer(LLMPlayer):
    """Poker agent using LangGraph react agent with tool-based table access."""

    type = "langchain"

    def __init__(self, name: str, model: str = "gpt-4o-mini"):
        super().__init__(name, model)
        self.llm = ChatOpenAI(model=model)

    async def get_action(
        self, table: TexasHoldEm, hand: Sequence[Card]
    ) -> tuple[ActionType, int | None]:
        context = TableContext(table=table, hand=hand)
        tools = make_tools(context)

        agent = create_agent(self.llm, tools, system_prompt=POKER_SYSTEM_PROMPT)

        await agent.ainvoke({"messages": [HumanMessage(content="Action's on you")]})

        if context.action_taken is not None:
            return context.action_taken

        return fallback_action(table)


def make_tools(context: TableContext) -> list[StructuredTool]:
    """Create tools that close over the given context."""

    def get_table_state() -> str:
        """Get the current game state.

        Includes your cards, the board, pot, and available actions.
        """
        return table_state_prompt(context.table, context.hand)

    def place_action(
        action: Literal["FOLD", "CHECK", "CALL", "RAISE"],
        amount: int | None = None,
    ) -> str:
        """Place your poker action.

        Args:
            action: The action to take (FOLD, CHECK, CALL, or RAISE).
            amount: Required for RAISE - the total amount to raise to.
        """
        result = validate_action(context.table, action, amount)
        if isinstance(result, str):
            return result
        context.action_taken = result
        return f"Action {action} placed successfully."

    return [
        StructuredTool.from_function(get_table_state),
        StructuredTool.from_function(place_action),
    ]
