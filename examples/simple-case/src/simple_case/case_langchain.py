"""LangChain/LangGraph example demonstrating tool message capture.

This agent makes a simple tool call, similar to case_openai.py. Unlike the
OpenAI Agents SDK, LangChain instrumentation captures tool input/output
in the traceloop.entity.input and traceloop.entity.output attributes.

Run with: python run_case.py langchain
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from scout_otel.agent_task import run_agent_task

SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked about the time, use the get_time tool."
)


def get_time() -> str:
    """Get the current time."""
    return "The current time is 3:00 PM."


async def main():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [StructuredTool.from_function(get_time)]
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    async def run():
        return await agent.ainvoke(
            {"messages": [HumanMessage(content="What time is it?")]}
        )

    result = await run_agent_task(
        run(),
        agent="LangChainAgent",
        model="gpt-4o-mini",
    )
    final_message = result["messages"][-1]
    print(f"Response: {final_message.content}")
