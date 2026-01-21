"""OpenAI Agents SDK example demonstrating span-to-message issue.

This agent makes a simple tool call, which creates multiple spans. We
show that the leaf span for the second LLM call is missing messages that
appear in its parent span.

Run with: python run_case.py agent
"""

import os

from agents import Agent, Runner, function_tool

from scout_otel.agent_task import run_agent_task


@function_tool
def get_time() -> str:
    """Get the current time."""
    return "The current time is 3:00 PM."


async def main():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

    agent = Agent(
        name="TimeAgent",
        instructions=(
            "You are a helpful assistant. When asked about the time, "
            "use the get_time tool."
        ),
        model="gpt-4o-mini",
        tools=[get_time],
    )

    result = await run_agent_task(
        Runner.run(agent, input="What time is it?"),
        agent="TimeAgent",
        model="gpt-4o-mini",
    )
    print(f"Response: {result.final_output}")
