# API Guide

Scout OTEL provides three interfaces for marking LLM calls as part of an
agent task. Each serves different use patterns.

- Context Manager: `AgentTask()`
- Decorator: `@agent_task()`
- Explicit command: `run_agent_task()`

### 1. Context Manager: `AgentTask()`

All LLM calls within the `with` block are associated with a single
transcript.

```python
from scout_otel import AgentTask

def run_poker_agent():
    with AgentTask(agent="poker-agent", task_id="hand-42"):
        response = call_llm("What should I do with pocket aces?")
        another_response = call_llm("Should I raise?")
```

**Parameters**

- `transcript_id` - Unique ID for this transcript. Auto-generated UUID
  if not provided.
- `agent` - Name of the agent executing the task.
- `task_id` - Identifier for the task (e.g., dataset sample id).
- `model` - Main model used by the agent.
- `**kw` - Additional fields passed to Scout's `TranscriptInfo`.

**When to use it**

- You have a clear runtime context (start and end times) to your agent's
  execution
- You want explicit, visible scope boundaries
- You're integrating into existing code and want minimal changes
- You need fine-grained control (e.g., only part of a function should
  generate a transcript)

### 2. Decorator: `@agent_task()`

Wraps the entire function in an agent task scope. Every LLM call within
the function (including nested calls) is associated with a transcript.

```python
from scout_otel import agent_task

@agent_task(agent="poker-agent")  # Auto-generates transcript_id
def run_poker_agent():
    response = call_llm("What should I do with pocket aces?")
    another_response = call_llm("Should I raise?")

@agent_task()  # Auto-generates transcript_id
def run_anonymous_agent():
    response = call_llm("Help me with something")
```

**Parameters**

Same as `AgentTask()` context manager (see above).

**NOTE:** `transcript_id` must be unique per transcript and so the
decorator does not accept a non-None value for this parameter.

**When to use it**

- Your agent's execution maps cleanly to a function boundary
- You're defining agent classes/modules and want capture built-in
- You don't need a meaningful transcript ID (auto-UUID is fine)

### 3. Explicit command: `run_agent_task()`

Run agent with a function to generate a transcript.

```python
from scout_otel import run_agent_task
import asyncio

async def poker_agent():
    return await async_call_llm("What should I do with pocket aces?")

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_agent_task(poker_agent(), agent="poker-agent"))
        tg.create_task(run_agent_task(chess_agent(), agent="chess-agent"))
```

**When to use it**

- You're starting async tasks with `asyncio.create_task()` or
  `TaskGroup`
- Multiple agents run concurrently and need isolated contexts
- You don't want to modify the coroutine function itself

**NOTE:** Python's `contextvars` propagate correctly to child
coroutines, but only if context is set _before_ the coroutine starts
executing. When you spawn tasks, you need to wrap the coroutine at spawn
time. The decorator can't help here as it wraps the function definition.
You need to wrap the coroutine instance.

## Which Should I Use?

| Situation                         | Interface                             |
| --------------------------------- | ------------------------------------- |
| Trace an existing function        | Context manager                       |
| Define new agent function         | Decorator                             |
| Trace part of a function          | Context manager                       |
| Spawn concurrent async tasks      | `run_agent_task()`                    |
| Don't care about transcript ID    | Any (all auto-generate UUIDs)         |
| Need to pass metadata dynamically | Context manager or `run_agent_task()` |

## Implementation

All three interfaces use the same mechanism:

1. A `ContextVar` named `agent_task_info` stores the current task
   metadata (internal)
2. `agent_task()` sets the var, yields, then resets it
3. `AgentTaskProcessor` (an OTEL SpanProcessor) reads the var when each
   span starts and sets `span.set_attribute("scout.{key}", value)` for
   each field

Decorator and async helper are thin wrappers around `agent_task()`:

## Design goals

**Minimal invasion.** The API doesn't require you to change how you call
LLMs. Traceloop handles that. You just mark task boundaries.

**Explicit over implicit.** Task boundaries are visible in the code. You
can trace the flow by reading the source.

**Use Python idioms.** Context managers for scoped resources, decorators
for function-level behavior, helpers for async task spawning. Nothing
exotic.

**One mechanism, multiple surfaces.** All interfaces share the same
ContextVar
