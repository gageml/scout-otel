"""Tests for AgentTask interface."""

import asyncio

import pytest
from inspect_scout._transcript.types import TranscriptInfo

from scout_otel import AgentTask, agent_task, run_agent_task


class TestInit:
    def test_transcript_info(self):
        task = AgentTask(
            TranscriptInfo(
                transcript_id="abc123",
                agent="test-agent",
                model="claude-3",
                task_id="task-1",
                task_set="eval-set",
            )
        )
        assert task.transcript_id == "abc123"
        assert task.agent == "test-agent"
        assert task.model == "claude-3"
        assert task.task_id == "task-1"
        assert task.task_set == "eval-set"

    def test_kwargs(self):
        task = AgentTask("my-id", agent="poker", model="gpt-4")
        assert task.transcript_id == "my-id"
        assert task.agent == "poker"
        assert task.model == "gpt-4"

    def test_auto_id(self):
        task = AgentTask(agent="test")
        assert task.transcript_id is not None
        assert len(task.transcript_id) == 32  # UUID hex

    def test_none_id(self):
        task = AgentTask(None, agent="test")
        assert task.transcript_id is not None
        assert len(task.transcript_id) == 32

    def test_bad_attr(self):
        task = AgentTask("test-id")
        with pytest.raises(AttributeError):
            _ = task.nonexistent_field


class TestContextManager:
    def test_enter_returns_self(self):
        task = AgentTask("test-id")
        with task as ctx:
            assert ctx is task
            assert task.transcript_id == "test-id"


class TestDecorator:
    def test_with_transcript_info(self):
        info = TranscriptInfo(transcript_id="dec-1", agent="decorated")

        @agent_task(info)
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_with_kwargs(self):
        @agent_task(agent="poker-agent")
        def my_func():
            return 42

        assert my_func() == 42

    def test_preserves_function_metadata(self):
        @agent_task(agent="test")
        def my_named_function():
            """My docstring."""
            pass

        assert my_named_function.__name__ == "my_named_function"
        assert my_named_function.__doc__ == "My docstring."

    def test_rejects_transcript_id_kwarg(self):
        with pytest.raises(ValueError, match="Do not specify transcript_id"):
            @agent_task(transcript_id="bad-id")
            def my_func():
                pass


class TestRunAsync:
    def test_with_transcript_info(self):
        info = TranscriptInfo(transcript_id="async-1", agent="async-agent")

        async def coro():
            return "async result"

        result = asyncio.run(run_agent_task(coro(), info))
        assert result == "async result"

    def test_with_kwargs(self):
        async def coro():
            return 123

        result = asyncio.run(run_agent_task(coro(), agent="test-agent"))
        assert result == 123
