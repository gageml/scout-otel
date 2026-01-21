"""Agent task context management for Scout transcript capture."""

import uuid
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from inspect_scout._transcript.types import TranscriptInfo

# Stores transcript info for the current agent task
_agent_task_info: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_task_info", default=None
)


class AgentTask:
    """Context manager for marking code as part of an agent task.

    Wraps code that makes LLM calls so they are captured into a Scout
    transcript.
    """

    _info: "TranscriptInfo"
    _token: Any

    @overload
    def __init__(self, info: "TranscriptInfo") -> None: ...

    @overload
    def __init__(self, info: str | None = None, **kw: Any) -> None: ...

    def __init__(self, info: "TranscriptInfo | str | None" = None, **kw: Any):
        from inspect_scout._transcript.types import TranscriptInfo

        if isinstance(info, TranscriptInfo):
            self._info = info
        else:
            transcript_id = info or uuid.uuid4().hex
            self._info = TranscriptInfo(
                transcript_id=transcript_id,
                **kw,
            )
        self._token = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._info, name)

    def __enter__(self) -> "AgentTask":
        self._token = _agent_task_info.set(self._info.model_dump(exclude_none=True))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _agent_task_info.reset(self._token)
        return False


@overload
def agent_task(info: "TranscriptInfo") -> Any: ...


@overload
def agent_task(info: str | None = None, **kw: Any) -> Any: ...


def agent_task(info: "TranscriptInfo | str | None" = None, **kw: Any):
    """Decorator to run a function as an agent task."""

    if "transcript_id" in kw:
        raise ValueError("Do not specify transcript_id for this decorator")

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **fn_kwargs):
            with AgentTask(info, **kw):
                return fn(*args, **fn_kwargs)

        return wrapper

    return decorator


@overload
async def run_agent_task(coro, info: "TranscriptInfo") -> Any: ...


@overload
async def run_agent_task(coro, info: str | None = None, **kw: Any) -> Any: ...


async def run_agent_task(coro, info: "TranscriptInfo | str | None" = None, **kw: Any):
    """Run a coroutine as an agent task."""
    with AgentTask(info, **kw):
        return await coro
