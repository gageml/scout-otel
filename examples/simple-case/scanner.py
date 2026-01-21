"""Spec-based scanner for validating transcript message structure.

Checks that transcripts match an expected sequence of message types,
verifying the span-to-transcript pipeline produces correct output.
"""

from inspect_ai.model import ChatMessage
from inspect_scout import Result, Scanner, scanner
from inspect_scout._transcript.types import Transcript

# Expected message flow for the time-query cases
EXPECTED_SPEC = ["system", "user", "assistant+tool_call", "tool", "assistant"]


@scanner(messages="all")
def spec_check(spec: list[str] = EXPECTED_SPEC) -> Scanner[Transcript]:
    """Check transcript messages match expected spec."""

    async def execute(transcript: Transcript) -> Result:
        messages = transcript.messages
        actual = [_message_type(m) for m in messages]

        if actual == spec:
            return Result(value=True, explanation="Matches spec")

        # Build mismatch explanation
        lines = ["Expected vs Actual:"]
        max_len = max(len(spec), len(actual))
        for i in range(max_len):
            exp = spec[i] if i < len(spec) else "(none)"
            act = actual[i] if i < len(actual) else "(none)"
            match = "✓" if exp == act else "✗"
            lines.append(f"  {match} {exp} vs {act}")

        return Result(value=False, explanation="\n".join(lines))

    return execute


def _message_type(msg: ChatMessage) -> str:
    """Classify a message by role and tool_call presence."""
    role = msg.role
    if role == "assistant" and getattr(msg, "tool_calls", None):
        return "assistant+tool_call"
    return role
