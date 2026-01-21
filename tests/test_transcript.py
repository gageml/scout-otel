"""Tests for transcript conversion - the core AgentTask → Transcript pipeline."""

from inspect_scout._transcript.types import TranscriptInfo

from scout_otel.transcript import _scout_attrs


class TestScoutAttrsRoundtrip:
    """Verify TranscriptInfo fields survive the span attribute roundtrip.

    This is the crux of the library: AgentTask metadata flows through
    OTEL spans and arrive intact in the final Transcript.

    1. AgentTask stores TranscriptInfo in contextvar
    2. ScoutSpanProcessor writes fields as scout.* span attributes
    3. ScoutTranscriptExporter reads scout.* attrs back
    4. _scout_attrs_for_span strips prefix, returns dict for Transcript(**kwargs)
    """

    def test_roundtrip(self):
        # Simulate what ScoutSpanProcessor does: TranscriptInfo → span attrs
        info = TranscriptInfo(
            transcript_id="test-123",
            source_type="otel",
            source_id="source-456",
            source_uri="/path/to/source",
            date="2024-01-15T10:30:00Z",
            task_set="poker-eval",
            task_id="hand-42",
            task_repeat=3,
            agent="poker-bot",
            model="claude-3",
            score=0.85,
            success=True,
        )

        # Convert to span attributes (what ScoutSpanProcessor does)
        span_attrs = {
            f"scout.{k}": v for k, v in info.model_dump(exclude_none=True).items()
        }

        # Add some non-scout attrs that would exist on real spans
        span_attrs["gen_ai.request.model"] = "claude-3-opus"
        span_attrs["llm.usage.total_tokens"] = 1500

        extracted = _scout_attrs(span_attrs)

        # transcript_id is handled separately, should not be in extracted
        assert "transcript_id" not in extracted

        # All other TranscriptInfo fields are present
        assert extracted["source_type"] == "otel"
        assert extracted["source_id"] == "source-456"
        assert extracted["source_uri"] == "/path/to/source"
        assert extracted["date"] == "2024-01-15T10:30:00Z"
        assert extracted["task_set"] == "poker-eval"
        assert extracted["task_id"] == "hand-42"
        assert extracted["task_repeat"] == 3
        assert extracted["agent"] == "poker-bot"
        assert extracted["model"] == "claude-3"
        assert extracted["score"] == 0.85
        assert extracted["success"] is True

    def test_non_scout_attrs_excluded(self):
        span_attrs = {
            "scout.agent": "test-agent",
            "scout.model": "gpt-4",
            "gen_ai.request.model": "gpt-4-turbo",
            "llm.usage.total_tokens": 500,
            "other.random.attr": "ignored",
        }

        extracted = _scout_attrs(span_attrs)

        assert "agent" in extracted
        assert "model" in extracted
        assert "gen_ai.request.model" not in extracted
        assert "llm.usage.total_tokens" not in extracted
        assert "other.random.attr" not in extracted

    def test_empty_attrs(self):
        assert _scout_attrs({}) == {}

    def test_no_scout_attrs(self):
        span_attrs = {
            "gen_ai.request.model": "claude-3",
            "llm.usage.total_tokens": 100,
        }
        assert _scout_attrs(span_attrs) == {}
