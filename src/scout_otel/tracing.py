"""Traceloop initialization and span export."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_scout._transcript.types import TranscriptContent

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from .agent_task import _agent_task_info
from .transcript import ScoutTranscriptExporter


def init_tracing(
    transcripts_location: str,
    *,
    transcript_content: "TranscriptContent | None" = None,
    spans_dir: str | Path | None = None,
    app_name: str = "scout-otel",
) -> None:
    """Initialize tracing with Scout span processor.

    Args:
        transcripts_location: Scout transcript database location. Passed through
            to Scout's transcripts_db() API - can be local path, S3 URL, etc.
        transcript_content: Filter for transcript content (messages/events).
        spans_dir: Directory for local JSON span files. None disables local export.
        app_name: Application name for OTEL resource attributes.
    """
    from traceloop.sdk import Traceloop

    Traceloop.init(
        app_name=app_name,
        processor=(
            [
                # Tag spans with AgentTask context (transcript_id, etc.)
                ScoutSpanProcessor(),
                # Write completed spans to Scout transcript database
                SimpleSpanProcessor(
                    ScoutTranscriptExporter(
                        transcripts_location,
                        transcript_content,
                    )
                ),
            ]
            # Optionally write spans locally (e.g. debugging)
            + _maybe_spans(spans_dir)
        ),
        # Disable batching when debugging locally
        disable_batch=spans_dir is not None,
    )


def _maybe_spans(dir: str | Path | None) -> list[SpanProcessor]:
    """Export spans as individual JSON files."""
    return [SimpleSpanProcessor(JsonFileSpanExporter(dir))] if dir else []


class ScoutSpanProcessor(SpanProcessor):
    """Processor that tags spans with Scout transcript metadata."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        info = _agent_task_info.get()
        if info is None:
            return
        for key, value in info.items():
            # OTEL attributes only support primitives and sequences of primitives
            if value is None or isinstance(value, dict):
                continue
            span.set_attribute(f"scout.{key}", value)

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True  # OTEL's MultiSpanProcessor checks return value


class JsonFileSpanExporter(SpanExporter):
    """Export spans as individual JSON files."""

    def __init__(self, output_dir: str | Path = "spans"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            assert span.context
            span_id = format(span.context.span_id, "016x")
            file_path = self.output_dir / f"{span_id}.json"
            file_path.write_text(span.to_json(indent=2))
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True
