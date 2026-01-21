"""Context variables for simple_case."""

from contextvars import ContextVar
from pathlib import Path

transcripts_dir_var: ContextVar[Path] = ContextVar("transcripts_dir")
