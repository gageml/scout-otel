"""Functions shared across commands."""

import os
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree


class CompactTree(Tree):
    """Tree with narrower 3-character indentation."""

    TREE_GUIDES = [
        ("   ", "│  ", "├─ ", "└─ "),
    ]


@contextmanager
def pager(console: Console, no_pager: bool = False):
    """Use pager unless no_pager is True."""
    if not no_pager:
        old_less = os.environ.get("LESS", "")
        os.environ["LESS"] = "-R"
        try:
            with console.pager(styles=True):
                yield
        finally:
            if old_less:
                os.environ["LESS"] = old_less
            else:
                os.environ.pop("LESS", None)
    else:
        yield


def format_duration(start_time: str, end_time: str) -> str:
    """Format duration between two ISO timestamps."""
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        delta = end - start
        ms = delta.total_seconds() * 1000
        if ms < 1000:
            return f"{ms:.0f}ms"
        else:
            return f"{ms / 1000:.1f}s"
    except (ValueError, TypeError):
        return ""


def format_timestamp(timestamp: str) -> str:
    """Format an ISO timestamp for display in local time.

    Shows time only for today, full date/time otherwise.
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        now = datetime.now().astimezone()

        if local_dt.date() == now.date():
            return local_dt.strftime("%H:%M:%S")
        else:
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""


def highlighted_content(content: str) -> Padding:
    """Wrap content in a styled block with padding."""
    return Padding(
        Padding(content, (0, 1), style="on grey19"),
        (0, 1, 0, 0),
    )


def pad_y(elements: Sequence[RenderableType]):
    return [Text(""), *elements, Text("")] if elements else elements
