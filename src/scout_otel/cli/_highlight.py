"""JSON highlighting by path patterns.

Supports a simple path syntax using `/` as delimiter:
  - `attributes/gen_ai.prompt.*.content` - glob pattern on keys
  - `/` separates path segments (object traversal)
  - `.` is literal (part of key names)
  - `*` matches any characters within a key (glob-style)

Also supports JSONPath syntax (starting with `$`):
  - `$.attributes."gen_ai.prompt.0.content"` - exact key match
"""

import fnmatch
import json
from copy import deepcopy
from typing import Any

from jsonpath_ng import parse
from rich.text import Text

HIGHLIGHT_KEY = "__highlight__"
HIGHLIGHT_STYLE = "bold bright_yellow"


def mark_highlights(data: Any, specs: list[str]) -> Any:
    """Return a copy of data with matched paths wrapped in highlight markers.

    Each spec is applied to the data. Matched values are wrapped
    as {"__highlight__": value}.

    Specs can be:
      - JSONPath (starting with $): $.attributes."key.name"
      - Glob path (using /): attributes/gen_ai.prompt.*.content
    """
    if not specs:
        return data

    result = deepcopy(data)

    for spec in specs:
        if spec.startswith("$"):
            _mark_jsonpath(result, spec)
        else:
            _mark_glob_path(result, spec)

    return result


def _mark_jsonpath(data: Any, spec: str) -> None:
    """Mark matches using JSONPath syntax."""
    expr = parse(spec)
    for match in expr.find(data):
        match.full_path.update(data, {HIGHLIGHT_KEY: match.value})


def _mark_glob_path(data: Any, spec: str) -> None:
    """Mark matches using glob path syntax (attributes/gen_ai.*.content)."""
    parts = spec.split("/")
    _mark_glob_recursive(data, parts)


def _mark_glob_recursive(data: Any, parts: list[str]) -> None:
    """Recursively traverse and mark matches for glob path."""
    if not parts or not isinstance(data, dict):
        return

    pattern = parts[0]
    remaining = parts[1:]

    for key in list(data.keys()):
        if fnmatch.fnmatch(key, pattern):
            if remaining:
                _mark_glob_recursive(data[key], remaining)
            else:
                # Leaf match - wrap value
                data[key] = {HIGHLIGHT_KEY: data[key]}


def render_json(data: Any, indent: int = 2) -> Text:
    """Render data as JSON with highlighted values styled.

    Values wrapped in {"__highlight__": value} are rendered with
    bright yellow styling, with the wrapper removed from output.
    """
    text = Text()
    _render_value(data, text, indent, 0)
    return text


def _render_value(
    value: Any,
    text: Text,
    indent: int,
    depth: int,
    highlight: bool = False,
) -> None:
    """Recursively render a JSON value to Rich Text."""
    style = HIGHLIGHT_STYLE if highlight else None

    # Check for highlight wrapper
    if isinstance(value, dict) and list(value.keys()) == [HIGHLIGHT_KEY]:
        _render_value(value[HIGHLIGHT_KEY], text, indent, depth, highlight=True)
        return

    if value is None:
        text.append("null", style)
    elif isinstance(value, bool):
        text.append("true" if value else "false", style)
    elif isinstance(value, (int, float)):
        text.append(str(value), style)
    elif isinstance(value, str):
        escaped = json.dumps(value)
        text.append(escaped, style)
    elif isinstance(value, list):
        _render_list(value, text, indent, depth, style)
    elif isinstance(value, dict):
        _render_dict(value, text, indent, depth, style)
    else:
        text.append(repr(value), style)


def _render_list(
    items: list,
    text: Text,
    indent: int,
    depth: int,
    style: str | None,
) -> None:
    """Render a JSON array."""
    highlight = style is not None
    if not items:
        text.append("[]", style)
        return

    text.append("[\n", style)
    for i, item in enumerate(items):
        text.append(" " * indent * (depth + 1), style)
        _render_value(item, text, indent, depth + 1, highlight=highlight)
        if i < len(items) - 1:
            text.append(",", style)
        text.append("\n", style)
    text.append(" " * indent * depth, style)
    text.append("]", style)


def _render_dict(
    obj: dict,
    text: Text,
    indent: int,
    depth: int,
    style: str | None,
) -> None:
    """Render a JSON object."""
    highlight = style is not None
    if not obj:
        text.append("{}", style)
        return

    text.append("{\n", style)
    items = list(obj.items())
    for i, (key, value) in enumerate(items):
        # Check if this value is marked for highlighting
        value_is_highlighted = _is_highlight_wrapper(value)
        key_style = HIGHLIGHT_STYLE if value_is_highlighted else style

        text.append(" " * indent * (depth + 1), style)
        text.append(json.dumps(key), key_style)
        text.append(": ", key_style)
        _render_value(value, text, indent, depth + 1, highlight=highlight)
        if i < len(items) - 1:
            text.append(",", key_style if value_is_highlighted else style)
        text.append("\n", style)
    text.append(" " * indent * depth, style)
    text.append("}", style)


def _is_highlight_wrapper(value: Any) -> bool:
    """Check if value is a highlight wrapper dict."""
    return isinstance(value, dict) and list(value.keys()) == [HIGHLIGHT_KEY]
