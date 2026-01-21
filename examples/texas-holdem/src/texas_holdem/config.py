"""Table configuration for No-Limit Texas Hold'em games.

Supports loading table setups from TOML files, defining players, game
settings, and table parameters.
"""

import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dacite import from_dict


@dataclass
class PlayerConfig:
    """Configuration for a single player at the table."""

    name: str  # Poker name (e.g., "Fast Eddie", "Billy the Hat")
    type: str  # "openai", "anthropic", "pydantic", "langchain", or "bot"
    model: str | None = None  # Model identifier for LLM players (ignored for bots)


@dataclass
class TableConfig:
    """Configuration for a poker table and game."""

    game_name: str | None = None  # Human-friendly name (e.g., "Friday Night Poker")
    players: list[PlayerConfig] = field(default_factory=list)
    buyin: int = 500  # Starting chip stack
    small_blind: int = 5
    big_blind: int = 10
    response_timeout: float = 30.0  # Max time to wait for player response


def make_game_id(game_name: str | None = None) -> str:
    """Generate a unique game ID from an optional game name.

    Returns a lowercase kebab-case string derived from the game name with
    a 6-character random suffix for uniqueness.

    Args:
        game_name: Optional human-friendly game name.

    Returns:
        Unique game ID string.

    Examples:
        make_game_id("Friday Night Poker") -> "friday-night-poker-a3f9c2"
        make_game_id() -> "game-a3f9c2"
    """
    if game_name:
        base = game_name.lower().replace(" ", "-")
        # Remove non-alphanumeric except hyphens
        base = "".join(c for c in base if c.isalnum() or c == "-")
    else:
        base = "game"

    suffix = secrets.token_hex(3)  # 6 hex characters
    return f"{base}-{suffix}"


def load_table_config(path: str | Path) -> TableConfig:
    """Load table configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file, or a table name.
            If the path doesn't exist as a file, it's treated as a name
            and looked up in the bundled tables/ directory.

    Returns:
        TableConfig instance populated from the file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is invalid or missing required fields.

    Example TOML format:
        ```toml
        game_name = "Friday Night Poker"
        buyin = 500
        small_blind = 5
        big_blind = 10

        [[players]]
        name = "Fast Eddie"
        type = "openai"
        model = "gpt-4o-mini"

        [[players]]
        name = "Billy the Hat"
        type = "anthropic"

        [[players]]
        name = "Quiet Joe"
        type = "bot"
        ```
    """
    path = _resolve_table_path(path)

    with path.open("rb") as f:
        data = tomllib.load(f)

    if data.get("game_name") is None:
        data["game_name"] = path.stem

    return from_dict(data_class=TableConfig, data=data)


def _resolve_table_path(path: str | Path) -> Path:
    """Resolve a table config path or name to an actual file path.

    If path exists as-is, use it. Otherwise, treat it as a name and look
    in the bundled tables/ directory relative to this source file.
    """
    p = Path(path)
    if p.exists():
        return p

    # Try as a name in the bundled tables directory
    tables_dir = Path(__file__).parent.parent.parent / "tables"
    name = p.stem if p.suffix else str(p)
    bundled = tables_dir / f"{name}.toml"
    if bundled.exists():
        return bundled

    raise FileNotFoundError(f"Table config not found: {path}")
