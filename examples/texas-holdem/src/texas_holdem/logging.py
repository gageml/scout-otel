"""Logging configuration for Texas Hold'em demo.

Configures Python standard logging to write to both file and console.
- INFO: Game status (what we currently print)
- DEBUG: Flow tracing (entering functions, waiting for responses, etc.)
"""

import logging
import sys
from pathlib import Path

# Module logger - other modules import this
logger = logging.getLogger("texas_holdem")


class OmitTypeFilter(logging.Filter):
    """Filter out 'Invalid type Omit' warnings from OTEL.

    PydanticAI passes Omit sentinel values for unset parameters (temperature,
    top_p) which OTEL's attribute validation warns about. These are harmless.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid type Omit" not in record.getMessage()


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure logging to file and console.

    Args:
        log_file: Path to log file. If None, only console output.
        level: Logging level (default INFO).
    """
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Format: timestamp - level - message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler - always enabled
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler - if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress 'Invalid type Omit' warnings from OTEL attributes
    logging.getLogger("opentelemetry.attributes").addFilter(OmitTypeFilter())
