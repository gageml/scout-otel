"""Test import performance."""

import subprocess
import sys


def test_import_time():
    """Import scout_otel in under 200ms."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import time
start = time.perf_counter()
import scout_otel
end = time.perf_counter()
print(f"{(end - start) * 1000:.1f}")
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import_time_ms = float(result.stdout.strip())
    assert import_time_ms < 200, f"Import took {import_time_ms:.1f}ms, expected < 200ms"
