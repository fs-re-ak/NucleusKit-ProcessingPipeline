"""Pipeline log lines follow the current stdout, including a redirected GUI log."""

from __future__ import annotations

import io
import sys

from nucleuskit_toolkit.logging_utils import configure_logging, printInfo


def test_printinfo_follows_redirected_stdout() -> None:
    configure_logging()
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        printInfo("hello-screen")
    finally:
        sys.stdout = old
    assert "hello-screen" in buf.getvalue()
