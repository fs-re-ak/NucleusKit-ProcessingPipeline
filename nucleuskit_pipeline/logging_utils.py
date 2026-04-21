"""Console logging helpers used across the pipeline (GUI-safe, no ANSI when not a TTY)."""

from __future__ import annotations

import logging
import sys

_LOGGER = logging.getLogger("nucleuskit_pipeline")


class _AnsiAwareHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            if stream and hasattr(stream, "isatty") and stream.isatty():
                if record.levelno >= logging.ERROR:
                    msg = f"\033[91m{msg}\033[0m"
                elif record.levelno >= logging.WARNING:
                    msg = f"\033[93m{msg}\033[0m"
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler if none exist (idempotent)."""
    if _LOGGER.handlers:
        _LOGGER.setLevel(level)
        return
    handler = _AnsiAwareHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)
    _LOGGER.propagate = False


def printInfo(message: object) -> None:
    configure_logging()
    _LOGGER.info("%s", message)


def printWarning(message: object) -> None:
    configure_logging()
    _LOGGER.warning("%s", message)


def printError(message: object) -> None:
    configure_logging()
    _LOGGER.error("%s", message)


class bcolors:
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
