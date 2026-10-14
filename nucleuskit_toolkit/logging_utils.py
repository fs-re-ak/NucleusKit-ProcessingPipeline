"""Console logging helpers used across the pipeline (GUI-safe, no ANSI when not a TTY)."""

from __future__ import annotations

import logging
import sys

_LOGGER = logging.getLogger("nucleuskit_toolkit")
_CORE_LOGGER = logging.getLogger("reak_core")  # processing messages of re-ak-core


class _AnsiAwareHandler(logging.StreamHandler):
    """Write to the current ``sys.stdout``.

    The stream is resolved on every record so a GUI worker can redirect
    stdout into the on-screen log after this handler was created.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = sys.stdout
            if stream is None:
                return
            self.stream = stream
            if hasattr(stream, "isatty") and stream.isatty():
                if record.levelno >= logging.ERROR:
                    msg = f"\033[91m{msg}\033[0m"
                elif record.levelno >= logging.WARNING:
                    msg = f"\033[93m{msg}\033[0m"
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler to the toolkit and re-ak-core loggers if none exist (idempotent)."""
    for logger in (_LOGGER, _CORE_LOGGER):
        if logger.handlers:
            logger.setLevel(level)
            continue
        handler = _AnsiAwareHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


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
