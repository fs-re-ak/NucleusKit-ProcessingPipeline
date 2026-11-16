"""Shared fixtures: one QApplication for the whole run, shut down before the interpreter exits.

Left to Python's finalization, the QApplication and the widgets the tests created (pages whose workers live in
finished QThreads, plots) are destroyed in no fixed order, possibly after Qt's own statics. Qt then sometimes
aborts after every test has passed ("QThreadStorage: entry N destroyed before end of thread", exit code 3), which
makes the release script stop. Tearing everything down here, while Qt is intact, keeps the exit code clean.
"""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(scope="session")
def qapp():
    """The single QApplication of the test run; tests reach it through their module's ``app`` fixture."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication([])
    yield instance
    _shutdown_qt(instance)


def _shutdown_qt(app) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import QThread, QThreadPool, QTimer
    from PySide6.QtWidgets import QApplication

    # A short event loop emits aboutToQuit, as when the app closes: pages stop their own threads on it.
    QTimer.singleShot(0, app.quit)
    app.exec()

    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    gc.collect()
    for obj in gc.get_objects():
        if isinstance(obj, QThread) and obj.isRunning():
            obj.quit()
            obj.wait(2000)
    QThreadPool.globalInstance().waitForDone(2000)

    # Run the deferred deletions, then destroy the QApplication itself while Qt is still intact.
    QTimer.singleShot(0, app.quit)
    app.exec()
    gc.collect()
    app.shutdown()
