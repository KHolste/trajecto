"""Einstiegspunkt der Trajecto-Desktopanwendung."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from trajecto.core import applog
from trajecto.ui.main_window import MainWindow


def _install_crash_log() -> None:
    """Schreibe unbehandelte Ausnahmen in die Logdatei (Crash-Log)."""
    previous = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        applog.logger.critical(
            "Unbehandelte Ausnahme", exc_info=(exc_type, exc_value, exc_tb)
        )
        previous(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def main() -> int:
    """Starte die Qt-Anwendung und zeige das Hauptfenster."""
    _install_crash_log()
    applog.logger.info("Trajecto startet (Log: %s)", applog.log_file())
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Trajecto")
    app.setApplicationDisplayName("Trajecto")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
