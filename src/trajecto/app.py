"""Einstiegspunkt der Trajecto-Desktopanwendung."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from trajecto.ui.main_window import MainWindow


def main() -> int:
    """Starte die Qt-Anwendung und zeige das Hauptfenster."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Trajecto")
    app.setApplicationDisplayName("Trajecto")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
