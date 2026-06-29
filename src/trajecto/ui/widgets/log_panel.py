"""Manöver-Log – zeigt die protokollierten Eintraege eines Moduls.

Generisch: liest ``module.log_entries()`` (Liste von Zeilen). Die GUI bleibt
modul-unabhaengig.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class LogPanel(QWidget):
    """Schreibgeschuetztes Textfeld fuer Log-/Manoeverzeilen (Monospace)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas"))
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

    def set_entries(self, entries: list[str]) -> None:
        self._text.setPlainText("\n".join(entries) if entries else "(keine Eintraege)")
        # Ans Ende scrollen (neuestes Manoever sichtbar).
        bar = self._text.verticalScrollBar()
        bar.setValue(bar.maximum())
