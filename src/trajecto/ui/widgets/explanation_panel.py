"""Erklaerbereich – zeigt den didaktischen Text des aktiven Moduls."""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class ExplanationPanel(QWidget):
    """Schreibgeschuetztes Textfeld fuer die Modul-Erklaerung."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

    def set_text(self, text: str) -> None:
        self._text.setPlainText(text)
