"""Ergebnisbereich – stellt die Kennzahlen einer Berechnung dar."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtGui import QBrush, QColor, QFont

from trajecto.core.units import from_si
from trajecto.modules.base import ModuleResult, ResultItem, ResultSection


class ResultPanel(QWidget):
    """Zeigt ``ModuleResult.items`` als Tabelle plus optionale Hinweise (notes)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Groesse", "Wert", "Einheit"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        # Prominente Statuszeile (z. B. aktuelle Phase).
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        status_font = QFont(self._status.font())
        status_font.setBold(True)
        self._status.setFont(status_font)
        self._status.setStyleSheet(
            "background-color: #dbeafe; color: #1d3557; padding: 4px; border-radius: 3px;"
        )

        self._notes = QLabel(self)
        self._notes.setWordWrap(True)
        self._notes.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._status)
        layout.addWidget(self._table)
        layout.addWidget(self._notes)

    def show_result(self, result: ModuleResult) -> None:
        """Fuelle die Tabelle; Werte werden in die Anzeige-Einheit umgerechnet.

        ``ResultSection``-Eintraege werden als fette, ueber die ganze Breite
        gespannte Gruppenueberschriften dargestellt.
        """
        self._table.clearSpans()
        self._table.setRowCount(len(result.items))
        for row, entry in enumerate(result.items):
            if isinstance(entry, ResultSection):
                header = QTableWidgetItem(entry.title)
                font = QFont(header.font())
                font.setBold(True)
                header.setFont(font)
                header.setBackground(QBrush(QColor(235, 235, 235)))
                self._table.setItem(row, 0, header)
                self._table.setItem(row, 1, QTableWidgetItem(""))
                self._table.setItem(row, 2, QTableWidgetItem(""))
                self._table.setSpan(row, 0, 1, 3)
                continue
            value = from_si(entry.value_si, entry.display_unit)
            self._table.setItem(row, 0, QTableWidgetItem(entry.label))
            self._table.setItem(row, 1, QTableWidgetItem(f"{value:,.3f}"))
            self._table.setItem(row, 2, QTableWidgetItem(entry.display_unit))
        self._set_status(result.status)
        self._set_notes(result.notes)

    def show_error(self, message: str) -> None:
        """Zeige eine Fehlermeldung anstelle von Ergebnissen."""
        self._table.setRowCount(1)
        self._table.setItem(0, 0, QTableWidgetItem("Fehler"))
        self._table.setItem(0, 1, QTableWidgetItem(message))
        self._table.setItem(0, 2, QTableWidgetItem(""))
        self._set_status(None)
        self._set_notes([])

    def _set_status(self, status: str | None) -> None:
        if status:
            self._status.setText(status)
            self._status.setVisible(True)
        else:
            self._status.clear()
            self._status.setVisible(False)

    def _set_notes(self, notes: list[str]) -> None:
        if notes:
            self._notes.setText("\n".join(f"• {n}" for n in notes))
            self._notes.setVisible(True)
        else:
            self._notes.clear()
            self._notes.setVisible(False)
