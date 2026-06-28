"""Ergebnisbereich – stellt die Kennzahlen einer Berechnung dar."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trajecto.core.units import from_si
from trajecto.modules.base import ModuleResult


class ResultPanel(QWidget):
    """Zeigt ``ModuleResult.items`` als einfache Tabelle (Groesse/Wert/Einheit)."""

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def show_result(self, result: ModuleResult) -> None:
        """Fuelle die Tabelle; Werte werden in die Anzeige-Einheit umgerechnet."""
        self._table.setRowCount(len(result.items))
        for row, item in enumerate(result.items):
            value = from_si(item.value_si, item.display_unit)
            self._table.setItem(row, 0, QTableWidgetItem(item.label))
            value_cell = QTableWidgetItem(f"{value:,.3f}")
            self._table.setItem(row, 1, value_cell)
            self._table.setItem(row, 2, QTableWidgetItem(item.display_unit))

    def show_error(self, message: str) -> None:
        """Zeige eine Fehlermeldung anstelle von Ergebnissen."""
        self._table.setRowCount(1)
        self._table.setItem(0, 0, QTableWidgetItem("Fehler"))
        self._table.setItem(0, 1, QTableWidgetItem(message))
        self._table.setItem(0, 2, QTableWidgetItem(""))
