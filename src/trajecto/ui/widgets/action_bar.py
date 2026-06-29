"""Aktions-Buttons – generisch aus ``module.actions()``.

Die GUI bleibt datengetrieben: Sie erzeugt je Action einen Button und meldet
Klicks ueber das Signal ``triggered(name)``. Aktiviert-Zustand und Wirkung
bestimmt das Modul; hier wird nichts modul-spezifisch hartkodiert.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QWidget

from trajecto.modules.base import Action, Module

#: Buttons werden in so vielen Spalten angeordnet (zweizeilige Gruppe).
_COLUMNS = 2


class ActionBar(QWidget):
    """Knopfleiste fuer die Aktionen eines Moduls (z. B. Δv ausfuehren).

    Die Buttons werden in einem Raster mit zwei Spalten angeordnet, damit sie
    bei Standardfensterbreite nicht abgeschnitten werden.
    """

    triggered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._buttons: dict[str, QPushButton] = {}
        self._actions: dict[str, Action] = {}

    def set_module(self, module: Module) -> None:
        """Baue die Buttons fuer die Aktionen des Moduls neu auf (2-Spalten-Raster)."""
        self._clear()
        for index, action in enumerate(module.actions()):
            button = QPushButton(action.label)
            if action.tooltip:
                button.setToolTip(action.tooltip)
            button.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _=False, n=action.name: self.triggered.emit(n))
            self._layout.addWidget(button, index // _COLUMNS, index % _COLUMNS)
            self._buttons[action.name] = button
            self._actions[action.name] = action
        self.setVisible(bool(self._buttons))

    def has_actions(self) -> bool:
        return bool(self._buttons)

    def apply_visibility(self, param_values: dict[str, Any]) -> None:
        """Zeige/verberge Buttons anhand ihrer ``visible_when``-Bedingung."""
        any_visible = False
        for name, button in self._buttons.items():
            condition = self._actions[name].visible_when
            visible = True if condition is None else bool(condition(param_values))
            button.setVisible(visible)
            any_visible = any_visible or visible
        self.setVisible(any_visible)

    def update_enabled(self, module: Module, param_values: dict[str, Any]) -> None:
        """Setze den Aktiviert-Zustand der Buttons ueber das Modul."""
        for name, button in self._buttons.items():
            button.setEnabled(module.is_action_enabled(name, param_values))

    def _clear(self) -> None:
        self._buttons.clear()
        self._actions.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
