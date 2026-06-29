"""Anzeigeoptionen – generische Checkboxen aus den Plot-Optionen eines Moduls.

Die GUI bleibt datengetrieben: Sie liest ``module.plot_options()`` und erzeugt
je Option eine Checkbox. Keine modul-spezifischen Spezial-Checkboxen.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from trajecto.modules.base import Module, PlotOption


class PlotOptionsPanel(QWidget):
    """Checkbox-Leiste fuer die ein-/ausblendbaren Plot-Elemente eines Moduls."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._checks: dict[str, QCheckBox] = {}
        self._options: dict[str, PlotOption] = {}

    def set_module(self, module: Module) -> None:
        """Baue die Checkboxen fuer die Plot-Optionen des Moduls neu auf."""
        self._clear()
        for option in module.plot_options():
            box = QCheckBox(option.label)
            box.setChecked(option.default)
            box.toggled.connect(self.changed)
            self._layout.addWidget(box)
            self._checks[option.name] = box
            self._options[option.name] = option

    def values(self) -> dict[str, bool]:
        """Aktueller Zustand aller Optionen als ``name -> bool``."""
        return {name: box.isChecked() for name, box in self._checks.items()}

    def apply_visibility(self, param_values: dict[str, Any]) -> None:
        """Zeige/verberge Optionen generisch anhand ihrer ``visible_when``-Bedingung."""
        for name, box in self._checks.items():
            condition = self._options[name].visible_when
            box.setVisible(True if condition is None else bool(condition(param_values)))

    def has_options(self) -> bool:
        return bool(self._checks)

    def _clear(self) -> None:
        self._checks.clear()
        self._options.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
