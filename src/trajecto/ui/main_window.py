"""Hauptfenster von Trajecto – Layout und Verdrahtung der Bereiche.

Das Fenster ist datengetrieben: Es kennt nur das Modul-Interface, baut den
Parameterbereich aus dem aktiven Modul auf und ruft bei jeder Parameter-
aenderung ``compute()`` und ``plot()`` auf.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from trajecto.modules.base import Module
from trajecto.modules.registry import available_modules
from trajecto.ui.widgets.explanation_panel import ExplanationPanel
from trajecto.ui.widgets.parameter_panel import ParameterPanel
from trajecto.ui.widgets.plot_panel import PlotPanel
from trajecto.ui.widgets.result_panel import ResultPanel


class MainWindow(QMainWindow):
    """Trajecto-Hauptfenster mit Navigation, Plot und Seitenbereich."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajecto")
        self.resize(1100, 700)

        self._modules: list[Module] = available_modules()
        self._current: Module | None = None

        self._module_list = self._build_module_list()
        self._plot = PlotPanel()
        self._parameters = ParameterPanel()
        self._results = ResultPanel()
        self._explanation = ExplanationPanel()

        self.setCentralWidget(self._build_layout())

        self._parameters.changed.connect(self._recompute)
        self._module_list.currentRowChanged.connect(self._on_module_changed)

        if self._modules:
            self._module_list.setCurrentRow(0)

    # -- Aufbau ---------------------------------------------------------------

    def _build_module_list(self) -> QListWidget:
        widget = QListWidget()
        for module in self._modules:
            item = QListWidgetItem(module.title)
            item.setToolTip(module.subtitle)
            widget.addItem(item)
        widget.setMaximumWidth(240)
        return widget

    def _build_layout(self) -> QWidget:
        # Linke Navigation in einer beschrifteten Box.
        nav_box = QGroupBox("Module")
        nav_layout = QVBoxLayout(nav_box)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        nav_layout.addWidget(self._module_list)

        # Rechter Seitenbereich: Parameter, Ergebnisse, Erklaerung.
        side = QSplitter(Qt.Orientation.Vertical)
        side.addWidget(self._wrap("Parameter", self._parameters))
        side.addWidget(self._wrap("Ergebnisse", self._results))
        side.addWidget(self._wrap("Erklaerung", self._explanation))
        side.setStretchFactor(0, 0)
        side.setStretchFactor(1, 0)
        side.setStretchFactor(2, 1)
        side.setMinimumWidth(320)

        plot_box = self._wrap("Visualisierung", self._plot)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(nav_box)
        main.addWidget(plot_box)
        main.addWidget(side)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setStretchFactor(2, 0)
        main.setSizes([220, 560, 340])
        return main

    @staticmethod
    def _wrap(title: str, inner: QWidget) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(inner)
        return box

    # -- Reaktion -------------------------------------------------------------

    def _on_module_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._modules):
            return
        self._current = self._modules[row]
        self._parameters.set_module(self._current)
        self._explanation.set_text(self._current.explanation())
        self._recompute()

    def _recompute(self) -> None:
        if self._current is None:
            return
        values = self._parameters.values()
        try:
            result = self._current.compute(values)
        except ValueError as exc:
            self._results.show_error(str(exc))
            self._plot.show_placeholder()
            return

        self._results.show_result(result)
        self._current.plot(self._plot.figure, values, result)
        self._plot.redraw()
