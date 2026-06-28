"""Zentraler Plotbereich auf Basis von Matplotlib."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotPanel(QWidget):
    """Kapselt eine Matplotlib-Figure als Qt-Widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.show_placeholder()

    def show_placeholder(self) -> None:
        """Zeige einen neutralen Hinweis, solange nichts berechnet wurde."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Modul waehlen und Parameter setzen",
            ha="center",
            va="center",
            fontsize="medium",
            color="#888888",
        )
        ax.set_axis_off()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        """Nach externem Zeichnen in ``self.figure`` neu rendern."""
        self.canvas.draw_idle()
