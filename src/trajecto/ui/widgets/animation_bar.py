"""Steuerzeile fuer die zeitbasierte Animation (Play/Pause, Reset, Slider, Tempo).

Das Widget ist rein passiv: es haelt die Bedienelemente und gibt sie nach aussen
frei. Die Verdrahtung mit dem animierbaren Parameter und dem Timer geschieht im
``MainWindow`` – die GUI bleibt damit generisch (kein Modul-Wissen hier).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

#: Aufloesung des Sliders (interne Schritte 0..SLIDER_STEPS) -> tau in [0, 1].
SLIDER_STEPS = 1000

#: Tempo-Stufen: voller Durchlauf (Umlauf/Transfer) in so vielen Sekunden.
SPEED_SWEEP_SECONDS: dict[str, float] = {
    "langsam": 12.0,
    "normal": 6.0,
    "schnell": 3.0,
}
#: Default-Tempo (entspricht dem bisherigen Verhalten: 6 s je Durchlauf).
DEFAULT_SPEED = "normal"


class AnimationBar(QWidget):
    """Kompakte Steuerzeile: Play/Pause, Reset, Zeitposition-Slider, Tempo."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.play_button = QPushButton("Abspielen")
        self.play_button.setCheckable(True)
        self.reset_button = QPushButton("Reset")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip("Zeitposition τ (0..1)")

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(list(SPEED_SWEEP_SECONDS.keys()))
        self.speed_combo.setCurrentText(DEFAULT_SPEED)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.play_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(QLabel("τ"))
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(QLabel("Tempo"))
        layout.addWidget(self.speed_combo)

    def sweep_seconds(self) -> float:
        """Dauer eines vollen Durchlaufs (Umlauf/Transfer) in Sekunden."""
        return SPEED_SWEEP_SECONDS.get(self.speed_combo.currentText(), 6.0)

    def set_controls_enabled(self, enabled: bool) -> None:
        """Alle Bedienelemente aktivieren/deaktivieren (z. B. ohne tau-Parameter)."""
        self.play_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.speed_combo.setEnabled(enabled)
