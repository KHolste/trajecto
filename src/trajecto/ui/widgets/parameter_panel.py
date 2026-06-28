"""Parameterbereich – baut Eingaben generisch aus der Modulbeschreibung.

Float-Parameter werden als ``QDoubleSpinBox`` in der jeweiligen Anzeige-Einheit
dargestellt; intern liefert ``values()`` jedoch SI-Werte zurueck. Choice-
Parameter werden zu einer ``QComboBox``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QWidget,
)

from trajecto.core.units import from_si, to_si
from trajecto.modules.base import ChoiceParameter, FloatParameter, Module, Parameter


class ParameterPanel(QWidget):
    """Erzeugt Eingabe-Widgets und meldet Aenderungen ueber ``changed``."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._editors: dict[str, tuple[Parameter, QWidget]] = {}

    def set_module(self, module: Module) -> None:
        """Baue die Eingabemaske fuer das gegebene Modul neu auf."""
        self._clear()
        for param in module.parameters():
            editor = self._build_editor(param)
            self._editors[param.name] = (param, editor)
            self._form.addRow(self._row_label(param), editor)

    def values(self) -> dict[str, Any]:
        """Aktuelle Eingaben: Float-Parameter in SI, Choices als String."""
        result: dict[str, Any] = {}
        for name, (param, editor) in self._editors.items():
            if isinstance(param, FloatParameter):
                assert isinstance(editor, QDoubleSpinBox)
                result[name] = to_si(editor.value(), param.display_unit)
            elif isinstance(param, ChoiceParameter):
                assert isinstance(editor, QComboBox)
                result[name] = editor.currentText()
        return result

    # -- intern ---------------------------------------------------------------

    def _row_label(self, param: Parameter) -> str:
        if isinstance(param, FloatParameter):
            return f"{param.label} [{param.display_unit}]"
        return param.label

    def _build_editor(self, param: Parameter) -> QWidget:
        if isinstance(param, FloatParameter):
            return self._build_float_editor(param)
        if isinstance(param, ChoiceParameter):
            return self._build_choice_editor(param)
        raise TypeError(f"Unbekannter Parametertyp: {type(param)!r}")

    def _build_float_editor(self, param: FloatParameter) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setMinimum(from_si(param.minimum_si, param.display_unit))
        if param.maximum_si is not None:
            box.setMaximum(from_si(param.maximum_si, param.display_unit))
        else:
            box.setMaximum(1.0e12)
        box.setValue(from_si(param.default_si, param.display_unit))
        # Sinnvolle Schrittweite aus dem Wertebereich ableiten; sonst fester Default.
        if param.maximum_si is not None:
            span = from_si(param.maximum_si, param.display_unit) - box.minimum()
            box.setSingleStep(max(span / 100.0, 1e-3))
        box.setKeyboardTracking(False)
        box.valueChanged.connect(self.changed)
        return box

    def _build_choice_editor(self, param: ChoiceParameter) -> QComboBox:
        box = QComboBox()
        box.addItems(list(param.choices))
        box.setCurrentText(param.default)
        box.currentTextChanged.connect(self.changed)
        return box

    def _clear(self) -> None:
        self._editors.clear()
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
