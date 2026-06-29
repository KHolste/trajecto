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
        # Labels ueber den Feldern -> schmal-robust, keine abgeschnittenen Felder.
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._editors: dict[str, tuple[Parameter, QWidget]] = {}
        # Gruppenueberschriften: (Gruppenname, Header-Label).
        self._group_headers: list[tuple[str, QLabel]] = []

    def set_module(self, module: Module) -> None:
        """Baue die Eingabemaske fuer das gegebene Modul neu auf (mit Gruppen)."""
        self._clear()
        current_group: str | None = None
        for param in module.parameters():
            group = getattr(param, "group", "")
            if group and group != current_group:
                header = QLabel(group)
                font = header.font()
                font.setBold(True)
                header.setFont(font)
                self._form.addRow(header)
                self._group_headers.append((group, header))
                current_group = group
            editor = self._build_editor(param)
            self._editors[param.name] = (param, editor)
            self._form.addRow(self._row_label(param), editor)
            self._apply_tooltip(param, editor)
        self._apply_visibility()

    def _apply_tooltip(self, param: Parameter, editor: QWidget) -> None:
        tip = getattr(param, "tooltip", "")
        if not tip:
            return
        editor.setToolTip(tip)
        label = self._form.labelForField(editor)
        if label is not None:
            label.setToolTip(tip)

    def _apply_visibility(self) -> None:
        """Zeige/verberge Parameter (und leere Gruppen) generisch."""
        current = self.values()
        visible_groups: set[str] = set()
        for _name, (param, editor) in self._editors.items():
            condition = getattr(param, "visible_when", None)
            visible = True if condition is None else bool(condition(current))
            editor.setVisible(visible)
            label = self._form.labelForField(editor)
            if label is not None:
                label.setVisible(visible)
            if visible:
                visible_groups.add(getattr(param, "group", ""))
        # Gruppenueberschrift nur zeigen, wenn die Gruppe sichtbare Parameter hat.
        for group, header in self._group_headers:
            header.setVisible(group in visible_groups)

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

    def set_value_si(self, name: str, value_si: float) -> None:
        """Setze einen Float-Parameter auf einen SI-Wert (in Anzeige-Einheit)."""
        entry = self._editors.get(name)
        if entry is None:
            return
        param, editor = entry
        if isinstance(param, FloatParameter) and isinstance(editor, QDoubleSpinBox):
            editor.setValue(from_si(value_si, param.display_unit))

    def animatable_editor(self) -> tuple[FloatParameter, QDoubleSpinBox] | None:
        """Erster animierbarer Float-Parameter mit endlichem Maximum, oder None.

        Damit kann die GUI generisch den deklarierten Animationsparameter
        (die Zeitposition tau) ueber seinen Wertebereich fahren – ohne Modul-Sonderfall.
        """
        for param, editor in self._editors.values():
            if (
                isinstance(param, FloatParameter)
                and param.animatable
                and param.maximum_si is not None
            ):
                assert isinstance(editor, QDoubleSpinBox)
                return param, editor
        return None

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
        box.valueChanged.connect(self._on_editor_changed)
        return box

    def _build_choice_editor(self, param: ChoiceParameter) -> QComboBox:
        box = QComboBox()
        box.addItems(list(param.choices))
        box.setCurrentText(param.default)
        box.currentTextChanged.connect(self._on_editor_changed)
        return box

    def _on_editor_changed(self, *_args: object) -> None:
        # Sichtbarkeit kann von Werten abhaengen (z. B. Modus) -> zuerst aktualisieren.
        self._apply_visibility()
        self.changed.emit()

    def _clear(self) -> None:
        self._editors.clear()
        self._group_headers.clear()
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
