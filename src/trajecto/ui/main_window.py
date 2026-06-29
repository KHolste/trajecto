"""Hauptfenster von Trajecto – Layout und Verdrahtung der Bereiche.

Das Fenster ist datengetrieben: Es kennt nur das Modul-Interface, baut den
Parameterbereich aus dem aktiven Modul auf und ruft bei jeder Parameter-
aenderung ``compute()`` und ``plot()`` auf.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trajecto.core import applog
from trajecto.modules.base import Module
from trajecto.modules.registry import available_modules
from trajecto.ui.widgets.action_bar import ActionBar
from trajecto.ui.widgets.animation_bar import SLIDER_STEPS, AnimationBar
from trajecto.ui.widgets.explanation_panel import ExplanationPanel
from trajecto.ui.widgets.log_panel import LogPanel
from trajecto.ui.widgets.parameter_panel import ParameterPanel
from trajecto.ui.widgets.plot_options_panel import PlotOptionsPanel
from trajecto.ui.widgets.plot_panel import PlotPanel
from trajecto.ui.widgets.result_panel import ResultPanel

#: Bildrate der Animation (Tick-Intervall in ms). Die Dauer eines vollen
#: Durchlaufs (sweep_seconds) liefert die Steuerzeile (Tempo-Auswahl).
_ANIM_INTERVAL_MS = 50


class MainWindow(QMainWindow):
    """Trajecto-Hauptfenster mit Navigation, Plot und Seitenbereich."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trajecto")
        self.resize(1100, 700)

        self._modules: list[Module] = available_modules()
        self._current: Module | None = None
        # Letzter Rechenstand, damit reine Anzeige-Aenderungen ohne Neurechnung
        # nur neu gezeichnet werden koennen.
        self._last_values: dict | None = None
        self._last_result = None

        self._module_list = self._build_module_list()
        self._plot = PlotPanel()
        self._parameters = ParameterPanel()
        self._actions = ActionBar()
        self._plot_options = PlotOptionsPanel()
        self._results = ResultPanel()
        self._explanation = ExplanationPanel()
        self._log = LogPanel()

        # Steuerzeile der Animation; Play-Button bleibt unter altem Namen erreichbar.
        self._anim_bar = AnimationBar()
        self._play_button = self._anim_bar.play_button
        # Aktuell vom Slider/Timer gesteuerte tau-Spinbox (oder None).
        self._anim_editor = None
        # Verhindert Rueckkopplung beim wechselseitigen Sync Slider <-> Spinbox.
        self._syncing = False

        # Timer treibt die Animation: er faehrt den animierbaren Parameter
        # (die normierte Zeitposition tau) ueber seinen Wertebereich. Das Setzen
        # des Werts loest ueber das ParameterPanel die normale Neuberechnung aus.
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(_ANIM_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._advance_animation)

        self.setCentralWidget(self._build_layout())

        self._parameters.changed.connect(self._recompute)
        self._plot_options.changed.connect(self._replot)
        self._actions.triggered.connect(self._on_action)
        self._module_list.currentRowChanged.connect(self._on_module_changed)
        self._play_button.toggled.connect(self._on_play_toggled)
        self._anim_bar.reset_button.clicked.connect(self._on_reset)
        self._anim_bar.slider.valueChanged.connect(self._on_slider_changed)

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

        # Rechter Seitenbereich als Tabs: Parameter, Ergebnisse, Erklaerung.
        # Parameter-Tab: Aktionen (Manoever), Eingaben und ein einklappbarer
        # Bereich mit den (datengetriebenen) Anzeigeoptionen – standardmaessig
        # eingeklappt. Alles in einem Scrollbereich, damit nichts abgeschnitten wird.
        self._options_box = QGroupBox("Anzeigeoptionen")
        self._options_box.setCheckable(True)
        ob_layout = QVBoxLayout(self._options_box)
        ob_layout.setContentsMargins(6, 6, 6, 6)
        ob_layout.addWidget(self._plot_options)
        self._options_box.toggled.connect(self._plot_options.setVisible)
        self._options_box.setChecked(False)  # initial eingeklappt

        self._actions_box = QGroupBox("Manöver")
        ab_layout = QVBoxLayout(self._actions_box)
        ab_layout.setContentsMargins(6, 6, 6, 6)
        ab_layout.addWidget(self._actions)

        param_inner = QWidget()
        pt_layout = QVBoxLayout(param_inner)
        pt_layout.setContentsMargins(6, 6, 6, 6)
        pt_layout.addWidget(self._actions_box)
        pt_layout.addWidget(self._parameters)
        pt_layout.addWidget(self._options_box)
        pt_layout.addStretch(1)

        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setWidget(param_inner)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._param_scroll, "Parameter")
        self._tabs.addTab(self._results, "Ergebnisse")
        self._tabs.addTab(self._explanation, "Erklaerung")
        self._tabs.setMinimumWidth(360)
        # Der Manoever-Log-Tab wird nur fuer Module mit Log eingeblendet.
        self._log_tab_index = -1
        side = self._tabs

        # Plotbereich mit Steuerzeile darunter.
        plot_box = QGroupBox("Visualisierung")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addWidget(self._plot, stretch=1)
        plot_layout.addWidget(self._anim_bar)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(nav_box)
        main.addWidget(plot_box)
        main.addWidget(side)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setStretchFactor(2, 0)
        main.setSizes([220, 560, 340])
        return main

    # -- Reaktion -------------------------------------------------------------

    def _on_module_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._modules):
            return
        self._stop_animation()
        self._current = self._modules[row]
        # Module mit internem Zustand (z. B. Hohmann-Simulation) zuruecksetzen.
        self._current.on_activated()
        self._parameters.set_module(self._current)
        self._actions.set_module(self._current)
        self._actions_box.setVisible(self._actions.has_actions())
        self._plot_options.set_module(self._current)
        self._explanation.set_text(self._current.explanation())
        self._update_log_tab()
        self._bind_animation_controls()
        self._recompute()

    def _update_log_tab(self) -> None:
        """Manoever-Log-Tab je nach Modul ein-/ausblenden."""
        wants_log = self._current is not None and self._current.has_log()
        has_tab = self._log_tab_index >= 0
        if wants_log and not has_tab:
            self._log_tab_index = self._tabs.addTab(self._log, "Manöver-Log")
        elif not wants_log and has_tab:
            self._tabs.removeTab(self._log_tab_index)
            self._log_tab_index = -1

    # -- Animation ------------------------------------------------------------

    def _bind_animation_controls(self) -> None:
        """Steuerzeile generisch an den animierbaren Parameter des Moduls binden."""
        # Vorherige Spinbox-Verbindung loesen.
        if self._anim_editor is not None:
            try:
                self._anim_editor.valueChanged.disconnect(self._on_anim_spinbox_changed)
            except (TypeError, RuntimeError):
                pass
            self._anim_editor = None

        target = self._parameters.animatable_editor()
        has_anim = target is not None
        clock = self._current is not None and self._current.is_clock_driven()
        # Steuerung aktiv bei tau-Parameter ODER durchlaufender Missionsuhr.
        self._anim_bar.set_controls_enabled(has_anim or clock)
        # Slider nur fuer tau-Parameter sinnvoll (Missionsuhr ist unbegrenzt).
        self._anim_bar.slider.setEnabled(has_anim)
        if has_anim:
            _param, editor = target
            self._anim_editor = editor
            editor.valueChanged.connect(self._on_anim_spinbox_changed)
            self._sync_slider_from_spinbox()

    def _is_clock(self) -> bool:
        return self._current is not None and self._current.is_clock_driven()

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and (self._anim_editor is not None or self._is_clock()):
            self._play_button.setText("Pause")
            self._anim_timer.start()
        else:
            self._play_button.setText("Abspielen")
            self._anim_timer.stop()

    def _stop_animation(self) -> None:
        self._anim_timer.stop()
        if self._play_button.isChecked():
            # loest _on_play_toggled(False) aus -> Timer-Stop + Text zuruecksetzen.
            self._play_button.setChecked(False)
        self._play_button.setText("Abspielen")

    def _on_reset(self) -> None:
        """Reset: Animation stoppen und tau auf 0 (Bereichsanfang) setzen.

        Bewusst die robustere Variante – Stoppen vermeidet ein sofortiges
        Weiterlaufen direkt nach dem Zuruecksetzen.
        """
        self._stop_animation()
        if self._is_clock():
            # Taktmodul: Missionsuhr (und Bahn/Log) zuruecksetzen.
            self._current.reset_clock()
            self._recompute()
            return
        if self._anim_editor is None:
            return
        self._anim_editor.setValue(self._anim_editor.minimum())

    def _anim_step(self) -> float:
        """Schrittweite pro Timer-Tick im Wertebereich des Parameters."""
        if self._anim_editor is None:
            return 0.0
        span = self._anim_editor.maximum() - self._anim_editor.minimum()
        sweep_seconds = self._anim_bar.sweep_seconds()
        if span <= 0.0 or sweep_seconds <= 0.0:
            return 0.0
        return span * (_ANIM_INTERVAL_MS / 1000.0) / sweep_seconds

    def _advance_animation(self) -> None:
        if self._is_clock():
            # Durchlaufende Missionsuhr: pro Tick einen Bruchteil einer Bahn
            # (eine volle Bahn dauert sweep_seconds Wanduhrzeit).
            sweep = self._anim_bar.sweep_seconds()
            orbit_fraction = (_ANIM_INTERVAL_MS / 1000.0) / sweep if sweep > 0 else 0.0
            self._current.advance_clock(orbit_fraction)
            self._recompute()
            return
        if self._anim_editor is None:
            self._stop_animation()
            return
        low, high = self._anim_editor.minimum(), self._anim_editor.maximum()
        span = high - low
        if span <= 0.0:
            return
        # Umlaufend: am Ende (tau = 1) wieder von vorn (tau = 0) beginnen.
        new_value = low + ((self._anim_editor.value() - low + self._anim_step()) % span)
        self._anim_editor.setValue(new_value)  # loest changed -> _recompute aus

    # -- Slider <-> Spinbox-Synchronisation -----------------------------------

    def _on_slider_changed(self, slider_value: int) -> None:
        if self._syncing or self._anim_editor is None:
            return
        low, high = self._anim_editor.minimum(), self._anim_editor.maximum()
        value = low + (high - low) * slider_value / SLIDER_STEPS
        self._syncing = True
        try:
            self._anim_editor.setValue(value)  # loest changed -> _recompute aus
        finally:
            self._syncing = False

    def _on_anim_spinbox_changed(self, _value: float) -> None:
        if self._syncing:
            return
        self._sync_slider_from_spinbox()

    def _sync_slider_from_spinbox(self) -> None:
        if self._anim_editor is None:
            return
        low, high = self._anim_editor.minimum(), self._anim_editor.maximum()
        span = high - low
        position = 0
        if span > 0.0:
            position = round((self._anim_editor.value() - low) / span * SLIDER_STEPS)
        self._syncing = True
        try:
            self._anim_bar.slider.setValue(int(position))
        finally:
            self._syncing = False

    def _on_action(self, name: str) -> None:
        """Generische Behandlung eines Aktions-Buttons (Modul fuehrt sie aus).

        Eine laufende Animation laeuft nach der Aktion weiter; nur wenn die
        Aktion ``stop_animation`` meldet (z. B. Reset), wird sie gestoppt. So
        bleibt der Play-Zustand bei Δv₁/Δv₂ erhalten.
        """
        if self._current is None:
            return
        values = self._parameters.values()
        try:
            result = self._current.perform_action(name, values)
        except Exception:
            applog.logger.exception("Aktion '%s' fehlgeschlagen", name)
            self._results.show_error(f"Aktion '{name}' fehlgeschlagen (siehe Log).")
            return
        applog.logger.info("Aktion '%s' ausgefuehrt (reset_tau=%s, stop=%s)",
                           name, result.reset_tau, result.stop_animation)
        for param_name, value_si in result.set_params_si.items():
            self._parameters.set_value_si(param_name, value_si)
        if result.reset_tau and self._anim_editor is not None:
            # tau fuer die neue Bahn auf 0 setzen (Play-Zustand bleibt erhalten).
            self._anim_editor.setValue(self._anim_editor.minimum())
        if result.stop_animation:
            self._stop_animation()
        self._recompute()

    def _recompute(self) -> None:
        if self._current is None:
            return
        values = self._parameters.values()
        # Anzeigeoptionen/Aktionen koennen von Parameterwerten abhaengen (z. B. Modus).
        self._plot_options.apply_visibility(values)
        self._actions.apply_visibility(values)
        self._actions.update_enabled(self._current, values)
        try:
            result = self._current.compute(values)
        except ValueError as exc:
            self._last_values = None
            self._last_result = None
            self._results.show_error(str(exc))
            self._plot.show_placeholder()
            return
        except Exception:
            applog.logger.exception("compute() fehlgeschlagen")
            self._last_values = None
            self._last_result = None
            self._results.show_error("Interner Fehler bei der Berechnung (siehe Log).")
            self._plot.show_placeholder()
            return

        self._last_values = values
        self._last_result = result
        self._results.show_result(result)
        if self._current.has_log():
            self._log.set_entries(self._current.log_entries())
        self._render_plot()

    def _replot(self) -> None:
        """Nur neu zeichnen (z. B. nach Umschalten einer Anzeigeoption)."""
        if self._last_result is None:
            return
        self._render_plot()

    def _render_plot(self) -> None:
        if self._current is None or self._last_result is None:
            return
        try:
            self._current.plot(
                self._plot.figure,
                self._last_values,
                self._last_result,
                self._plot_options.values(),
            )
            self._plot.redraw()
        except Exception:
            applog.logger.exception("plot() fehlgeschlagen")
            self._plot.show_placeholder()
