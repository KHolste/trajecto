"""GUI-Tests fuer die Animationssteuerung (Slider, Reset, Play/Pause, Tempo).

Laeuft headless ueber das Qt-Offscreen-Plugin. Es wird kein Event-Loop
gestartet; die Logik (Slider-Sync, Timer-Zustand, Schrittweite) wird direkt
geprueft.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pytest  # noqa: E402

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from trajecto.modules.base import (  # noqa: E402
    ChoiceParameter,
    Module,
    ModuleResult,
)
from trajecto.ui.main_window import MainWindow  # noqa: E402
from trajecto.ui.widgets.animation_bar import AnimationBar  # noqa: E402
from trajecto.ui.widgets.parameter_panel import ParameterPanel  # noqa: E402

# Modulreihenfolge in der Registry: 0 Kreisbahn, 1 Kepler, 2 Hohmann, 3 Sandbox.
KEPLER_ROW = 1
HOHMANN_ROW = 2
SANDBOX_ROW = 3


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def window(app):
    win = MainWindow()
    win._module_list.setCurrentRow(KEPLER_ROW)
    return win


def test_sidebar_has_parameter_result_explanation_tabs(window) -> None:
    titles = [window._tabs.tabText(i) for i in range(window._tabs.count())]
    assert titles == ["Parameter", "Ergebnisse", "Erklaerung"]


def test_plot_options_are_data_driven(window) -> None:
    # Die Checkboxen entsprechen genau module.plot_options() (datengetrieben).
    win_options = set(window._plot_options.values().keys())
    module_options = {o.name for o in window._current.plot_options()}
    assert win_options == module_options


# --- Modusabhaengige Sichtbarkeit im Hohmann-Modul --------------------------

from trajecto.modules.hohmann_transfer import (  # noqa: E402
    MODE_EXPERIMENT,
    MODE_IDEAL,
    PHASE_AFTER_DV1,
    PHASE_PARKING,
)


def _mode_combo(window):
    return window._parameters._editors["mode"][1]


def test_dv1_hidden_in_ideal_mode(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_IDEAL)
    dv1 = window._parameters._editors["dv1_exp"][1]
    assert dv1.isHidden() is True


def test_dv1_visible_in_experiment_mode(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    dv1 = window._parameters._editors["dv1_exp"][1]
    assert dv1.isHidden() is False


def test_mode_change_updates_parameter_panel(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    combo = _mode_combo(window)
    dv1 = window._parameters._editors["dv1_exp"][1]
    combo.setCurrentText(MODE_IDEAL)
    assert dv1.isHidden() is True
    combo.setCurrentText(MODE_EXPERIMENT)
    assert dv1.isHidden() is False
    combo.setCurrentText(MODE_IDEAL)
    assert dv1.isHidden() is True


def test_experiment_plot_options_hidden_in_ideal_mode(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_IDEAL)
    checks = window._plot_options._checks
    assert checks["show_orbit_after_dv1"].isHidden() is True
    assert checks["show_ideal_orbit"].isHidden() is True
    # Allgemeine Optionen bleiben sichtbar.
    assert checks["show_radius"].isHidden() is False


def test_experiment_plot_options_visible_in_experiment_mode(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    checks = window._plot_options._checks
    assert checks["show_orbit_after_dv1"].isHidden() is False
    assert checks["show_ideal_orbit"].isHidden() is False


def test_other_modules_keep_all_parameters_visible(window) -> None:
    # Kreisbahn und Kepler haben keine visible_when-Bedingungen.
    for row in (0, 1):
        window._module_list.setCurrentRow(row)
        for _name, (_p, ed) in window._parameters._editors.items():
            assert ed.isHidden() is False


# --- Aktions-Buttons (manoevergetriebener Ablauf) ---------------------------


def test_action_buttons_present_in_experiment_mode(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    names = set(window._actions._buttons.keys())
    assert {"execute_dv1", "execute_dv2", "reset_sim", "set_ideal"} <= names


def test_dv1_button_enabled_dv2_disabled_in_parking(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    buttons = window._actions._buttons
    assert buttons["execute_dv1"].isEnabled() is True
    assert buttons["execute_dv2"].isEnabled() is False


def test_execute_dv1_via_gui_changes_phase(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    module = window._current
    assert module._phase == PHASE_PARKING
    window._on_action("execute_dv1")
    assert module._phase == PHASE_AFTER_DV1
    # Nach Δv₁ ist Δv₁ deaktiviert; Δv₂ ist jederzeit (an beliebiger Position) aktiv.
    assert window._actions._buttons["execute_dv1"].isEnabled() is False
    assert window._actions._buttons["execute_dv2"].isEnabled() is True


def test_reset_button_via_gui_returns_to_parking(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    module = window._current
    window._on_action("execute_dv1")
    assert module._phase == PHASE_AFTER_DV1
    window._on_action("reset_sim")
    assert module._phase == PHASE_PARKING
    # Animation gestoppt, tau = 0.
    assert window._anim_timer.isActive() is False
    assert window._anim_editor.value() == pytest.approx(0.0)


def test_action_buttons_have_tooltips(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    for name in ("execute_dv1", "execute_dv2", "reset_sim"):
        assert window._actions._buttons[name].toolTip()


def test_module_switch_resets_hohmann_simulation(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    module = window._current
    window._on_action("execute_dv1")
    assert module._phase == PHASE_AFTER_DV1
    # Modul wechseln und zurueck -> Simulation ist zurueckgesetzt (Variante A).
    window._module_list.setCurrentRow(0)
    window._module_list.setCurrentRow(HOHMANN_ROW)
    assert module._phase == PHASE_PARKING


def test_phase_status_shown_prominently(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    # Statuszeile sichtbar (nicht versteckt) und zeigt die Startphase.
    assert window._results._status.isHidden() is False
    assert "Startkreisbahn" in window._results._status.text()


# --- Animation laeuft nach Δv₁/Δv₂ weiter, Reset stoppt ---------------------


def test_animation_continues_after_dv1(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    window._play_button.setChecked(True)
    assert window._anim_timer.isActive() is True
    window._on_action("execute_dv1")
    # Animation laeuft direkt auf der neuen Bahn weiter.
    assert window._anim_timer.isActive() is True
    assert window._anim_editor.value() == pytest.approx(0.0)  # tau fuer neue Bahn = 0


def test_animation_continues_after_dv2(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    window._play_button.setChecked(True)
    window._on_action("execute_dv1")
    window._anim_editor.setValue(0.5)  # zum Apsidenpunkt
    window._on_action("execute_dv2")
    assert window._anim_timer.isActive() is True


def test_reset_stops_running_animation(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    window._play_button.setChecked(True)
    window._on_action("execute_dv1")
    assert window._anim_timer.isActive() is True
    window._on_action("reset_sim")
    assert window._anim_timer.isActive() is False  # nur Reset stoppt


# --- Rechter Parameterbereich: Layout / Labels / Optionen -------------------


def test_parameter_tab_is_scrollable(window) -> None:
    from PySide6.QtWidgets import QScrollArea

    tab = window._tabs.widget(0)
    assert isinstance(tab, QScrollArea)
    assert tab.widgetResizable() is True


def test_action_buttons_use_two_column_grid(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    # 4 Aktionen -> 2 Spalten -> 2 Zeilen (nicht in eine enge Zeile gequetscht).
    assert window._actions._layout.columnCount() == 2
    assert window._actions._layout.rowCount() == 2


def test_hohmann_labels_shortened_with_tooltips(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    params = {p.name: p for p in window._current.parameters()}
    assert params["dv1_exp"].label == "Δv₁ exp."
    assert params["dv2_exp"].label == "Δv₂ exp."
    assert params["tau"].label == "τ"
    # Tooltips vorhanden (Parameter + Editor-Widget).
    assert params["dv1_exp"].tooltip
    assert window._parameters._editors["dv1_exp"][1].toolTip()
    assert window._parameters._editors["tau"][1].toolTip()


def test_parameter_groups_have_headers(window) -> None:
    window._module_list.setCurrentRow(HOHMANN_ROW)
    _mode_combo(window).setCurrentText(MODE_EXPERIMENT)
    groups = {g for g, _h in window._parameters._group_headers}
    assert {"Setup", "Manöver", "Animation"} <= groups


def test_display_options_collapsed_by_default(window) -> None:
    # Anzeigeoptionen initial eingeklappt ...
    assert window._options_box.isChecked() is False
    # ... aber weiterhin datengetrieben vorhanden/nutzbar.
    assert set(window._plot_options.values().keys())


# --- Manoever-Sandbox: Missionsuhr + Log-Tab --------------------------------


def test_sandbox_shows_maneuver_log_tab(window) -> None:
    window._module_list.setCurrentRow(SANDBOX_ROW)
    titles = [window._tabs.tabText(i) for i in range(window._tabs.count())]
    assert "Manöver-Log" in titles
    # Bei einem Modul ohne Log verschwindet der Tab wieder.
    window._module_list.setCurrentRow(KEPLER_ROW)
    titles = [window._tabs.tabText(i) for i in range(window._tabs.count())]
    assert "Manöver-Log" not in titles


def test_sandbox_clock_animation_runs_without_tau(window) -> None:
    window._module_list.setCurrentRow(SANDBOX_ROW)
    # Kein tau-Parameter, aber Play ist aktivierbar (Missionsuhr).
    assert window._anim_editor is None
    assert window._play_button.isEnabled() is True
    assert window._anim_bar.slider.isEnabled() is False
    window._play_button.setChecked(True)
    assert window._anim_timer.isActive() is True
    before = window._current._mission_time
    for _ in range(10):
        window._advance_animation()
    assert window._current._mission_time > before


def test_sandbox_dv_via_gui_logs_and_keeps_running(window) -> None:
    window._module_list.setCurrentRow(SANDBOX_ROW)
    window._play_button.setChecked(True)
    for _ in range(5):
        window._advance_animation()
    window._on_action("execute_dv")
    assert len(window._current._maneuvers) == 1
    assert "#1" in window._log._text.toPlainText()
    assert window._anim_timer.isActive() is True  # laeuft weiter


def test_sandbox_reset_clears_clock(window) -> None:
    window._module_list.setCurrentRow(SANDBOX_ROW)
    window._play_button.setChecked(True)
    for _ in range(5):
        window._advance_animation()
    window._on_action("execute_dv")
    window._on_reset()
    assert window._current._mission_time == 0.0
    assert len(window._current._maneuvers) == 0
    assert window._anim_timer.isActive() is False


def test_slider_updates_spinbox(window) -> None:
    editor = window._anim_editor
    window._anim_bar.slider.setValue(500)
    assert editor.value() == pytest.approx(0.5, abs=1e-3)
    window._anim_bar.slider.setValue(0)
    assert editor.value() == pytest.approx(0.0, abs=1e-3)


def test_spinbox_updates_slider(window) -> None:
    window._anim_editor.setValue(0.25)
    assert window._anim_bar.slider.value() == 250
    window._anim_editor.setValue(1.0)
    assert window._anim_bar.slider.value() == 1000


def test_reset_sets_tau_zero_and_stops(window) -> None:
    window._anim_editor.setValue(0.8)
    window._play_button.setChecked(True)
    window._on_reset()
    assert window._anim_editor.value() == pytest.approx(0.0)
    assert window._anim_bar.slider.value() == 0
    assert window._anim_timer.isActive() is False
    assert window._play_button.isChecked() is False


def test_play_pause_toggles_timer(window) -> None:
    assert window._anim_timer.isActive() is False
    window._play_button.setChecked(True)
    assert window._anim_timer.isActive() is True
    assert window._play_button.text() == "Pause"
    window._play_button.setChecked(False)
    assert window._anim_timer.isActive() is False
    assert window._play_button.text() == "Abspielen"


def test_speed_changes_step_size(window) -> None:
    window._anim_bar.speed_combo.setCurrentText("langsam")
    slow = window._anim_step()
    window._anim_bar.speed_combo.setCurrentText("normal")
    normal = window._anim_step()
    window._anim_bar.speed_combo.setCurrentText("schnell")
    fast = window._anim_step()
    assert slow < normal < fast
    # Schneller bedeutet kuerzere Durchlaufzeit -> groessere Schrittweite.
    assert fast == pytest.approx(2.0 * normal, rel=1e-6)


def test_advance_animation_uses_speed(window) -> None:
    window._anim_editor.setValue(0.0)
    window._anim_bar.speed_combo.setCurrentText("normal")
    step = window._anim_step()
    window._advance_animation()
    assert window._anim_editor.value() == pytest.approx(step, abs=1e-3)


def test_animation_wraps_around(window) -> None:
    # Nahe tau = 1 springt die Animation zurueck Richtung 0 (kein Ping-Pong).
    window._anim_bar.speed_combo.setCurrentText("schnell")
    window._anim_editor.setValue(1.0)
    window._advance_animation()
    assert window._anim_editor.value() < 0.5


def test_module_change_stops_animation(window) -> None:
    window._play_button.setChecked(True)
    assert window._anim_timer.isActive() is True
    window._module_list.setCurrentRow(0)  # Kreisbahn
    assert window._anim_timer.isActive() is False
    assert window._play_button.isChecked() is False
    # Neuer animierbarer Parameter ist gebunden, Steuerung aktiv.
    assert window._anim_editor is not None
    assert window._play_button.isEnabled() is True


# --- Modul ohne animierbaren Parameter --------------------------------------


class _StaticModule(Module):
    id = "static_dummy"
    title = "Statisch"

    def parameters(self):
        return [ChoiceParameter(name="body", label="Koerper",
                                choices=("A", "B"), default="A")]

    def compute(self, values):
        return ModuleResult()

    def plot(self, figure, values, result, options=None):
        figure.add_subplot(111)

    def explanation(self):
        return "kein animierbarer Parameter"


def test_module_without_animatable_has_no_editor(app) -> None:
    panel = ParameterPanel()
    panel.set_module(_StaticModule())
    assert panel.animatable_editor() is None


def test_controls_disabled_when_no_animatable(app) -> None:
    bar = AnimationBar()
    bar.set_controls_enabled(False)
    assert bar.play_button.isEnabled() is False
    assert bar.reset_button.isEnabled() is False
    assert bar.slider.isEnabled() is False
    assert bar.speed_combo.isEnabled() is False
