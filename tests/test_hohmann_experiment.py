"""Tests fuer den manoevergetriebenen Δv-Experimentablauf im Hohmann-Modul."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from trajecto.core import orbit_determination as od  # noqa: E402
from trajecto.core import orbital_mechanics as om  # noqa: E402
from trajecto.core.bodies import EARTH  # noqa: E402
from trajecto.modules.base import ModuleResult  # noqa: E402
from trajecto.modules.hohmann_transfer import (  # noqa: E402
    MODE_EXPERIMENT,
    PHASE_AFTER_DV1,
    PHASE_AFTER_DV2,
    PHASE_PARKING,
    HohmannTransferModule,
    experiment_state,
)

MU = EARTH.mu
R_LEO = 6_678_000.0
R_GEO = 42_164_000.0
RADIUS_MODE = "Radius vom Mittelpunkt"


def _values(dv1: float = 0.0, dv2: float = 0.0, r1: float = R_LEO, r2: float = R_GEO,
            tau: float = 0.0) -> dict:
    return {"mode": MODE_EXPERIMENT, "body": EARTH.name, "input_mode": RADIUS_MODE,
            "r1": r1, "r2": r2, "dv1_exp": dv1, "dv2_exp": dv2, "tau": tau}


def _map(result: ModuleResult) -> dict[str, float]:
    return {i.label: i.value_si for i in result.items if hasattr(i, "value_si")}


def _ideal() -> tuple[float, float]:
    t = om.hohmann_transfer(MU, R_LEO, R_GEO)
    return t.dv1, t.dv2


# --- Darstellung: aktive Bahn, Legende, Status, Umlaute ---------------------


def _plot_labels(module, values) -> list[str]:
    result = module.compute(values)
    fig = Figure()
    module.plot(fig, values, result)
    return fig.axes[0].get_legend_handles_labels()[1]


def test_active_orbit_labeled_after_dv1() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.3)
    module.perform_action("execute_dv1", values)
    labels = _plot_labels(module, values)
    assert any("aktive Bahn" in lbl and "Δv₁" in lbl for lbl in labels)


def test_active_orbit_labeled_after_dv2() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    apo = {**values, "tau": 0.5}
    module.perform_action("execute_dv2", apo)
    labels = _plot_labels(module, apo)
    assert any("aktive Bahn" in lbl and "Δv₂" in lbl for lbl in labels)


def test_legend_not_overcrowded_by_default() -> None:
    # Standard: nur Bahnen/Koerper in der Legende, keine Vektor-/Pfeileintraege.
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.3)
    module.perform_action("execute_dv1", values)
    labels = _plot_labels(module, values)
    assert len(labels) <= 6
    assert not any("Geschwindigkeit" in lbl for lbl in labels)
    assert not any("Radiusvektor" in lbl for lbl in labels)


def test_status_box_contains_phase_eval_and_budget() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    apo = {**values, "tau": 0.5}
    module.perform_action("execute_dv2", apo)
    status = module.compute(apo).status
    assert "Phase:" in status
    assert "Zielkreisbahn erreicht" in status
    assert "Budget:" in status
    assert "Nächster Schritt:" in status


def test_visible_german_texts_use_umlauts() -> None:
    from trajecto.core.orbit_input import INPUT_MODE_ALTITUDE

    module = HohmannTransferModule()
    labels = [a.label for a in module.actions()]
    assert "Δv₁ ausführen" in labels
    assert "Simulation zurücksetzen" in labels
    assert "über" in INPUT_MODE_ALTITUDE  # "Höhe über Oberfläche"
    params = {p.name: p for p in module.parameters()}
    assert params["body"].label == "Zentralkörper"
    assert params["dv1_exp"].group == "Manöver"


# --- Kontinuitaet beim Impuls (kein Sprung / keine Teleportation) -----------


def test_dv1_at_arbitrary_tau_keeps_position() -> None:
    # Burn an beliebiger Position (tau = 0.3) auf der Startkreisbahn.
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.3)
    pos_before = module.compute(values).data["exp_sc_position"].copy()
    module.perform_action("execute_dv1", values)
    # Direkt nach Δv₁ (Animation bei tau = 0 am Burnpunkt): gleiche Position.
    pos_after = module.compute({**values, "tau": 0.0}).data["exp_sc_position"]
    assert np.allclose(pos_before, pos_after, atol=1.0)


def test_dv1_changes_only_velocity_by_tangential_dv() -> None:
    module = HohmannTransferModule()
    dv1 = _ideal()[0]
    values = _values(dv1=dv1, tau=0.3)
    data = module.compute(values).data
    pos_before = data["exp_sc_position"].copy()
    vel_before = data["exp_sc_velocity"].copy()
    module.perform_action("execute_dv1", values)
    burn = module._burn1
    assert np.allclose(burn["position"], pos_before, atol=1.0)  # Position identisch
    dv_vec = burn["velocity"] - vel_before
    assert float(np.linalg.norm(dv_vec)) == pytest.approx(abs(dv1), rel=1e-6)
    tang = vel_before / np.linalg.norm(vel_before)
    assert np.allclose(dv_vec, dv1 * tang, atol=1e-3)  # tangential


def test_dv1_orbit_passes_through_burn_point() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.3)
    module.perform_action("execute_dv1", values)
    burn = module._burn1
    st0 = od.propagate(burn["mu"], burn["position"], burn["velocity"], 0.0)
    assert np.allclose(st0.position, burn["position"], atol=1.0)


def test_dv1_orbit_not_fixed_to_plus_x() -> None:
    # Burn bei tau = 0.25 (90 Grad) -> Periapsis nicht bei +x.
    module = HohmannTransferModule()
    values = _values(dv1=500.0, tau=0.25)
    module.perform_action("execute_dv1", values)
    burn = module._burn1
    el = od.orbital_elements(burn["mu"], burn["position"], burn["velocity"])
    assert abs(el.arg_periapsis) > 0.5  # deutlich von 0 (+x) verschieden


def test_animation_after_dv1_starts_at_burn_point() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.4)
    module.perform_action("execute_dv1", values)
    sc0 = module.compute({**values, "tau": 0.0}).data["exp_sc_position"]
    assert np.allclose(sc0, module._burn1["position"], atol=1.0)


def test_dv2_no_position_jump() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    apo = {**values, "tau": 0.5}
    data_before = module.compute(apo).data
    pos_before = data_before["exp_sc_position"].copy()
    vel_before = data_before["exp_sc_velocity"].copy()
    module.perform_action("execute_dv2", apo)
    burn2 = module._burn2
    # Position am Δv₂-Punkt identisch (kein Sprung).
    assert np.allclose(burn2["position"], pos_before, atol=1.0)
    pos_after = module.compute({**apo, "tau": 0.0}).data["exp_sc_position"]
    assert np.allclose(pos_before, pos_after, atol=1.0)
    # Nur die Geschwindigkeit aendert sich (um Δv₂).
    dv_vec = burn2["velocity"] - vel_before
    assert float(np.linalg.norm(dv_vec)) == pytest.approx(abs(dv2), rel=1e-6)


# --- Prominente Phasenanzeige (status) --------------------------------------


def test_status_shows_phase_parking() -> None:
    module = HohmannTransferModule()
    result = module.compute(_values())
    assert result.status is not None
    assert "Startkreisbahn" in result.status


def test_status_shows_phase_after_dv1() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0])
    module.perform_action("execute_dv1", values)
    assert "nach Δv₁" in module.compute(values).status


def test_status_shows_phase_after_dv2() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    module.perform_action("execute_dv2", values)
    assert "nach Δv₂" in module.compute(values).status


def test_ideal_mode_has_no_phase_status() -> None:
    module = HohmannTransferModule()
    result = module.compute(
        {"mode": "Idealer Hohmann-Transfer", "body": EARTH.name,
         "input_mode": RADIUS_MODE, "r1": R_LEO, "r2": R_GEO, "tau": 0.0}
    )
    assert result.status is None


# --- Didaktische vs. Core-Klassifikation ------------------------------------


def test_ideal_dv2_didactic_class_is_target_circle() -> None:
    dv1, dv2 = _ideal()
    _m, _vals, result = _after_dv2(dv1, dv2)
    # Nutzerseitig: Zielkreisbahn; nicht "gebundene Ellipse".
    klass_note = next(n for n in result.notes if n.startswith("Aktuelle Bahn:"))
    assert "Zielkreisbahn erreicht" in klass_note or "kreisfoermig" in klass_note
    assert "gebundene Ellipse" not in klass_note


def test_core_classification_unchanged_for_near_circular() -> None:
    # Die physikalische Core-Klassifikation bleibt unveraendert (e > 1e-8 -> Ellipse).
    e_small = od.orbital_elements(
        MU, np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, om.circular_orbit_velocity(MU, 7.0e6) * 1.0005, 0.0]),
    )
    assert e_small.classification == od.CLASS_ELLIPSE
    assert 0.0 < e_small.eccentricity < 5e-3


# --- Tooltips ---------------------------------------------------------------


def test_actions_have_tooltips() -> None:
    module = HohmannTransferModule()
    tips = {a.name: a.tooltip for a in module.actions()}
    assert tips["execute_dv1"]
    assert tips["execute_dv2"]
    assert tips["reset_sim"]


# --- Phasen / Aktionen ------------------------------------------------------


def test_initial_phase_is_parking() -> None:
    module = HohmannTransferModule()
    assert module._phase == PHASE_PARKING


def test_on_activated_resets_simulation() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0])
    module.perform_action("execute_dv1", values)
    assert module._phase == PHASE_AFTER_DV1
    module.on_activated()
    assert module._phase == PHASE_PARKING


def test_button_states_in_after_dv2() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    module.perform_action("execute_dv2", values)
    assert module._phase == PHASE_AFTER_DV2
    assert module.is_action_enabled("execute_dv1", values) is False
    assert module.is_action_enabled("execute_dv2", values) is False
    assert module.is_action_enabled("reset_sim", values) is True


def test_parking_phase_has_no_transfer_orbit_in_plot_data() -> None:
    module = HohmannTransferModule()
    result = module.compute(_values())
    # Vor Δv₁ existiert keine Bahn nach Δv₁/Δv₂.
    assert result.data["exp_dv1_orbit"] is None
    assert result.data["exp_dv2_orbit"] is None
    assert any("Startkreisbahn" in n for n in result.notes)


def test_execute_dv1_makes_orbit_visible() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0])
    module.perform_action("execute_dv1", values)
    assert module._phase == PHASE_AFTER_DV1
    result = module.compute(values)
    assert result.data["exp_dv1_orbit"] is not None


def test_dv2_button_inactive_before_dv1() -> None:
    module = HohmannTransferModule()
    values = _values()
    assert module.is_action_enabled("execute_dv1", values) is True
    assert module.is_action_enabled("execute_dv2", values) is False
    # Δv₂ ausfuehren vor Δv₁ ist wirkungslos.
    module.perform_action("execute_dv2", values)
    assert module._phase == PHASE_PARKING


def test_dv2_enabled_anywhere_after_dv1() -> None:
    # Δv₂ ist nach Δv₁ an beliebiger Position (jedem tau) aktiv, nicht nur am Apsis.
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0])
    module.perform_action("execute_dv1", values)
    for tau in (0.0, 0.2, 0.5, 0.85):
        assert module.is_action_enabled("execute_dv2", {**values, "tau": tau}) is True


def test_dv2_off_apsis_does_not_reach_target_circle() -> None:
    # Δv₂ ausserhalb des Apsidenpunkts -> keine Zielkreisbahn (andere Ellipse).
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    off = {**values, "tau": 0.25}  # nicht am Apsidenpunkt
    module.perform_action("execute_dv2", off)
    assert module._phase == PHASE_AFTER_DV2
    result = module.compute(off)
    assert any("nicht erreicht" in n for n in result.notes)


def test_dv2_always_available_after_dv1() -> None:
    # Δv₂ ist nach Δv₁ jederzeit und an jeder Position ausfuehrbar – auch an der
    # Periapsis (tau = 0), wo prograd typischerweise eine Fluchtbahn entsteht.
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    for tau in (0.0, 0.25, 0.5, 0.9):
        assert module.is_action_enabled("execute_dv2", {**values, "tau": tau}) is True
    # An der Periapsis (tau = 0) -> Fluchtbahn, aber kein Fehler.
    peri = {**values, "tau": 0.0}
    module.perform_action("execute_dv2", peri)
    assert module._phase == PHASE_AFTER_DV2
    result = module.compute(peri)
    assert any("nicht gebunden" in n or "Fluchtbahn" in n for n in result.notes)


def test_reset_returns_to_parking() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0])
    module.perform_action("execute_dv1", values)
    assert module._phase == PHASE_AFTER_DV1
    res = module.perform_action("reset_sim", values)
    assert module._phase == PHASE_PARKING
    assert res.reset_tau is True
    data = module.compute(values).data
    assert data["exp_dv1_orbit"] is None
    assert data["exp_dv2_orbit"] is None


# --- Δv₁-Verhalten ----------------------------------------------------------


def _after_dv1(dv1: float):
    module = HohmannTransferModule()
    values = _values(dv1=dv1)
    module.perform_action("execute_dv1", values)
    return module, _map(module.compute(values)), module.compute(values)


def test_ideal_dv1_apoapsis_reaches_target() -> None:
    _m, vals, result = _after_dv1(_ideal()[0])
    assert vals["Apoapsisradius"] == pytest.approx(R_GEO, rel=1e-3)
    assert any("erreicht die Zielbahn" in n for n in result.notes)


def test_small_dv1_apoapsis_below_target() -> None:
    _m, vals, result = _after_dv1(_ideal()[0] - 200.0)
    assert vals["Apoapsisradius"] < R_GEO
    assert any("zu klein" in n for n in result.notes)


def test_large_dv1_apoapsis_above_target() -> None:
    _m, vals, result = _after_dv1(_ideal()[0] + 200.0)
    assert vals["Apoapsisradius"] > R_GEO
    assert any("zu groß" in n for n in result.notes)


def test_zero_dv1_stays_on_start_circle() -> None:
    _m, vals, _result = _after_dv1(0.0)
    assert vals["Exzentrizität"] == pytest.approx(0.0, abs=1e-9)
    assert vals["Periapsisradius"] == pytest.approx(R_LEO)
    assert vals["Apoapsisradius"] == pytest.approx(R_LEO)


def test_very_large_dv1_is_escape() -> None:
    v_circ = om.circular_orbit_velocity(MU, R_LEO)
    v_esc = om.escape_velocity(MU, R_LEO)
    _m, vals, result = _after_dv1((v_esc - v_circ) + 1000.0)
    assert any("nicht gebunden" in n for n in result.notes)
    assert "Apoapsisradius" not in vals  # ungebunden -> keine Apoapsis
    # Δv₂ bleibt auch nach ungebundener Δv₁-Bahn jederzeit ausfuehrbar.
    assert _m.is_action_enabled("execute_dv2", _values()) is True


def test_bound_orbit_energy_negative() -> None:
    _m, vals, _result = _after_dv1(_ideal()[0])
    assert vals["Spez. Gesamtenergie"] < 0.0


def test_unbound_spacecraft_moves_along_arc() -> None:
    # Fluchtbahn nach Δv₂ am Periapsis: Raumfahrzeug friert NICHT ein, sondern
    # bewegt sich entlang des Bogens (verschiedene tau -> verschiedene Position).
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    module.perform_action("execute_dv2", {**values, "tau": 0.0})  # Periapsis -> Flucht
    positions = []
    for tau in (0.0, 0.3, 0.6, 0.9):
        d = module.compute({**values, "tau": tau}).data
        positions.append(d["exp_sc_position"][:2].copy())
    # Position aendert sich mit tau (nicht statisch).
    assert not np.allclose(positions[0], positions[2], atol=1.0)
    # Plot in allen Phasen weiterhin robust (auch ungebunden).
    fig = Figure()
    module.plot(fig, {**values, "tau": 0.4}, module.compute({**values, "tau": 0.4}))
    assert fig.axes


def test_propagate_handles_bound_and_unbound() -> None:
    from trajecto.core import orbit_determination as od2

    # Gebunden: tau=0 == Startzustand, tau=1 zurueck.
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, om.circular_orbit_velocity(MU, 7.0e6) * 1.1, 0.0])
    s0 = od2.propagate(MU, r, v, 0.0)
    s1 = od2.propagate(MU, r, v, 1.0)
    assert np.allclose(s0.position, r, atol=1.0)
    assert np.allclose(s1.position, r, atol=1.0)
    # Ungebunden (Fluchtbahn): tau=0 == Start, tau>0 weiter aussen (kein Crash).
    v_esc = np.array([0.0, om.escape_velocity(MU, 7.0e6) * 1.2, 0.0])
    u0 = od2.propagate(MU, r, v_esc, 0.0)
    u1 = od2.propagate(MU, r, v_esc, 1.0)
    assert np.allclose(u0.position, r, atol=1.0)
    assert float(np.linalg.norm(u1.position)) > float(np.linalg.norm(u0.position))


# --- Δv₂-Verhalten ----------------------------------------------------------


def _after_dv2(dv1: float, dv2: float):
    module = HohmannTransferModule()
    values = _values(dv1=dv1, dv2=dv2)
    module.perform_action("execute_dv1", values)
    # Zum gegenueberliegenden Apsidenpunkt fliegen (tau = 0.5), dort Δv₂.
    values_apo = {**values, "tau": 0.5}
    module.perform_action("execute_dv2", values_apo)
    return module, _map(module.compute(values_apo)), module.compute(values_apo)


def test_ideal_dv2_reaches_target_circle() -> None:
    dv1, dv2 = _ideal()
    module, vals, result = _after_dv2(dv1, dv2)
    assert module._phase == PHASE_AFTER_DV2
    # Nahezu kreisfoermig bei r2.
    assert vals["Periapsisradius"] == pytest.approx(R_GEO, rel=2e-3)
    assert vals["Apoapsisradius"] == pytest.approx(R_GEO, rel=2e-3)
    assert vals["Exzentrizität"] == pytest.approx(0.0, abs=2e-3)
    assert any("Zielkreisbahn erreicht" in n for n in result.notes)


def test_zero_dv2_keeps_dv1_orbit() -> None:
    dv1, _dv2 = _ideal()
    # Bahn nach Δv₁ (Apoapsis = r2) bleibt bei Δv₂ = 0 erhalten.
    _m1, vals1, _r1 = _after_dv1(dv1)
    _m2, vals2, _r2 = _after_dv2(dv1, 0.0)
    assert vals2["Apoapsisradius"] == pytest.approx(vals1["Apoapsisradius"], rel=1e-6)
    assert vals2["Periapsisradius"] == pytest.approx(vals1["Periapsisradius"], rel=1e-6)


# --- Δv-Budget --------------------------------------------------------------


def test_applied_budget_is_sum_of_impulses() -> None:
    dv1, dv2 = _ideal()
    _m, vals, _r = _after_dv2(dv1, dv2)
    total = abs(dv1) + abs(dv2)
    assert vals["Angewendetes Delta-v 1"] == pytest.approx(dv1)
    assert vals["Angewendetes Delta-v 2"] == pytest.approx(dv2)
    assert vals["Angewendetes Gesamt-Delta-v"] == pytest.approx(total)


def test_ideal_budget_matches_applied_for_ideal_impulses() -> None:
    dv1, dv2 = _ideal()
    _m, vals, _r = _after_dv2(dv1, dv2)
    assert vals["Ideales Gesamt-Delta-v"] == pytest.approx(
        vals["Angewendetes Gesamt-Delta-v"], abs=1.0
    )
    assert vals["Abweichung vom idealen Budget"] == pytest.approx(0.0, abs=1.0)


# --- experiment_state (Animation) -------------------------------------------


def test_experiment_state_returns_to_impulse_point() -> None:
    vy = om.circular_orbit_velocity(MU, R_LEO) + _ideal()[0]
    s0 = experiment_state(MU, R_LEO, vy, 0.0)
    s1 = experiment_state(MU, R_LEO, vy, 1.0)
    assert s0.position[0] == pytest.approx(R_LEO)
    assert np.allclose(s0.position, s1.position, atol=1.0)


def test_experiment_state_halfway_is_apoapsis() -> None:
    vy = om.circular_orbit_velocity(MU, R_LEO) + _ideal()[0]
    r_half = float(np.linalg.norm(experiment_state(MU, R_LEO, vy, 0.5).position))
    assert r_half == pytest.approx(R_GEO, rel=1e-3)


# --- Headless-Plot in allen Phasen ------------------------------------------


def test_plot_headless_all_phases() -> None:
    module = HohmannTransferModule()
    dv1, dv2 = _ideal()
    values = _values(dv1=dv1, dv2=dv2, tau=0.3)
    # parking
    figure = Figure(); module.plot(figure, values, module.compute(values)); assert figure.axes
    # after_dv1
    module.perform_action("execute_dv1", values)
    figure = Figure(); module.plot(figure, values, module.compute(values)); assert figure.axes
    # after_dv2
    module.perform_action("execute_dv2", values)
    figure = Figure(); module.plot(figure, values, module.compute(values)); assert figure.axes


def test_plot_headless_unbound_after_dv1() -> None:
    module = HohmannTransferModule()
    v_circ = om.circular_orbit_velocity(MU, R_LEO)
    v_esc = om.escape_velocity(MU, R_LEO)
    values = _values(dv1=(v_esc - v_circ) + 1500.0, tau=0.3)
    module.perform_action("execute_dv1", values)
    figure = Figure()
    module.plot(figure, values, module.compute(values))
    assert figure.axes


def test_plot_headless_with_energy_in_after_dv1() -> None:
    module = HohmannTransferModule()
    values = _values(dv1=_ideal()[0], tau=0.4)
    module.perform_action("execute_dv1", values)
    figure = Figure()
    module.plot(figure, values, module.compute(values), {"show_energy": True})
    assert len(figure.axes) == 2
