"""Tests fuer die Kepler-Kernfunktionen und das Modul ``kepler_laws``."""

from __future__ import annotations

import math

import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core.bodies import EARTH
from trajecto.modules.base import ModuleResult
from trajecto.modules.kepler_laws import KeplerLawsModule


def _result_map(result: ModuleResult) -> dict[str, float]:
    return {item.label: item.value_si for item in result.items}


# --- Kernfunktionen ---------------------------------------------------------


def test_orbital_period_formula_and_circular_equivalence() -> None:
    mu, a = EARTH.mu, 2.4e7
    assert om.orbital_period(mu, a) == pytest.approx(
        2.0 * math.pi * math.sqrt(a**3 / mu)
    )
    # Fuer a = r identisch mit der Kreisbahn-Umlaufzeit.
    r = 7.0e6
    assert om.orbital_period(mu, r) == pytest.approx(om.circular_orbit_period(mu, r))


def test_periapsis_and_apoapsis_radius() -> None:
    a, e = 2.0e7, 0.3
    assert om.periapsis_radius(a, e) == pytest.approx(a * (1.0 - e))
    assert om.apoapsis_radius(a, e) == pytest.approx(a * (1.0 + e))
    assert om.apoapsis_radius(a, e) > om.periapsis_radius(a, e)


def test_periapsis_apoapsis_collapse_to_a_for_circle() -> None:
    a = 1.5e7
    assert om.periapsis_radius(a, 0.0) == pytest.approx(a)
    assert om.apoapsis_radius(a, 0.0) == pytest.approx(a)


def test_vis_viva_reduces_to_circular_for_e_zero() -> None:
    mu, a = EARTH.mu, 8.0e6
    # Auf der Kreisbahn (r = a) liefert Vis-Viva die Kreisbahngeschwindigkeit.
    assert om.vis_viva_speed(mu, a, a) == pytest.approx(
        om.circular_orbit_velocity(mu, a)
    )


def test_vis_viva_periapsis_faster_than_apoapsis() -> None:
    mu, a, e = EARTH.mu, 2.4e7, 0.3
    r_p = om.periapsis_radius(a, e)
    r_a = om.apoapsis_radius(a, e)
    v_p = om.vis_viva_speed(mu, r_p, a)
    v_a = om.vis_viva_speed(mu, r_a, a)
    assert v_p > v_a
    # Analytisch: v_peri / v_apo = (1 + e) / (1 - e).
    assert v_p / v_a == pytest.approx((1.0 + e) / (1.0 - e))


def test_solve_eccentric_anomaly_basic() -> None:
    assert om.solve_eccentric_anomaly(0.0, 0.4) == pytest.approx(0.0)
    assert om.solve_eccentric_anomaly(math.pi, 0.4) == pytest.approx(math.pi)
    # Fuer e = 0 gilt E = M.
    assert om.solve_eccentric_anomaly(1.0, 0.0) == pytest.approx(1.0)


def test_solve_eccentric_anomaly_satisfies_kepler_equation() -> None:
    m, e = 1.2, 0.6
    eccentric = om.solve_eccentric_anomaly(m, e)
    assert eccentric - e * math.sin(eccentric) == pytest.approx(m, abs=1e-10)


@pytest.mark.parametrize("a", [0.0, -1.0e6])
def test_core_invalid_semi_major_axis_raises(a: float) -> None:
    with pytest.raises(ValueError):
        om.orbital_period(EARTH.mu, a)
    with pytest.raises(ValueError):
        om.periapsis_radius(a, 0.3)


@pytest.mark.parametrize("e", [-0.1, 1.0, 1.5])
def test_core_invalid_eccentricity_raises(e: float) -> None:
    with pytest.raises(ValueError):
        om.periapsis_radius(2.0e7, e)
    with pytest.raises(ValueError):
        om.apoapsis_radius(2.0e7, e)


# --- Modul ------------------------------------------------------------------


def test_module_metadata_and_parameters() -> None:
    module = KeplerLawsModule()
    assert module.id == "kepler_laws"
    names = {p.name for p in module.parameters()}
    assert {"body", "a", "e", "segments"} <= names


def test_module_plausibility() -> None:
    module = KeplerLawsModule()
    a, e = 2.4e7, 0.3
    result = module.compute({"body": EARTH.name, "a": a, "e": e, "segments": "6"})
    values = _result_map(result)

    assert values["Periapsisradius"] == pytest.approx(a * (1.0 - e))
    assert values["Apoapsisradius"] == pytest.approx(a * (1.0 + e))
    assert values["Umlaufzeit"] == pytest.approx(om.orbital_period(EARTH.mu, a))
    assert values["Geschwindigkeit im Periapsis"] > values[
        "Geschwindigkeit im Apoapsis"
    ]
    assert values["Verhaeltnis v_peri / v_apo"] == pytest.approx(
        (1.0 + e) / (1.0 - e)
    )


def test_module_circle_case() -> None:
    module = KeplerLawsModule()
    result = module.compute({"body": EARTH.name, "a": 1.5e7, "e": 0.0, "segments": "4"})
    values = _result_map(result)
    assert values["Periapsisradius"] == pytest.approx(values["Apoapsisradius"])
    assert values["Verhaeltnis v_peri / v_apo"] == pytest.approx(1.0)


def test_module_invalid_eccentricity_raises() -> None:
    module = KeplerLawsModule()
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "a": 2.0e7, "e": 1.0, "segments": "6"})
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "a": 2.0e7, "e": -0.2, "segments": "6"})


def test_module_invalid_semi_major_axis_raises() -> None:
    module = KeplerLawsModule()
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "a": 0.0, "e": 0.3, "segments": "6"})


def test_module_plot_runs_headless() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    module = KeplerLawsModule()
    values = {"body": EARTH.name, "a": 2.4e7, "e": 0.3, "segments": "6"}
    result = module.compute(values)
    figure = Figure()
    module.plot(figure, values, result)
    assert figure.axes
