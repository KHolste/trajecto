"""Tests fuer das didaktische Referenzmodul ``circular_orbit``."""

from __future__ import annotations

import math

import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core.bodies import EARTH
from trajecto.modules.base import ModuleResult
from trajecto.modules.circular_orbit import CircularOrbitModule


def _result_map(result: ModuleResult) -> dict[str, float]:
    return {item.label: item.value_si for item in result.items if hasattr(item, "value_si")}


def test_module_metadata() -> None:
    module = CircularOrbitModule()
    assert module.id == "circular_orbit"
    assert module.title
    assert module.parameters()  # nicht leer


def test_compute_plausibility_for_earth() -> None:
    module = CircularOrbitModule()
    radius = EARTH.mean_radius + 500_000.0
    result = module.compute({"body": EARTH.name, "radius": radius})
    values = _result_map(result)

    # Werte stimmen mit dem Rechenkern ueberein.
    assert values["Kreisbahngeschwindigkeit"] == pytest.approx(
        om.circular_orbit_velocity(EARTH.mu, radius)
    )
    assert values["Umlaufzeit"] == pytest.approx(
        om.circular_orbit_period(EARTH.mu, radius)
    )
    assert values["Fluchtgeschwindigkeit"] == pytest.approx(
        om.escape_velocity(EARTH.mu, radius)
    )

    # Plausibilitaet: LEO-Geschwindigkeit ~7,6 km/s, Flucht = sqrt(2) * Kreis.
    assert 7000.0 < values["Kreisbahngeschwindigkeit"] < 8000.0
    assert values["Fluchtgeschwindigkeit"] == pytest.approx(
        math.sqrt(2.0) * values["Kreisbahngeschwindigkeit"]
    )
    assert values["Bahnhöhe über Oberfläche"] == pytest.approx(500_000.0)


def test_compute_data_for_plot() -> None:
    module = CircularOrbitModule()
    radius = 7.0e6
    result = module.compute({"body": EARTH.name, "radius": radius})
    assert result.data["orbit_radius"] == pytest.approx(radius)
    assert result.data["body_radius"] == pytest.approx(EARTH.mean_radius)
    assert result.data["body_name"] == EARTH.name


def test_invalid_radius_raises() -> None:
    module = CircularOrbitModule()
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "radius": 0.0})
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "radius": -1.0e6})


def test_unknown_body_raises() -> None:
    module = CircularOrbitModule()
    with pytest.raises(ValueError):
        module.compute({"body": "Unbekannt", "radius": 7.0e6})


def test_module_exposes_dynamic_state_results() -> None:
    module = CircularOrbitModule()
    import math as _math

    result = module.compute(
        {"body": EARTH.name, "radius": 7.0e6, "nu": _math.radians(45.0)}
    )
    values = _result_map(result)
    labels = set(values)
    for label in (
        "Aktueller Radius",
        "Aktuelle Geschwindigkeit",
        "Spez. kinetische Energie",
        "Spez. potentielle Energie",
        "Spez. Gesamtenergie",
        "Betrag spez. Drehimpuls",
        "Betrag Exzentrizitätsvektor",
    ):
        assert label in labels
    # Kreisbahn: aktueller Radius == Bahnradius, v == Kreisbahngeschwindigkeit.
    assert values["Aktueller Radius"] == pytest.approx(7.0e6)
    assert values["Aktuelle Geschwindigkeit"] == pytest.approx(
        om.circular_orbit_velocity(EARTH.mu, 7.0e6)
    )
    # Exzentrizitaetsvektor numerisch ~ 0.
    assert values["Betrag Exzentrizitätsvektor"] == pytest.approx(0.0, abs=1e-9)


def test_plot_runs_without_gui() -> None:
    # Stellt sicher, dass die Visualisierung headless (ohne Qt) funktioniert.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    module = CircularOrbitModule()
    values = {"body": EARTH.name, "radius": 7.0e6}
    result = module.compute(values)
    figure = Figure()
    module.plot(figure, values, result)
    assert figure.axes  # es wurde eine Achse gezeichnet
