"""Tests fuer die Hohmann-Kernfunktion und das Modul ``hohmann_transfer``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import EARTH
from trajecto.modules.base import ModuleResult
from trajecto.modules.hohmann_transfer import HohmannTransferModule, transfer_state

# Referenzfall LEO -> GEO (Radien vom Erdmittelpunkt).
R_LEO = 6_678_000.0
R_GEO = 42_164_000.0


def _result_map(result: ModuleResult) -> dict[str, float]:
    return {item.label: item.value_si for item in result.items if hasattr(item, "value_si")}


# --- Kernfunktion -----------------------------------------------------------


def test_transfer_semi_major_axis() -> None:
    t = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    assert t.a_transfer == pytest.approx(0.5 * (R_LEO + R_GEO))


def test_leo_to_geo_delta_v_known_values() -> None:
    # Klassischer Lehrbuchfall: dv1 ~ 2.42 km/s, dv2 ~ 1.47 km/s, total ~ 3.9 km/s.
    t = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    assert abs(t.dv1) == pytest.approx(2420.0, abs=60.0)
    assert abs(t.dv2) == pytest.approx(1470.0, abs=60.0)
    assert t.dv_total == pytest.approx(3900.0, abs=80.0)
    # Aussentransfer: beide Schuebe prograd (positiv).
    assert t.outward is True
    assert t.dv1 > 0.0
    assert t.dv2 > 0.0


def test_total_delta_v_is_sum_of_magnitudes() -> None:
    t = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    assert t.dv_total == pytest.approx(abs(t.dv1) + abs(t.dv2))


def test_transfer_time_is_half_period() -> None:
    t = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    expected = 0.5 * om.orbital_period(EARTH.mu, t.a_transfer)
    assert t.transfer_time == pytest.approx(expected)
    # LEO->GEO Transfer dauert ca. 5,3 Stunden.
    assert 5.0 * 3600.0 < t.transfer_time < 5.6 * 3600.0


def test_inner_transfer_is_retrograde_and_symmetric() -> None:
    outward = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    inward = om.hohmann_transfer(EARTH.mu, R_GEO, R_LEO)
    # Innentransfer: nach innen, beide Schuebe retrograd (negativ).
    assert inward.outward is False
    assert inward.dv1 < 0.0
    assert inward.dv2 < 0.0
    # Gleiche Halbachse, gleiche Transferzeit, gleiches Gesamt-Delta-v.
    assert inward.a_transfer == pytest.approx(outward.a_transfer)
    assert inward.transfer_time == pytest.approx(outward.transfer_time)
    assert inward.dv_total == pytest.approx(outward.dv_total)


def test_transfer_velocities_consistent_with_vis_viva() -> None:
    t = om.hohmann_transfer(EARTH.mu, R_LEO, R_GEO)
    a = t.a_transfer
    assert t.v_transfer_1 == pytest.approx(om.vis_viva_speed(EARTH.mu, R_LEO, a))
    assert t.v_transfer_2 == pytest.approx(om.vis_viva_speed(EARTH.mu, R_GEO, a))


def test_equal_radii_raises() -> None:
    with pytest.raises(ValueError):
        om.hohmann_transfer(EARTH.mu, R_LEO, R_LEO)


@pytest.mark.parametrize("r1,r2", [(0.0, R_GEO), (-1.0, R_GEO), (R_LEO, 0.0), (R_LEO, -5.0)])
def test_non_positive_radius_raises(r1: float, r2: float) -> None:
    with pytest.raises(ValueError):
        om.hohmann_transfer(EARTH.mu, r1, r2)


def test_invalid_mu_raises() -> None:
    with pytest.raises(ValueError):
        om.hohmann_transfer(0.0, R_LEO, R_GEO)


# --- Modul ------------------------------------------------------------------


def test_module_metadata_and_parameters() -> None:
    module = HohmannTransferModule()
    assert module.id == "hohmann_transfer"
    names = {p.name for p in module.parameters()}
    assert {"body", "r1", "r2"} <= names


def test_module_plausibility_outward() -> None:
    module = HohmannTransferModule()
    result = module.compute({"body": EARTH.name, "r1": R_LEO, "r2": R_GEO})
    values = _result_map(result)

    assert values["Gesamt-Delta-v"] == pytest.approx(3900.0, abs=80.0)
    assert values["Halbachse Transferellipse"] == pytest.approx(0.5 * (R_LEO + R_GEO))
    # Aussentransfer: Zielbahn langsamer als Startbahn (GEO < LEO Geschwindigkeit).
    assert values["Kreisbahngeschw. Zielbahn"] < values["Kreisbahngeschw. Startbahn"]
    # Hinweis auf Transferart vorhanden.
    assert any("nach aussen" in n for n in result.notes)


def test_module_warns_below_body_radius() -> None:
    module = HohmannTransferModule()
    # r1 unterhalb des Erdradius -> Warnhinweis, aber Rechnung laeuft.
    result = module.compute({"body": EARTH.name, "r1": 1_000_000.0, "r2": R_GEO})
    assert result.items  # Ergebnis wurde berechnet
    assert any("Warnung" in n for n in result.notes)


def test_module_equal_radii_raises() -> None:
    module = HohmannTransferModule()
    with pytest.raises(ValueError):
        module.compute({"body": EARTH.name, "r1": R_LEO, "r2": R_LEO})


def test_module_plot_runs_headless() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    module = HohmannTransferModule()
    # Beide Richtungen, verschiedene Transferpositionen muessen robust zeichnen.
    for values in (
        {"body": EARTH.name, "r1": R_LEO, "r2": R_GEO, "s": 0.0},
        {"body": EARTH.name, "r1": R_LEO, "r2": R_GEO, "s": math.radians(90.0)},
        {"body": EARTH.name, "r1": R_GEO, "r2": R_LEO, "s": math.radians(90.0)},
        {"body": EARTH.name, "r1": R_GEO, "r2": R_LEO, "s": math.pi},
    ):
        result = module.compute(values)
        figure = Figure()
        module.plot(figure, values, result)
        assert figure.axes


# --- Dynamik auf der Transferellipse ----------------------------------------


def _transfer_a_e(r1: float, r2: float) -> tuple[float, float]:
    return 0.5 * (r1 + r2), abs(r2 - r1) / (r1 + r2)


@pytest.mark.parametrize("r1,r2", [(R_LEO, R_GEO), (R_GEO, R_LEO)])
def test_transfer_state_endpoints_radius_and_speed(r1: float, r2: float) -> None:
    a, e = _transfer_a_e(r1, r2)
    outward = r2 > r1
    t = om.hohmann_transfer(EARTH.mu, r1, r2)
    # s = 0 -> Startpunkt (r1, v_transfer_1); s = pi -> Zielpunkt (r2, v_transfer_2).
    start = transfer_state(EARTH.mu, a, e, outward, 0.0)
    end = transfer_state(EARTH.mu, a, e, outward, math.pi)
    assert sv.radius(start) == pytest.approx(r1)
    assert sv.radius(end) == pytest.approx(r2)
    assert sv.speed(start) == pytest.approx(t.v_transfer_1)
    assert sv.speed(end) == pytest.approx(t.v_transfer_2)


def test_transfer_state_endpoints_on_plot_axis() -> None:
    # Startpunkt liegt auf +x, Zielpunkt auf -x (Plot-/Fokus-Rahmen).
    a, e = _transfer_a_e(R_LEO, R_GEO)
    start = transfer_state(EARTH.mu, a, e, True, 0.0)
    end = transfer_state(EARTH.mu, a, e, True, math.pi)
    assert start.position[0] == pytest.approx(R_LEO)
    assert start.position[1] == pytest.approx(0.0, abs=1e-3)
    assert end.position[0] == pytest.approx(-R_GEO)
    assert end.position[1] == pytest.approx(0.0, abs=1e-3)


def test_transfer_outward_start_faster_than_end() -> None:
    a, e = _transfer_a_e(R_LEO, R_GEO)
    v_start = sv.speed(transfer_state(EARTH.mu, a, e, True, 0.0))
    v_end = sv.speed(transfer_state(EARTH.mu, a, e, True, math.pi))
    assert v_start > v_end


def test_transfer_total_energy_invariant_over_position() -> None:
    a, e = _transfer_a_e(R_LEO, R_GEO)
    energies = [
        sv.specific_total_energy(transfer_state(EARTH.mu, a, e, True, s))
        for s in (0.0, math.radians(45.0), math.radians(90.0), math.pi)
    ]
    for eps in energies:
        assert eps == pytest.approx(-EARTH.mu / (2.0 * a))


def test_transfer_energy_between_start_and_target_outward() -> None:
    a, _e = _transfer_a_e(R_LEO, R_GEO)
    e_start = -EARTH.mu / (2.0 * R_LEO)
    e_transfer = -EARTH.mu / (2.0 * a)
    e_target = -EARTH.mu / (2.0 * R_GEO)
    # Aeusserer Transfer: Startbahn am energieaermsten, Zielbahn am hoechsten.
    assert e_start < e_transfer < e_target


def test_module_exposes_dynamic_and_energy_results() -> None:
    module = HohmannTransferModule()
    result = module.compute(
        {"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
         "r1": R_LEO, "r2": R_GEO, "s": 0.0}
    )
    values = _result_map(result)
    for label in (
        "Aktueller Radius",
        "Aktuelle Geschwindigkeit",
        "Spez. kinetische Energie",
        "Spez. potentielle Energie",
        "Spez. Gesamtenergie",
        "Betrag spez. Drehimpuls",
        "Betrag Exzentrizitätsvektor",
        "Gesamtenergie Startkreisbahn",
        "Gesamtenergie Transferellipse",
        "Gesamtenergie Zielkreisbahn",
    ):
        assert label in values
    # Bei s = 0 entspricht der Zustand dem Startpunkt.
    assert values["Aktueller Radius"] == pytest.approx(R_LEO)
    assert values["Aktuelle Geschwindigkeit"] == pytest.approx(
        values["Transfergeschw. am Startpunkt"]
    )
    # Energievergleich: Start < Transfer < Ziel (aeusserer Transfer).
    assert (
        values["Gesamtenergie Startkreisbahn"]
        < values["Gesamtenergie Transferellipse"]
        < values["Gesamtenergie Zielkreisbahn"]
    )
    # Exzentrizitaet der Transferellipse.
    assert values["Betrag Exzentrizitätsvektor"] == pytest.approx(
        abs(R_GEO - R_LEO) / (R_GEO + R_LEO)
    )
