"""Tests fuer die zeitbasierte Animation (mittlere Anomalie statt linearer nu).

Physikalischer Kern: Eine gleichfoermige Zeitposition tau (M = 2*pi*tau)
erzeugt ueber die Keplergleichung eine zeitgetreue Bewegung – schnell nahe der
Periapsis, langsam nahe der Apoapsis. Auf der Kreisbahn bleibt sie gleichfoermig.
"""

from __future__ import annotations

import math

import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import EARTH
from trajecto.modules.base import ModuleResult
from trajecto.modules.circular_orbit import CircularOrbitModule
from trajecto.modules.hohmann_transfer import HohmannTransferModule
from trajecto.modules.kepler_laws import KeplerLawsModule

MU = EARTH.mu


def _result_map(result: ModuleResult) -> dict[str, float]:
    return {item.label: item.value_si for item in result.items if hasattr(item, "value_si")}


# --- true_anomaly_from_mean -------------------------------------------------


def test_true_anomaly_endpoints() -> None:
    # M = 0 -> nu = 0 (Periapsis); M = pi -> nu = pi (Apoapsis).
    assert om.true_anomaly_from_mean(0.0, 0.3) == pytest.approx(0.0)
    assert abs(om.true_anomaly_from_mean(math.pi, 0.3)) == pytest.approx(math.pi)


def test_true_anomaly_equals_mean_for_circle() -> None:
    for m in (0.3, 1.0, 2.5):
        assert om.true_anomaly_from_mean(m, 0.0) == pytest.approx(m)


# Anforderung 3: Gleiche Zeitschritte erzeugen nahe der Periapsis groessere
# Winkelaenderungen als nahe der Apoapsis.
def test_angle_change_larger_near_periapsis() -> None:
    e = 0.3
    dm = 0.02
    near_peri = om.true_anomaly_from_mean(dm, e) - om.true_anomaly_from_mean(0.0, e)
    apo = om.true_anomaly_from_mean(math.pi, e)
    near_apo = apo - om.true_anomaly_from_mean(math.pi - dm, e)
    assert near_peri > near_apo > 0.0


# --- Keplerellipse (Modul) --------------------------------------------------


# Anforderung 2: tau = 0 -> Periapsis, tau = 0.5 -> Apoapsis.
def test_kepler_tau_maps_to_apsides() -> None:
    module = KeplerLawsModule()
    a, e = 2.4e7, 0.3
    peri = _result_map(module.compute({"body": EARTH.name, "a": a, "e": e, "tau": 0.0}))
    apo = _result_map(module.compute({"body": EARTH.name, "a": a, "e": e, "tau": 0.5}))
    assert peri["Aktueller Radius"] == pytest.approx(a * (1.0 - e))
    assert apo["Aktueller Radius"] == pytest.approx(a * (1.0 + e))


# Anforderung 1: v(Periapsis) > v(Apoapsis) bei e = 0,3.
def test_kepler_periapsis_faster_than_apoapsis_via_tau() -> None:
    module = KeplerLawsModule()
    a, e = 2.4e7, 0.3
    v_peri = _result_map(
        module.compute({"body": EARTH.name, "a": a, "e": e, "tau": 0.0})
    )["Aktuelle Geschwindigkeit"]
    v_apo = _result_map(
        module.compute({"body": EARTH.name, "a": a, "e": e, "tau": 0.5})
    )["Aktuelle Geschwindigkeit"]
    assert v_peri > v_apo


def test_kepler_total_energy_invariant_over_tau() -> None:
    module = KeplerLawsModule()
    a, e = 2.4e7, 0.3
    energies = [
        _result_map(module.compute({"body": EARTH.name, "a": a, "e": e, "tau": tau}))[
            "Spez. Gesamtenergie"
        ]
        for tau in (0.0, 0.2, 0.5, 0.8)
    ]
    for eps in energies:
        assert eps == pytest.approx(-MU / (2.0 * a))


# --- Kreisbahn bleibt gleichfoermig (Anforderung 6) -------------------------


def test_circular_uniform_over_tau() -> None:
    module = CircularOrbitModule()
    speeds = []
    radii = []
    for tau in (0.0, 0.25, 0.5, 0.9):
        values = _result_map(
            module.compute(
                {"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                 "radius": 7.0e6, "tau": tau}
            )
        )
        speeds.append(values["Aktuelle Geschwindigkeit"])
        radii.append(values["Aktueller Radius"])
    # Geschwindigkeit und Radius konstant; nu = 2*pi*tau (gleichfoermig).
    for v in speeds:
        assert v == pytest.approx(speeds[0])
    for r in radii:
        assert r == pytest.approx(radii[0])


def test_circular_true_anomaly_linear_in_tau() -> None:
    module = CircularOrbitModule()

    def nu(tau: float) -> float:
        return _result_map(
            module.compute({"body": EARTH.name, "radius": 7.0e6, "tau": tau})
        )["Aktuelle wahre Anomalie"]

    # Gleiche tau-Schritte -> gleiche Winkelschritte (gleichfoermig).
    d1 = nu(0.2) - nu(0.1)
    d2 = nu(0.8) - nu(0.7)
    assert d1 == pytest.approx(d2)


# --- Hohmann-Transfer (Modul) -----------------------------------------------


# Anforderung 4: tau = 0 -> r1 & v_transfer_1; tau = 1 -> r2 & v_transfer_2.
def test_hohmann_outward_endpoints_via_tau() -> None:
    module = HohmannTransferModule()
    r1, r2 = 6_678_000.0, 42_164_000.0
    t = om.hohmann_transfer(MU, r1, r2)
    start = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 0.0})
    )
    end = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 1.0})
    )
    assert start["Aktueller Radius"] == pytest.approx(r1)
    assert start["Aktuelle Geschwindigkeit"] == pytest.approx(t.v_transfer_1)
    assert end["Aktueller Radius"] == pytest.approx(r2)
    assert end["Aktuelle Geschwindigkeit"] == pytest.approx(t.v_transfer_2)


# Anforderung 5: v am Start/Periapsis > v am Ziel/Apoapsis (aeusserer Transfer).
def test_hohmann_outward_start_faster_than_end_via_tau() -> None:
    module = HohmannTransferModule()
    r1, r2 = 6_678_000.0, 42_164_000.0
    v_start = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 0.0})
    )["Aktuelle Geschwindigkeit"]
    v_end = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 1.0})
    )["Aktuelle Geschwindigkeit"]
    assert v_start > v_end


def test_hohmann_inner_endpoints_via_tau() -> None:
    module = HohmannTransferModule()
    # Innerer Transfer: tau = 0 -> Startpunkt r1, tau = 1 -> Zielpunkt r2.
    r1, r2 = 42_164_000.0, 6_678_000.0
    t = om.hohmann_transfer(MU, r1, r2)
    start = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 0.0})
    )
    end = _result_map(
        module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                        "r1": r1, "r2": r2, "tau": 1.0})
    )
    assert start["Aktueller Radius"] == pytest.approx(r1)
    assert start["Aktuelle Geschwindigkeit"] == pytest.approx(t.v_transfer_1)
    assert end["Aktueller Radius"] == pytest.approx(r2)
    assert end["Aktuelle Geschwindigkeit"] == pytest.approx(t.v_transfer_2)


def test_hohmann_transfer_energy_invariant_over_tau() -> None:
    module = HohmannTransferModule()
    r1, r2 = 6_678_000.0, 42_164_000.0
    a = 0.5 * (r1 + r2)
    for tau in (0.0, 0.3, 0.5, 1.0):
        values = _result_map(
            module.compute({"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                            "r1": r1, "r2": r2, "tau": tau})
        )
        assert values["Spez. Gesamtenergie"] == pytest.approx(-MU / (2.0 * a))
