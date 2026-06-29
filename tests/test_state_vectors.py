"""Tests fuer Zustandsvektoren und dynamische Invarianten (``core.state_vectors``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import EARTH

MU = EARTH.mu


# --- Kreisbahn --------------------------------------------------------------


@pytest.mark.parametrize("nu_deg", [0.0, 45.0, 90.0, 217.0, 359.0])
def test_circular_radius_and_speed(nu_deg: float) -> None:
    r = 7.0e6
    state = sv.state_circular(MU, r, math.radians(nu_deg))
    assert sv.radius(state) == pytest.approx(r)
    assert sv.speed(state) == pytest.approx(om.circular_orbit_velocity(MU, r))


def test_circular_position_perpendicular_to_velocity() -> None:
    state = sv.state_circular(MU, 7.0e6, math.radians(123.0))
    dot = float(np.dot(state.position, state.velocity))
    # r senkrecht zu v -> Skalarprodukt ~ 0 (relativ zu Betragsprodukt).
    scale = sv.radius(state) * sv.speed(state)
    assert abs(dot) / scale < 1e-12


def test_circular_eccentricity_vector_near_zero() -> None:
    state = sv.state_circular(MU, 7.0e6, math.radians(40.0))
    assert float(np.linalg.norm(sv.eccentricity_vector(state))) < 1e-10


# --- Ellipse ----------------------------------------------------------------


def test_ellipse_radius_at_periapsis_and_apoapsis() -> None:
    a, e = 2.4e7, 0.3
    peri = sv.state_from_kepler(MU, a, e, 0.0)
    apo = sv.state_from_kepler(MU, a, e, math.pi)
    assert sv.radius(peri) == pytest.approx(om.periapsis_radius(a, e))
    assert sv.radius(apo) == pytest.approx(om.apoapsis_radius(a, e))


def test_ellipse_periapsis_faster_than_apoapsis() -> None:
    a, e = 2.4e7, 0.3
    v_peri = sv.speed(sv.state_from_kepler(MU, a, e, 0.0))
    v_apo = sv.speed(sv.state_from_kepler(MU, a, e, math.pi))
    assert v_peri > v_apo


def test_ellipse_total_energy_invariant() -> None:
    a, e = 2.4e7, 0.3
    energies = [
        sv.specific_total_energy(sv.state_from_kepler(MU, a, e, math.radians(d)))
        for d in (0.0, 37.0, 90.0, 180.0, 270.0)
    ]
    # Alle gleich und gleich -mu/(2a).
    for eps in energies:
        assert eps == pytest.approx(-MU / (2.0 * a))


def test_ellipse_angular_momentum_invariant() -> None:
    a, e = 2.4e7, 0.3
    mags = [
        float(np.linalg.norm(sv.specific_angular_momentum(
            sv.state_from_kepler(MU, a, e, math.radians(d)))))
        for d in (0.0, 50.0, 130.0, 200.0, 310.0)
    ]
    expected = math.sqrt(MU * a * (1.0 - e * e))  # h = sqrt(mu * p)
    for h in mags:
        assert h == pytest.approx(expected)


@pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.6, 0.85])
def test_eccentricity_vector_magnitude_equals_e(e: float) -> None:
    a = 2.0e7
    for nu_deg in (0.0, 73.0, 180.0, 300.0):
        state = sv.state_from_kepler(MU, a, e, math.radians(nu_deg))
        assert float(np.linalg.norm(sv.eccentricity_vector(state))) == pytest.approx(e, abs=1e-9)


def test_specific_energies_consistency() -> None:
    state = sv.state_from_kepler(MU, 2.4e7, 0.3, math.radians(60.0))
    assert sv.specific_total_energy(state) == pytest.approx(
        sv.specific_kinetic_energy(state) + sv.specific_potential_energy(state)
    )
    assert sv.specific_potential_energy(state) == pytest.approx(-MU / sv.radius(state))


# --- Validierung ------------------------------------------------------------


@pytest.mark.parametrize("a", [0.0, -1.0e6])
def test_invalid_a_raises(a: float) -> None:
    with pytest.raises(ValueError):
        sv.state_from_kepler(MU, a, 0.3, 0.0)


@pytest.mark.parametrize("e", [-0.1, 1.0, 1.5])
def test_invalid_e_raises(e: float) -> None:
    with pytest.raises(ValueError):
        sv.state_from_kepler(MU, 2.0e7, e, 0.0)
