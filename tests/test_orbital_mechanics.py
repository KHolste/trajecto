"""Tests fuer den Rechenkern der Bahnmechanik."""

from __future__ import annotations

import math

import pytest

from trajecto.core import orbital_mechanics as om
from trajecto.core.bodies import EARTH


def test_circular_orbit_velocity_known_value() -> None:
    # Erdnahe Kreisbahn auf Oberflaechenhoehe (~7,9 km/s, "erste kosmische
    # Geschwindigkeit").
    v = om.circular_orbit_velocity(EARTH.mu, EARTH.mean_radius)
    assert v == pytest.approx(7910.0, rel=1e-3)


def test_circular_orbit_velocity_formula() -> None:
    mu, r = EARTH.mu, 7.0e6
    assert om.circular_orbit_velocity(mu, r) == pytest.approx(math.sqrt(mu / r))


def test_circular_orbit_period_formula_and_value() -> None:
    mu, r = EARTH.mu, 6.771e6  # ~400 km Hoehe
    expected = 2.0 * math.pi * math.sqrt(r**3 / mu)
    T = om.circular_orbit_period(mu, r)
    assert T == pytest.approx(expected)
    # Eine LEO-Umlaufzeit liegt bei rund 90 Minuten.
    assert 5000.0 < T < 6000.0


def test_escape_velocity_is_sqrt2_times_circular() -> None:
    mu, r = EARTH.mu, 8.0e6
    v_circ = om.circular_orbit_velocity(mu, r)
    v_esc = om.escape_velocity(mu, r)
    assert v_esc == pytest.approx(math.sqrt(2.0) * v_circ)


@pytest.mark.parametrize("radius", [0.0, -1.0, -1.0e6])
def test_invalid_radius_raises(radius: float) -> None:
    with pytest.raises(ValueError):
        om.circular_orbit_velocity(EARTH.mu, radius)
    with pytest.raises(ValueError):
        om.circular_orbit_period(EARTH.mu, radius)
    with pytest.raises(ValueError):
        om.escape_velocity(EARTH.mu, radius)


@pytest.mark.parametrize("mu", [0.0, -1.0])
def test_invalid_mu_raises(mu: float) -> None:
    with pytest.raises(ValueError):
        om.circular_orbit_velocity(mu, 7.0e6)
