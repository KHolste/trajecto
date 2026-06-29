"""Tests fuer die Bahnbestimmung aus Zustandsvektoren (``core.orbit_determination``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trajecto.core import orbit_determination as od
from trajecto.core import orbital_mechanics as om
from trajecto.core.bodies import EARTH

MU = EARTH.mu
R = 7.0e6


def _circular_state(radius: float, factor: float = 1.0):
    """Position (radius,0,0), tangentiale Geschwindigkeit factor*v_kreis in +y."""
    v_circ = om.circular_orbit_velocity(MU, radius)
    return np.array([radius, 0.0, 0.0]), np.array([0.0, factor * v_circ, 0.0])


def test_circular_orbit_classified_as_circle() -> None:
    pos, vel = _circular_state(R, 1.0)
    el = od.orbital_elements(MU, pos, vel)
    assert el.classification == od.CLASS_CIRCLE
    assert el.eccentricity == pytest.approx(0.0, abs=1e-9)
    assert el.is_bound is True
    assert el.semi_major_axis == pytest.approx(R)


def test_prograde_burn_gives_bound_ellipse() -> None:
    pos, vel = _circular_state(R, 1.2)  # 20 % schneller -> Ellipse
    el = od.orbital_elements(MU, pos, vel)
    assert el.classification == od.CLASS_ELLIPSE
    assert el.is_bound is True
    # Energie gebundener Bahn ist negativ.
    assert el.specific_energy < 0.0
    # Impulspunkt ist Periapsis -> Periapsis = R.
    assert el.periapsis_radius == pytest.approx(R)
    assert el.apoapsis_radius > R


def test_escape_speed_is_not_bound() -> None:
    # Genau Fluchtgeschwindigkeit -> Parabel; mehr -> Hyperbel.
    v_esc = om.escape_velocity(MU, R)
    pos = np.array([R, 0.0, 0.0])
    el_hyper = od.orbital_elements(MU, pos, np.array([0.0, 1.2 * v_esc, 0.0]))
    assert el_hyper.classification == od.CLASS_HYPERBOLA
    assert el_hyper.is_bound is False
    assert el_hyper.specific_energy > 0.0
    assert el_hyper.semi_major_axis is None
    assert el_hyper.apoapsis_radius is None
    assert el_hyper.period is None


def test_eccentricity_matches_ecc_vector() -> None:
    pos, vel = _circular_state(R, 1.3)
    el = od.orbital_elements(MU, pos, vel)
    assert el.eccentricity == pytest.approx(float(np.linalg.norm(el.ecc_vector)))


def test_period_consistent_for_bound() -> None:
    pos, vel = _circular_state(R, 1.1)
    el = od.orbital_elements(MU, pos, vel)
    expected = 2.0 * math.pi * math.sqrt(el.semi_major_axis**3 / MU)
    assert el.period == pytest.approx(expected)


def test_invalid_mu_raises() -> None:
    with pytest.raises(ValueError):
        od.orbital_elements(0.0, np.array([R, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
