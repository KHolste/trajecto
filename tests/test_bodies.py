"""Tests fuer den erweiterten Himmelskoerper-Katalog."""

from __future__ import annotations

import pytest

from trajecto.core.bodies import (
    BODIES,
    EARTH,
    JUPITER,
    MARS,
    MOON,
    SUN,
    VENUS,
    get_body,
)


@pytest.mark.parametrize("body", [SUN, EARTH, MOON, MARS, VENUS, JUPITER])
def test_bodies_have_valid_si_values(body) -> None:
    assert body.name
    assert body.mu > 0.0
    assert body.mean_radius > 0.0


def test_required_bodies_present_in_catalog() -> None:
    for name in ("Sonne", "Erde", "Mond", "Mars", "Venus", "Jupiter"):
        assert name in BODIES
        assert get_body(name) is BODIES[name]


def test_mu_ordering_is_physically_plausible() -> None:
    # Sonne >> Jupiter > Erde/Venus > Mars > Mond
    assert SUN.mu > JUPITER.mu > EARTH.mu > MARS.mu > MOON.mu
    assert EARTH.mu > VENUS.mu  # Erde minimal schwerer als Venus


def test_mass_property_consistent_with_mu() -> None:
    from trajecto.core.constants import GRAVITATIONAL_CONSTANT

    assert EARTH.mass == pytest.approx(EARTH.mu / GRAVITATIONAL_CONSTANT)


def test_unknown_body_raises() -> None:
    with pytest.raises(ValueError):
        get_body("Pluto")
