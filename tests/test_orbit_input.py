"""Tests fuer die Eingabelogik Radius/Hoehe (``core.orbit_input``) und ihre
Anwendung in den Modulen ``circular_orbit`` und ``hohmann_transfer``."""

from __future__ import annotations

import pytest

from trajecto.core.bodies import EARTH
from trajecto.core.orbit_input import (
    INPUT_MODE_ALTITUDE,
    INPUT_MODE_RADIUS,
    altitude_from_radius,
    radius_from_input,
)
from trajecto.modules.base import ModuleResult
from trajecto.modules.circular_orbit import CircularOrbitModule
from trajecto.modules.hohmann_transfer import HohmannTransferModule

R_LEO = 6_678_000.0
R_GEO = 42_164_000.0


def _result_map(result: ModuleResult) -> dict[str, float]:
    return {item.label: item.value_si for item in result.items if hasattr(item, "value_si")}


# --- Core-Hilfslogik --------------------------------------------------------


def test_radius_mode_returns_value() -> None:
    assert radius_from_input(EARTH, 7.0e6, INPUT_MODE_RADIUS) == pytest.approx(7.0e6)


def test_altitude_mode_adds_body_radius() -> None:
    r = radius_from_input(EARTH, 500_000.0, INPUT_MODE_ALTITUDE)
    assert r == pytest.approx(EARTH.mean_radius + 500_000.0)


def test_altitude_from_radius_roundtrip() -> None:
    radius = EARTH.mean_radius + 500_000.0
    assert altitude_from_radius(EARTH, radius) == pytest.approx(500_000.0)


def test_default_mode_is_radius() -> None:
    assert radius_from_input(EARTH, 8.0e6) == pytest.approx(8.0e6)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        radius_from_input(EARTH, 7.0e6, "irgendwas")


def test_non_positive_resulting_radius_raises() -> None:
    # Hoehe so negativ, dass r <= 0.
    with pytest.raises(ValueError):
        radius_from_input(EARTH, -(EARTH.mean_radius + 1.0), INPUT_MODE_ALTITUDE)
    with pytest.raises(ValueError):
        radius_from_input(EARTH, 0.0, INPUT_MODE_RADIUS)


# --- circular_orbit ---------------------------------------------------------


def test_circular_altitude_500km_equals_radius_mode() -> None:
    module = CircularOrbitModule()
    by_altitude = module.compute(
        {"body": EARTH.name, "input_mode": INPUT_MODE_ALTITUDE, "radius": 500_000.0}
    )
    by_radius = module.compute(
        {
            "body": EARTH.name,
            "input_mode": INPUT_MODE_RADIUS,
            "radius": EARTH.mean_radius + 500_000.0,
        }
    )
    assert by_altitude.data["orbit_radius"] == pytest.approx(
        EARTH.mean_radius + 500_000.0
    )
    # Beide Eingabewege liefern identische Ergebnisse.
    assert _result_map(by_altitude) == pytest.approx(_result_map(by_radius))


def test_circular_radius_mode_backward_compatible() -> None:
    # Ohne input_mode wird weiterhin Radius angenommen (Bestandsverhalten).
    module = CircularOrbitModule()
    result = module.compute({"body": EARTH.name, "radius": 7.0e6})
    assert result.data["orbit_radius"] == pytest.approx(7.0e6)


def test_circular_negative_altitude_warns() -> None:
    module = CircularOrbitModule()
    result = module.compute(
        {"body": EARTH.name, "input_mode": INPUT_MODE_ALTITUDE, "radius": -100_000.0}
    )
    assert result.items  # Rechnung laeuft trotzdem
    assert any("Warnung" in n for n in result.notes)


def test_circular_resulting_radius_zero_raises() -> None:
    module = CircularOrbitModule()
    with pytest.raises(ValueError):
        module.compute(
            {
                "body": EARTH.name,
                "input_mode": INPUT_MODE_ALTITUDE,
                "radius": -EARTH.mean_radius,
            }
        )


# --- hohmann_transfer -------------------------------------------------------


def test_hohmann_altitude_mode_leo_to_geo() -> None:
    module = HohmannTransferModule()
    # Hoehen: 500 km -> 35786 km (GEO-Hoehe ueber mittlerem Erdradius).
    result = module.compute(
        {
            "body": EARTH.name,
            "input_mode": INPUT_MODE_ALTITUDE,
            "r1": 500_000.0,
            "r2": 35_786_000.0,
        }
    )
    values = _result_map(result)
    # Klassische LEO->GEO-Werte (Toleranz wegen mittlerem statt aequatorialem R).
    dv1 = next(v for k, v in values.items() if k.startswith("Delta-v 1"))
    dv2 = next(v for k, v in values.items() if k.startswith("Delta-v 2"))
    assert dv1 == pytest.approx(2430.0, abs=80.0)
    assert dv2 == pytest.approx(1470.0, abs=80.0)
    assert values["Gesamt-Delta-v"] == pytest.approx(3890.0, abs=100.0)
    # Verwendete Radien/Hoehen werden ausgewiesen.
    assert values["Verwendeter Startradius"] == pytest.approx(EARTH.mean_radius + 500_000.0)
    assert values["Verwendete Zielhöhe"] == pytest.approx(35_786_000.0)


def test_hohmann_radius_mode_backward_compatible() -> None:
    module = HohmannTransferModule()
    result = module.compute({"body": EARTH.name, "r1": R_LEO, "r2": R_GEO})
    values = _result_map(result)
    assert values["Verwendeter Startradius"] == pytest.approx(R_LEO)
    assert values["Verwendeter Zielradius"] == pytest.approx(R_GEO)
    assert values["Gesamt-Delta-v"] == pytest.approx(3900.0, abs=80.0)


def test_hohmann_negative_altitude_warns() -> None:
    module = HohmannTransferModule()
    result = module.compute(
        {
            "body": EARTH.name,
            "input_mode": INPUT_MODE_ALTITUDE,
            "r1": -50_000.0,  # 50 km unter der Oberflaeche
            "r2": 35_786_000.0,
        }
    )
    assert any("Warnung" in n for n in result.notes)


def test_hohmann_resulting_radius_non_positive_raises() -> None:
    module = HohmannTransferModule()
    with pytest.raises(ValueError):
        module.compute(
            {
                "body": EARTH.name,
                "input_mode": INPUT_MODE_ALTITUDE,
                "r1": -EARTH.mean_radius,  # r1 = 0
                "r2": 35_786_000.0,
            }
        )
