"""Tests fuer den Animations-Vertrag, auf den sich die GUI stuetzt.

Die GUI animiert generisch genau den als ``animatable`` deklarierten
Float-Parameter eines Moduls. Diese Tests sichern diesen Vertrag ohne Qt.
"""

from __future__ import annotations

import pytest

from trajecto.modules.base import FloatParameter
from trajecto.modules.circular_orbit import CircularOrbitModule
from trajecto.modules.hohmann_transfer import HohmannTransferModule
from trajecto.modules.kepler_laws import KeplerLawsModule


def _animatable(module) -> list[FloatParameter]:
    return [
        p
        for p in module.parameters()
        if isinstance(p, FloatParameter) and p.animatable
    ]


@pytest.mark.parametrize(
    "module",
    [CircularOrbitModule(), KeplerLawsModule(), HohmannTransferModule()],
)
def test_module_has_single_animatable_time_position(module) -> None:
    # Animiert wird die zeitbasierte Position tau (0..1), nicht der Winkel.
    animatable = _animatable(module)
    assert len(animatable) == 1
    param = animatable[0]
    assert param.name == "tau"
    assert param.minimum_si == 0.0
    assert param.maximum_si == 1.0


def test_float_parameter_not_animatable_by_default() -> None:
    p = FloatParameter(name="x", label="X", default_si=1.0, display_unit="km")
    assert p.animatable is False
