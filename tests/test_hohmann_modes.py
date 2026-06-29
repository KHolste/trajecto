"""Tests fuer die Bedienmodi (Ideal / Δv₁-Experiment) und Ergebnisgliederung."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from trajecto.core.bodies import EARTH  # noqa: E402
from trajecto.modules.base import ChoiceParameter, ResultSection  # noqa: E402
from trajecto.modules.hohmann_transfer import (  # noqa: E402
    MODE_EXPERIMENT,
    MODE_IDEAL,
    HohmannTransferModule,
)

RADIUS_MODE = "Radius vom Mittelpunkt"
R_LEO = 6_678_000.0
R_GEO = 42_164_000.0


def _values(mode: str) -> dict:
    return {"mode": mode, "body": EARTH.name, "input_mode": RADIUS_MODE,
            "r1": R_LEO, "r2": R_GEO, "tau": 0.3}


def _labels(result) -> set[str]:
    return {i.label for i in result.items if hasattr(i, "label")}


def test_mode_parameter_exists_with_both_modes() -> None:
    module = HohmannTransferModule()
    mode_params = [p for p in module.parameters()
                   if isinstance(p, ChoiceParameter) and p.name == "mode"]
    assert len(mode_params) == 1
    assert set(mode_params[0].choices) == {MODE_IDEAL, MODE_EXPERIMENT}
    assert mode_params[0].default == MODE_IDEAL


def test_ideal_mode_hides_experiment_budget() -> None:
    module = HohmannTransferModule()
    labels = _labels(module.compute(_values(MODE_IDEAL)))
    # Ideale Kernwerte sind da ...
    assert "Gesamt-Delta-v" in labels
    # ... das Δv-Budget des Experiments aber nicht.
    assert "Angewendetes Gesamt-Delta-v" not in labels


def test_experiment_mode_shows_budget_in_parking_phase() -> None:
    module = HohmannTransferModule()  # Startphase: parking_orbit
    result = module.compute(_values(MODE_EXPERIMENT))
    labels = _labels(result)
    # Budget ist sichtbar ...
    assert "Ideales Delta-v 1" in labels
    assert "Angewendetes Gesamt-Delta-v" in labels
    # ... Phasehinweis vorhanden.
    assert any("Startkreisbahn" in n for n in result.notes)


def test_results_are_grouped_with_sections() -> None:
    module = HohmannTransferModule()
    for mode in (MODE_IDEAL, MODE_EXPERIMENT):
        result = module.compute(_values(mode))
        sections = [i.title for i in result.items if isinstance(i, ResultSection)]
        assert sections  # mindestens eine Gruppenueberschrift


@pytest.mark.parametrize("mode", [MODE_IDEAL, MODE_EXPERIMENT])
def test_plot_headless_per_mode(mode: str) -> None:
    module = HohmannTransferModule()
    values = _values(mode)
    result = module.compute(values)
    figure = Figure()
    module.plot(figure, values, result)  # Default-Optionen
    assert figure.axes
