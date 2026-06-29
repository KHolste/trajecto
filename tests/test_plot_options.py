"""Tests fuer Plot-Optionen (ein-/ausblendbar) und den Energieverlauf."""

from __future__ import annotations

import itertools

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from trajecto.core import orbital_mechanics as om  # noqa: E402
from trajecto.core import state_vectors as sv  # noqa: E402
from trajecto.core.bodies import EARTH  # noqa: E402
from trajecto.modules.base import PlotOption  # noqa: E402
from trajecto.modules.circular_orbit import CircularOrbitModule  # noqa: E402
from trajecto.modules.dynamics_view import sample_energies  # noqa: E402
from trajecto.modules.hohmann_transfer import (  # noqa: E402
    HohmannTransferModule,
    transfer_s_from_tau,
    transfer_state,
)
from trajecto.modules.kepler_laws import KeplerLawsModule  # noqa: E402

MU = EARTH.mu


# --- Deklaration & Defaults -------------------------------------------------


def test_modules_declare_plot_options() -> None:
    for module in (CircularOrbitModule(), KeplerLawsModule(), HohmannTransferModule()):
        opts = module.plot_options()
        assert opts, f"{module.id} sollte Plot-Optionen deklarieren"
        assert all(isinstance(o, PlotOption) for o in opts)
        # Namen eindeutig.
        names = [o.name for o in opts]
        assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "module_cls, expected",
    [
        (CircularOrbitModule, {"show_radius": True, "show_velocity": True, "show_energy": False}),
        (
            KeplerLawsModule,
            {
                "show_radius": True, "show_velocity": True, "show_ecc": True,
                "show_segments": True, "show_axes": True, "show_energy": False,
            },
        ),
        (
            HohmannTransferModule,
            {"show_active_orbit": True, "show_velocity": True, "show_dv": True,
             "show_radius": False, "show_orbit_after_dv1": False,
             "show_ideal_orbit": False, "show_energy": False},
        ),
    ],
)
def test_plot_option_defaults(module_cls, expected: dict) -> None:
    module = module_cls()
    defaults = {o.name: o.default for o in module.plot_options()}
    assert defaults == expected
    # Energieplot ist ueberall standardmaessig aus.
    assert defaults["show_energy"] is False


def test_is_option_enabled_falls_back_to_default() -> None:
    module = KeplerLawsModule()
    # Ohne options-dict gilt der Default.
    assert module.is_option_enabled(None, "show_segments") is True
    assert module.is_option_enabled(None, "show_energy") is False
    # Explizit ueberschrieben.
    assert module.is_option_enabled({"show_segments": False}, "show_segments") is False
    assert module.is_option_enabled({"show_energy": True}, "show_energy") is True
    # Unbekannte Option -> False.
    assert module.is_option_enabled(None, "gibt_es_nicht") is False


# --- Plotfunktionen mit aktivierten/deaktivierten Optionen ------------------


def _all_combinations(module):
    names = [o.name for o in module.plot_options()]
    for bits in itertools.product([True, False], repeat=len(names)):
        yield dict(zip(names, bits))


def test_circular_plot_all_option_combinations_headless() -> None:
    module = CircularOrbitModule()
    result = module.compute({"body": EARTH.name, "radius": 7.0e6, "tau": 0.3})
    for options in _all_combinations(module):
        figure = Figure()
        module.plot(figure, {"tau": 0.3}, result, options)
        assert figure.axes
        # Mit Energieplot: zwei Achsen, sonst eine.
        assert len(figure.axes) == (2 if options["show_energy"] else 1)


def test_kepler_plot_options_on_and_off_headless() -> None:
    module = KeplerLawsModule()
    values = {"body": EARTH.name, "a": 2.4e7, "e": 0.3, "tau": 0.2, "segments": "6"}
    result = module.compute(values)
    for options in ({k: True for k in (o.name for o in module.plot_options())},
                    {k: False for k in (o.name for o in module.plot_options())}):
        figure = Figure()
        module.plot(figure, values, result, options)
        assert figure.axes
        assert len(figure.axes) == (2 if options["show_energy"] else 1)


def test_hohmann_plot_options_headless_both_directions() -> None:
    module = HohmannTransferModule()
    for r1, r2 in ((6.678e6, 4.2164e7), (4.2164e7, 6.678e6)):
        values = {"body": EARTH.name, "input_mode": "Radius vom Mittelpunkt",
                  "r1": r1, "r2": r2, "tau": 0.4}
        result = module.compute(values)
        for show_energy in (True, False):
            figure = Figure()
            module.plot(figure, values, result, {"show_energy": show_energy})
            assert len(figure.axes) == (2 if show_energy else 1)


def test_plot_without_options_uses_defaults_headless() -> None:
    # Aufruf ohne options-Argument (Bestandsverhalten) bleibt moeglich.
    module = KeplerLawsModule()
    values = {"body": EARTH.name, "a": 2.4e7, "e": 0.3, "tau": 0.2}
    result = module.compute(values)
    figure = Figure()
    module.plot(figure, values, result)  # options=None
    assert len(figure.axes) == 1  # show_energy Default False


# --- Energieverlauf ---------------------------------------------------------


def test_energy_curve_circular_total_constant() -> None:
    radius = 7.0e6
    _taus, kin, pot, tot = sample_energies(
        lambda tau: sv.state_circular(MU, radius, 2.0 * np.pi * tau)
    )
    # Auf der Kreisbahn sind alle drei Energien konstant.
    assert np.allclose(tot, tot[0])
    assert np.allclose(kin, kin[0])
    assert np.allclose(pot, pot[0])


def test_energy_curve_kepler_total_constant_kinetic_peri_gt_apo() -> None:
    a, e = 2.4e7, 0.3
    taus, kin, pot, tot = sample_energies(
        lambda tau: sv.state_from_kepler(
            MU, a, e, om.true_anomaly_from_mean(2.0 * np.pi * tau, e)
        )
    )
    # Gesamtenergie konstant = -mu/(2a).
    assert np.allclose(tot, -MU / (2.0 * a))
    # tau = 0 -> Periapsis (Index 0), tau = 0.5 -> Apoapsis (Mitte).
    mid = len(taus) // 2
    assert kin[0] > kin[mid]
    # Potentielle Energie ist im Periapsis niedriger (negativer) als im Apoapsis.
    assert pot[0] < pot[mid]


def test_energy_curve_hohmann_total_constant() -> None:
    r1, r2 = 6.678e6, 4.2164e7
    t = om.hohmann_transfer(MU, r1, r2)
    e_t = abs(r2 - r1) / (r1 + r2)
    _taus, _kin, _pot, tot = sample_energies(
        lambda tau: transfer_state(
            MU, t.a_transfer, e_t, t.outward, transfer_s_from_tau(tau, e_t, t.outward)
        )
    )
    # Gesamtenergie der Transferellipse konstant = -mu/(2*a_transfer).
    assert np.allclose(tot, -MU / (2.0 * t.a_transfer))
