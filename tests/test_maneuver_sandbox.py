"""Tests fuer das Modul ``maneuver_sandbox`` und die Zeit-/Propagationslogik."""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from trajecto.core import orbit_determination as od  # noqa: E402
from trajecto.core import orbital_mechanics as om  # noqa: E402
from trajecto.core import timeutils  # noqa: E402
from trajecto.core.bodies import EARTH  # noqa: E402
from trajecto.modules.maneuver_sandbox import (  # noqa: E402
    DIR_PROGRADE,
    DIR_RETROGRADE,
    ManeuverSandboxModule,
)

MU = EARTH.mu


def _values(a=7.0e6, e=0.0, dv=100.0, direction=DIR_PROGRADE, epoch=60850.0):
    return {"body": EARTH.name, "a": a, "e": e, "dv": dv,
            "direction": direction, "epoch_mjd": epoch}


def _map(result):
    return {i.label: i.value_si for i in result.items if hasattr(i, "value_si")}


# --- Zeit / MJD -------------------------------------------------------------


def test_mjd_known_date() -> None:
    assert timeutils.datetime_to_mjd(datetime.datetime(2000, 1, 1)) == pytest.approx(51544.0)


def test_format_duration() -> None:
    assert timeutils.format_duration(0) == "0 d 00:00:00"
    assert timeutils.format_duration(90000) == "1 d 01:00:00"
    assert timeutils.format_duration(3661) == "0 d 01:01:01"


# --- Zeitbasierte Propagation ----------------------------------------------


def test_propagate_time_bound_period() -> None:
    r = np.array([7.0e6, 0.0, 0.0])
    vc = om.circular_orbit_velocity(MU, 7.0e6)
    v = np.array([0.0, vc, 0.0])
    period = om.circular_orbit_period(MU, 7.0e6)
    assert np.allclose(od.propagate_time(MU, r, v, 0.0).position, r, atol=1.0)
    assert np.allclose(od.propagate_time(MU, r, v, period).position, r, atol=1.0)
    half = od.propagate_time(MU, r, v, period / 2.0).position
    assert half[0] == pytest.approx(-7.0e6, rel=1e-3)


def test_propagate_time_unbound_moves_outward() -> None:
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, om.escape_velocity(MU, 7.0e6) * 1.3, 0.0])
    u0 = od.propagate_time(MU, r, v, 0.0)
    u1 = od.propagate_time(MU, r, v, 3600.0)
    assert np.allclose(u0.position, r, atol=1.0)
    assert float(np.linalg.norm(u1.position)) > float(np.linalg.norm(u0.position))


# --- Modul: Defaults / Uhr / Manoever ---------------------------------------


def test_default_start_is_circle() -> None:
    module = ManeuverSandboxModule()
    result = module.compute(_values(e=0.0))
    assert any("Kreisbahn" in n for n in result.notes)
    assert module.is_clock_driven() is True
    assert module.has_log() is True


def test_clock_advances_mission_time() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    for _ in range(20):
        module.advance_clock(0.05)
    result = module.compute(_values())
    assert _map(result)["Missionszeit"] > 0.0


def test_prograde_dv_raises_orbit() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values(dv=100.0, direction=DIR_PROGRADE))
    vals = _map(module.compute(_values()))
    # Prograd am Periapsis hebt die gegenueberliegende Apsis (Apoapsis > Start).
    assert vals["Apoapsisradius"] > 7.0e6


def test_retrograde_dv_lowers_orbit() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values(dv=100.0, direction=DIR_RETROGRADE))
    vals = _map(module.compute(_values()))
    # Retrograd am Periapsis senkt die gegenueberliegende Apsis (Periapsis sinkt).
    assert vals["Periapsisradius"] < 7.0e6


def test_multiple_maneuvers_logged_with_time() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values())
    module.advance_clock(0.5)
    module.perform_action("execute_dv", _values(direction=DIR_RETROGRADE))
    entries = module.log_entries()
    assert len(entries) == 3  # Start + 2 Manoever
    assert "#1" in entries[1] and "MJD" in entries[1]
    assert "prograd" in entries[1]
    assert "retrograd" in entries[2]


def test_applied_total_dv_accumulates() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values(dv=100.0))
    module.perform_action("execute_dv", _values(dv=150.0, direction=DIR_RETROGRADE))
    vals = _map(module.compute(_values()))
    assert vals["Angewendetes Gesamt-Δv"] == pytest.approx(250.0)


def test_reset_clears_clock_and_log() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values())
    module.advance_clock(0.3)
    module.reset_clock()
    module.compute(_values())
    assert module._mission_time == 0.0
    assert module.log_entries() == ["Start (T+0): Startbahn, MJD 60850.0000"]


def test_changing_start_orbit_resets_simulation() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values(a=7.0e6))
    module.perform_action("execute_dv", _values(a=7.0e6))
    assert len(module._maneuvers) == 1
    # Andere Startbahn -> Simulation wird zurueckgesetzt.
    module.compute(_values(a=8.0e6))
    assert len(module._maneuvers) == 0
    assert module._mission_time == 0.0


def test_epoch_shifts_mjd_without_reset() -> None:
    module = ManeuverSandboxModule()
    module.perform_action("execute_dv", _values(epoch=60000.0))
    e1 = module.log_entries()[1]
    # Andere Epoche -> MJD verschiebt sich, Manoever bleibt erhalten.
    module.compute(_values(epoch=70000.0))
    e2 = module.log_entries()[1]
    assert len(module._maneuvers) == 1
    assert e1 != e2 and "MJD 70000" in e2


def test_energy_history_jumps_per_maneuver() -> None:
    module = ManeuverSandboxModule()
    module.compute(_values())
    module.perform_action("execute_dv", _values())
    module.perform_action("execute_dv", _values())
    data = module.compute(_values()).data
    assert len(data["energy_history"]) == 3  # Start + 2 Manoever


def test_plot_headless_with_and_without_energy() -> None:
    module = ManeuverSandboxModule()
    module.perform_action("execute_dv", _values())
    module.advance_clock(0.4)
    result = module.compute(_values())
    for show_energy in (True, False):
        fig = Figure()
        module.plot(fig, _values(), result, {"show_energy": show_energy})
        assert len(fig.axes) == (2 if show_energy else 1)


def test_plot_headless_unbound_escape() -> None:
    # Sehr grosses Δv -> Fluchtbahn; Plot bleibt robust.
    module = ManeuverSandboxModule()
    module.compute(_values())
    big = om.escape_velocity(MU, 7.0e6)  # > Fluchtgeschwindigkeit
    module.perform_action("execute_dv", _values(dv=big))
    module.advance_clock(0.3)
    result = module.compute(_values())
    fig = Figure()
    module.plot(fig, _values(), result)
    assert fig.axes
