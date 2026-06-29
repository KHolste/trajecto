"""Modul ``maneuver_sandbox`` – freie Manoever-Sandbox mit Missionsuhr.

Ein Raumfahrzeug startet auf einer einstellbaren Bahn (Kreis als Spezialfall der
Ellipse, Kreis ist Default). Die **Missionsuhr laeuft kontinuierlich** weiter; an
jeder Stelle kann beliebig oft ein tangentiales Delta-v (prograd = beschleunigen,
retrograd = abbremsen) gegeben werden. Jedes Manoever wird mit Zeit (Tage/Std/Min
und MJD) protokolliert. Optional zeigt ein Zusatzplot die spezifische mechanische
Gesamtenergie ueber die Missionszeit (mit Spruengen an jedem Manoever).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from trajecto.core import applog
from trajecto.core import orbit_determination as od
from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core import timeutils
from trajecto.core.bodies import BODIES, EARTH, get_body
from trajecto.core.units import from_si
from trajecto.modules.base import (
    Action,
    ActionResult,
    ChoiceParameter,
    FloatParameter,
    Module,
    ModuleResult,
    Parameter,
    PlotOption,
    ResultItem,
    ResultSection,
)
from trajecto.modules.dynamics_view import draw_state_vectors

#: Default-Startbahn: Kreisbahn mit a = 7000 km (e = 0).
_DEFAULT_A_SI = 7_000_000.0
_DEFAULT_E = 0.0
_DEFAULT_DV_SI = 100.0  # 0.1 km/s
_DEFAULT_EPOCH_MJD = timeutils.today_mjd()

#: Schubrichtungen.
DIR_PROGRADE = "prograd (beschleunigen)"
DIR_RETROGRADE = "retrograd (abbremsen)"
DIRECTIONS = (DIR_PROGRADE, DIR_RETROGRADE)


class ManeuverSandboxModule(Module):
    """Freie Manoever-Sandbox mit durchlaufender Missionsuhr."""

    id = "maneuver_sandbox"
    title = "Manöver-Sandbox"
    subtitle = "Freie Delta-v-Manöver mit Missionsuhr"

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._mission_time = 0.0          # Sekunden seit Missionsstart
        self._ref_time = 0.0              # Missionszeit zu Beginn des Bahnsegments
        self._ref_state: sv.OrbitState | None = None  # aktuelle Bahn (Referenz)
        self._maneuvers: list[dict] = []  # protokollierte Manoever
        self._energy_history: list[tuple[float, float]] = []  # (t, E spez.)
        self._signature: tuple | None = None  # Startbahn-Signatur (a, e, body)
        self._signature_epoch = _DEFAULT_EPOCH_MJD  # MJD-Bezug fuer das Log

    def on_activated(self) -> None:
        self._reset_state()

    # -- Parameter / Optionen / Aktionen --------------------------------------

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(name="body", label="Zentralkörper",
                            choices=tuple(BODIES.keys()), default=EARTH.name,
                            group="Startbahn"),
            FloatParameter(name="a", label="Große Halbachse a", default_si=_DEFAULT_A_SI,
                           display_unit="km", minimum_si=1.0,
                           tooltip="Große Halbachse der Startbahn (Kreis: a = r)",
                           group="Startbahn"),
            FloatParameter(name="e", label="Exzentrizität e", default_si=_DEFAULT_E,
                           display_unit="-", minimum_si=0.0, maximum_si=0.95,
                           tooltip="0 = Kreisbahn (Default), näher 1 = stark elliptisch",
                           group="Startbahn"),
            FloatParameter(name="dv", label="Δv-Betrag", default_si=_DEFAULT_DV_SI,
                           display_unit="km/s", minimum_si=0.0, maximum_si=2.0e4,
                           tooltip="Betrag des tangentialen Impulses",
                           group="Manöver"),
            ChoiceParameter(name="direction", label="Schubrichtung",
                            choices=DIRECTIONS, default=DIR_PROGRADE,
                            tooltip="prograd = beschleunigen, retrograd = abbremsen",
                            group="Manöver"),
            FloatParameter(name="epoch_mjd", label="Epoche (MJD)",
                           default_si=_DEFAULT_EPOCH_MJD, display_unit="-",
                           minimum_si=0.0, maximum_si=1.0e6,
                           tooltip="MJD-Bezugspunkt (Default: heute, frei einstellbar)",
                           group="Zeit"),
        ]

    def plot_options(self) -> list[PlotOption]:
        return [
            PlotOption("show_orbit", "Aktuelle Bahn anzeigen", True),
            PlotOption("show_velocity", "Geschwindigkeitsvektor anzeigen", True),
            PlotOption("show_radius", "Radiusvektor anzeigen", False),
            PlotOption("show_energy", "Energie über Missionszeit anzeigen", False),
        ]

    def actions(self) -> list[Action]:
        return [
            Action("execute_dv", "Δv ausführen",
                   tooltip="Tangentiales Δv an der aktuellen Position ausführen"),
            Action("reset_sim", "Simulation zurücksetzen",
                   tooltip="Startbahn, Uhr und Log zurücksetzen"),
        ]

    def is_action_enabled(self, name: str, values: dict[str, Any]) -> bool:
        if name == "reset_sim":
            return bool(self._maneuvers) or self._mission_time > 0.0
        return True  # execute_dv: jederzeit

    def perform_action(self, name: str, values: dict[str, Any]) -> ActionResult:
        if name == "reset_sim":
            self.reset_clock()
            return ActionResult(stop_animation=True)
        if name == "execute_dv":
            self._ensure_init(values)
            mu = get_body(values["body"]).mu
            state = self._current_state(mu)
            dv = float(values.get("dv", _DEFAULT_DV_SI))
            sign = 1.0 if values.get("direction", DIR_PROGRADE) == DIR_PROGRADE else -1.0
            vel = np.asarray(state.velocity, dtype=float)
            speed = float(np.linalg.norm(vel))
            if speed > 0.0:
                vel = vel + sign * dv * (vel / speed)
            new_state = sv.OrbitState(mu=mu, position=np.asarray(state.position, float),
                                      velocity=vel)
            self._ref_state = new_state
            self._ref_time = self._mission_time
            el = od.orbital_elements(mu, new_state.position, new_state.velocity)
            self._maneuvers.append({
                "time": self._mission_time, "dv": dv, "sign": sign,
                "classification": el.classification,
                "a": el.semi_major_axis, "e": el.eccentricity,
            })
            self._energy_history.append(
                (self._mission_time, sv.specific_total_energy(new_state)))
            applog.logger.info("Sandbox: Δv=%.1f m/s %s bei T=%.0f s -> %s",
                               dv, "prograd" if sign > 0 else "retrograd",
                               self._mission_time, el.classification)
            return ActionResult()  # Animation laeuft weiter
        return ActionResult()

    # -- Missionsuhr ----------------------------------------------------------

    def is_clock_driven(self) -> bool:
        return True

    def advance_clock(self, orbit_fraction: float) -> None:
        if self._ref_state is None:
            return
        self._mission_time += orbit_fraction * self._characteristic_time()

    def reset_clock(self) -> None:
        self._reset_state()  # erzwingt Neu-Initialisierung beim naechsten compute

    def _characteristic_time(self) -> float:
        """Charakteristische Zeit der aktuellen Bahn (Periode bzw. Hyperbel-Analog)."""
        if self._ref_state is None:
            return 1.0
        el = od.orbital_elements(self._ref_state.mu, self._ref_state.position,
                                 self._ref_state.velocity)
        a = el.semi_major_axis
        if a is not None and el.is_bound:
            return el.period
        a_abs = el.semi_latus_rectum / max(el.eccentricity**2 - 1.0, 1e-6)
        return 2.0 * np.pi * float(np.sqrt(a_abs**3 / self._ref_state.mu))

    # -- Zustand --------------------------------------------------------------

    def _ensure_init(self, values: dict[str, Any]) -> None:
        body = get_body(values["body"])
        a = float(values["a"])
        e = float(values["e"])
        signature = (body.name, round(a, 3), round(e, 6))
        if signature == self._signature and self._ref_state is not None:
            return
        # Neue Startbahn -> Simulation zuruecksetzen, Start am Periapsis.
        start = sv.state_from_kepler(body.mu, a, e, 0.0)
        self._mission_time = 0.0
        self._ref_time = 0.0
        self._ref_state = start
        self._maneuvers = []
        self._energy_history = [(0.0, sv.specific_total_energy(start))]
        self._signature = signature

    def _current_state(self, mu: float) -> sv.OrbitState:
        assert self._ref_state is not None
        try:
            return od.propagate_time(mu, self._ref_state.position,
                                     self._ref_state.velocity,
                                     self._mission_time - self._ref_time)
        except Exception:
            applog.logger.exception("propagate_time fehlgeschlagen (Sandbox)")
            return self._ref_state

    # -- Log ------------------------------------------------------------------

    def has_log(self) -> bool:
        return True

    def _epoch_mjd(self, values: dict[str, Any]) -> float:
        return float(values.get("epoch_mjd", _DEFAULT_EPOCH_MJD))

    def log_entries(self) -> list[str]:
        epoch = self._signature_epoch
        lines = [f"Start (T+0): Startbahn, MJD {epoch:.4f}"]
        for i, m in enumerate(self._maneuvers, start=1):
            mjd = epoch + m["time"] / 86400.0
            direction = "prograd" if m["sign"] > 0 else "retrograd"
            dv_kms = m["dv"] / 1000.0
            t_str = timeutils.format_duration(m["time"])
            lines.append(
                f"#{i}  T+ {t_str}  MJD {mjd:.4f}  |  Δv {dv_kms:.3f} km/s {direction}"
                f"  ->  {m['classification']}"
            )
        return lines

    # -- Berechnung -----------------------------------------------------------

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        mu = body.mu
        self._ensure_init(values)
        # Epoche fuer die MJD-Anzeige (frei einstellbar, kein Reset).
        self._signature_epoch = self._epoch_mjd(values)

        sc = self._current_state(mu)
        el = od.orbital_elements(mu, self._ref_state.position, self._ref_state.velocity)
        mjd = self._signature_epoch + self._mission_time / 86400.0

        orbit_block = [
            ResultItem("Aktueller Radius", float(np.linalg.norm(sc.position)), "km"),
            ResultItem("Aktuelle Geschwindigkeit", float(np.linalg.norm(sc.velocity)), "km/s"),
            ResultItem("Exzentrizität", el.eccentricity, "-"),
            ResultItem("Periapsisradius", el.periapsis_radius, "km"),
            ResultItem("Spez. Gesamtenergie", el.specific_energy, "MJ/kg"),
        ]
        if el.is_bound:
            orbit_block += [
                ResultItem("Apoapsisradius", el.apoapsis_radius, "km"),
                ResultItem("Große Halbachse", el.semi_major_axis, "km"),
                ResultItem("Umlaufzeit", el.period, "min"),
            ]
        time_block = [
            ResultItem("Missionszeit", self._mission_time, "h"),
            ResultItem("MJD", mjd, "-"),
            ResultItem("Anzahl Manöver", float(len(self._maneuvers)), "-"),
            ResultItem("Angewendetes Gesamt-Δv",
                       sum(m["dv"] for m in self._maneuvers), "km/s"),
        ]

        items: list = [
            ResultSection("Missionszeit"),
            *time_block,
            ResultSection("Aktuelle Bahn"),
            *orbit_block,
        ]
        klass = el.classification
        notes = [
            f"Aktuelle Bahn: {klass}.",
            "prograd beschleunigt (hebt die Bahn), retrograd bremst ab (senkt sie).",
        ]
        status = (
            f"Missionszeit: T+ {timeutils.format_duration(self._mission_time)}  •  "
            f"MJD {mjd:.4f}\n"
            f"Aktuelle Bahn: {klass}  •  Manöver: {len(self._maneuvers)}"
        )

        data = {
            "mu": mu,
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "ref_position": np.asarray(self._ref_state.position, float),
            "ref_velocity": np.asarray(self._ref_state.velocity, float),
            "sc_position": sc.position,
            "sc_velocity": sc.velocity,
            "is_bound": el.is_bound,
            "mission_time": self._mission_time,
            "energy_history": list(self._energy_history),
        }
        return ModuleResult(items=items, notes=notes, data=data, status=status)

    # -- Visualisierung -------------------------------------------------------

    def plot(self, figure, values, result, options=None) -> None:
        figure.clear()
        if self.is_option_enabled(options, "show_energy"):
            ax, ax_energy = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        else:
            ax = figure.add_subplot(111)
            ax_energy = None

        data = result.data
        mu = data["mu"]
        body_r = data["body_radius"] / 1000.0
        body_name = data["body_name"]
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        sc_r = float(np.linalg.norm(data["sc_position"][:2])) / 1000.0
        scale_km = max(sc_r, body_r * 2.0)

        # Zentralkoerper.
        ax.fill(body_r * np.cos(theta), body_r * np.sin(theta), color="#264653",
                alpha=0.9, zorder=4, label=body_name)

        # Aktuelle Bahn (Form ueber die normierte Bahnposition).
        if self.is_option_enabled(options, "show_orbit"):
            try:
                taus = np.linspace(0.0, 1.0, 241)
                pts = np.array([
                    od.propagate(mu, data["ref_position"], data["ref_velocity"],
                                 float(tt)).position
                    for tt in taus
                ]) / 1000.0
                ax.plot(pts[:, 0], pts[:, 1], color="#e07a00", linewidth=2.2,
                        zorder=5, label="aktuelle Bahn")
                scale_km = max(scale_km, float(np.max(np.abs(pts))))
            except Exception:
                applog.logger.exception("Sandbox-Bahnkurve fehlgeschlagen")

        # Raumfahrzeug + Vektoren.
        draw_state_vectors(ax, data["sc_position"], data["sc_velocity"],
                           velocity_length_km=0.30 * scale_km,
                           show_radius=self.is_option_enabled(options, "show_radius"),
                           show_velocity=self.is_option_enabled(options, "show_velocity"),
                           add_legend=False)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Manöver-Sandbox um {body_name}")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize="x-small", framealpha=0.7)

        if ax_energy is not None:
            self._plot_energy(ax_energy, data)

    def _plot_energy(self, ax_energy, data) -> None:
        """Spezifische mechanische Gesamtenergie als Stufenkurve ueber die Zeit."""
        history = data["energy_history"]
        mission_time = data["mission_time"]
        # Stufenkurve: konstant je Segment, Sprung an jedem Manoever.
        times_h = [t / 3600.0 for t, _e in history] + [max(mission_time, 0.0) / 3600.0]
        energies = [from_si(e, "MJ/kg") for _t, e in history]
        energies = energies + [energies[-1]]
        ax_energy.step(times_h, energies, where="post", color="#1d3557")
        ax_energy.axvline(mission_time / 3600.0, color="0.5", linestyle="--",
                          linewidth=0.8)
        # Manoeverpunkte markieren.
        for (t, e) in history[1:]:
            ax_energy.plot([t / 3600.0], [from_si(e, "MJ/kg")], "o", color="#c1121f",
                           markersize=4)
        ax_energy.set_xlabel("Missionszeit [h]")
        ax_energy.set_ylabel("spez. Gesamtenergie [MJ/kg]")
        ax_energy.set_title("Energie über Missionszeit")
        ax_energy.grid(True, linestyle=":", alpha=0.35)

    # -- Erklaerung -----------------------------------------------------------

    def explanation(self) -> str:
        return (
            "Manöver-Sandbox\n"
            "===============\n\n"
            "Ein Raumfahrzeug startet auf einer einstellbaren Bahn (Kreis als\n"
            "Spezialfall der Ellipse, Kreis ist Default). Die Missionsuhr läuft\n"
            "kontinuierlich weiter – anders als die rein periodische τ-Animation\n"
            "zeigt sie die tatsächlich verstrichene Zeit (Tage/Std/Min und MJD).\n\n"
            "Manöver:\n"
            "  Mit 'Δv ausführen' wird an der aktuellen Position ein tangentiales\n"
            "  Delta-v gegeben – beliebig oft. prograd beschleunigt (hebt die Bahn\n"
            "  bzw. die gegenüberliegende Apsis), retrograd bremst ab (senkt sie).\n"
            "  Jedes Manöver wird im Manöver-Log mit Zeit und MJD protokolliert.\n\n"
            "Energie:\n"
            "  Die spezifische mechanische Gesamtenergie (kinetisch + potentiell)\n"
            "  ist auf einer Keplerbahn konstant und springt nur beim Δv. Der\n"
            "  optionale Energieplot zeigt diese Sprünge über die Missionszeit.\n\n"
            "MJD:\n"
            "  Das Modifizierte Julianische Datum zählt Tage; Bezugspunkt ist per\n"
            "  Default der heutige Tag, lässt sich aber frei einstellen.\n\n"
            "Idealisierungen: ebenes Zwei-Körper-Problem, impulsive (augenblickliche)\n"
            "Manöver, nur tangentialer Schub (beliebige Winkel folgen später)."
        )
