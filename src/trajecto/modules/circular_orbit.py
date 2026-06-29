"""Referenzmodul ``circular_orbit`` – Kreisbahn um einen Zentralkoerper.

Didaktisches Minimalmodul: Es zeigt am Beispiel der idealen Kreisbahn die
drei grundlegenden Kennzahlen Kreisbahngeschwindigkeit, Umlaufzeit und
Fluchtgeschwindigkeit und visualisiert die Bahn um den Zentralkoerper.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import BODIES, EARTH, get_body
from trajecto.core.orbit_input import (
    INPUT_MODE_ALTITUDE,
    INPUT_MODE_RADIUS,
    INPUT_MODES,
    altitude_from_radius,
    radius_from_input,
)
from trajecto.modules.base import (
    ChoiceParameter,
    FloatParameter,
    Module,
    ModuleResult,
    Parameter,
    PlotOption,
    ResultItem,
)
from trajecto.modules.dynamics_view import (
    draw_energy_over_tau,
    draw_state_vectors,
    dynamic_result_items,
    sample_energies,
)

#: Default: Hoehe ueber Oberflaeche, 500 km (niedrige Erdumlaufbahn).
_DEFAULT_VALUE_SI = 500_000.0
#: Default-Zeitposition tau (0..1) fuer eine gut sichtbare Position.
_DEFAULT_TAU = 0.125  # entspricht nu = 45 Grad


class CircularOrbitModule(Module):
    """Kreisbahn: Geschwindigkeit, Umlaufzeit, Fluchtgeschwindigkeit."""

    id = "circular_orbit"
    title = "Kreisbahn"
    subtitle = "Geschwindigkeit, Umlaufzeit und Fluchtgeschwindigkeit"

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(
                name="body",
                label="Zentralkörper",
                choices=tuple(BODIES.keys()),
                default=EARTH.name,
            ),
            ChoiceParameter(
                name="input_mode",
                label="Eingabemodus",
                choices=INPUT_MODES,
                default=INPUT_MODE_ALTITUDE,
            ),
            FloatParameter(
                name="radius",
                label="Bahnradius bzw. Höhe",
                default_si=_DEFAULT_VALUE_SI,
                display_unit="km",
                minimum_si=-1.0e7,
            ),
            FloatParameter(
                name="tau",
                label="Zeitposition τ (0..1)",
                default_si=_DEFAULT_TAU,
                display_unit="-",
                minimum_si=0.0,
                maximum_si=1.0,
                animatable=True,
            ),
        ]

    def plot_options(self) -> list[PlotOption]:
        return [
            PlotOption("show_radius", "Radiusvektor anzeigen", True),
            PlotOption("show_velocity", "Geschwindigkeitsvektor anzeigen", True),
            PlotOption("show_energy", "Energieplot anzeigen", False),
        ]

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        mode = values.get("input_mode", INPUT_MODE_RADIUS)
        raw_value = float(values["radius"])

        # Bahnradius zentral aus Eingabewert + Modus bestimmen.
        # radius_from_input wirft bei resultierendem r <= 0 einen ValueError.
        radius = radius_from_input(body, raw_value, mode)
        altitude = altitude_from_radius(body, radius)
        # Zeitposition tau hat Vorrang; auf der Kreisbahn ist nu = 2*pi*tau
        # (gleichfoermige Bewegung). Direkter nu-Wert bleibt als Fallback moeglich.
        if "tau" in values:
            nu = 2.0 * math.pi * float(values["tau"])
        else:
            nu = float(values.get("nu", 0.0))

        v_circ = om.circular_orbit_velocity(body.mu, radius)
        period = om.circular_orbit_period(body.mu, radius)
        v_esc = om.escape_velocity(body.mu, radius)

        # Aktueller Bahnzustand am gewaehlten Bahnwinkel.
        state = sv.state_circular(body.mu, radius, nu)

        items = [
            ResultItem("Bahnradius", radius, "km"),
            ResultItem("Bahnhöhe über Oberfläche", altitude, "km"),
            ResultItem("Kreisbahngeschwindigkeit", v_circ, "km/s"),
            ResultItem("Umlaufzeit", period, "min"),
            ResultItem("Fluchtgeschwindigkeit", v_esc, "km/s"),
            ResultItem("Aktuelle wahre Anomalie", nu, "deg"),
            *dynamic_result_items(state),
        ]
        notes: list[str] = []
        if altitude < 0.0:
            notes.append(
                f"Warnung: Hoehe ist negativ ({altitude / 1000.0:,.0f} km) – "
                f"die Bahn liegt unterhalb der Oberflaeche von {body.name}."
            )
        data = {
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "orbit_radius": radius,
            "mu": body.mu,
            "position": state.position,
            "velocity": state.velocity,
        }
        return ModuleResult(items=items, notes=notes, data=data)

    def plot(
        self,
        figure: Figure,
        values: dict[str, Any],
        result: ModuleResult,
        options: dict[str, bool] | None = None,
    ) -> None:
        figure.clear()
        if self.is_option_enabled(options, "show_energy"):
            ax, ax_energy = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        else:
            ax = figure.add_subplot(111)
            ax_energy = None

        body_radius = result.data["body_radius"]
        orbit_radius = result.data["orbit_radius"]
        body_name = result.data["body_name"]

        # In km darstellen.
        body_r_km = body_radius / 1000.0
        orbit_r_km = orbit_radius / 1000.0

        theta = np.linspace(0.0, 2.0 * np.pi, 361)

        # Zentralkoerper als gefuellte Scheibe.
        ax.fill(
            body_r_km * np.cos(theta),
            body_r_km * np.sin(theta),
            color="#3a7bd5",
            alpha=0.85,
            zorder=2,
            label=body_name,
        )
        # Kreisbahn.
        ax.plot(
            orbit_r_km * np.cos(theta),
            orbit_r_km * np.sin(theta),
            color="#d2691e",
            linewidth=1.8,
            zorder=3,
            label="Kreisbahn",
        )
        # Aktueller Zustand: Satellit, Radius- und (skalierter) Geschwindigkeitsvektor.
        draw_state_vectors(
            ax,
            result.data["position"],
            result.data["velocity"],
            velocity_length_km=0.35 * orbit_r_km,
            show_radius=self.is_option_enabled(options, "show_radius"),
            show_velocity=self.is_option_enabled(options, "show_velocity"),
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Kreisbahn um {body_name}")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize="small")

        if ax_energy is not None:
            mu = result.data["mu"]
            taus, kin, pot, tot = sample_energies(
                lambda tau: sv.state_circular(mu, orbit_radius, 2.0 * math.pi * tau)
            )
            draw_energy_over_tau(ax_energy, taus, kin, pot, tot,
                                 current_tau=values.get("tau"))

    def explanation(self) -> str:
        return (
            "Kreisbahn\n"
            "=========\n\n"
            "Auf einer idealen Kreisbahn hält die Gravitation als "
            "Zentripetalkraft den Körper auf konstantem Radius r um den "
            "Zentralkörper (Gravitationsparameter mu = G*M).\n\n"
            "- Kreisbahngeschwindigkeit:  v = sqrt(mu / r)\n"
            "- Umlaufzeit:                T = 2*pi*sqrt(r^3 / mu)\n"
            "- Fluchtgeschwindigkeit:     v_esc = sqrt(2*mu / r) = sqrt(2) * v\n\n"
            "Je größer der Radius, desto kleiner die Bahngeschwindigkeit und "
            "desto länger die Umlaufzeit. Die Fluchtgeschwindigkeit am selben "
            "Radius ist stets um den Faktor sqrt(2) größer als die "
            "Kreisbahngeschwindigkeit.\n\n"
            "Zustand entlang der Bahn:\n"
            "  Der Radiusvektor zeigt vom Zentralkörper zum Satelliten, der\n"
            "  Geschwindigkeitsvektor steht tangential (senkrecht dazu).\n"
            "  Auf der Kreisbahn bleiben Geschwindigkeit, kinetische, potentielle\n"
            "  und Gesamtenergie sowie der Drehimpuls konstant.\n"
            "  Der Exzentrizitätsvektor ist (numerisch) null.\n\n"
            "Animation (zeitbasiert):\n"
            "  Der animierte Parameter ist die normierte Zeitposition τ (0..1),\n"
            "  also der Bruchteil einer Umlaufzeit. Auf der Kreisbahn gilt\n"
            "  nu = 2*pi*τ: die wahre Anomalie ist proportional zur Zeit, die\n"
            "  Bewegung ist gleichförmig. Die Ergebniszeile 'Aktuelle wahre\n"
            "  Anomalie' wird aus τ berechnet."
        )
