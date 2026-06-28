"""Referenzmodul ``circular_orbit`` – Kreisbahn um einen Zentralkoerper.

Didaktisches Minimalmodul: Es zeigt am Beispiel der idealen Kreisbahn die
drei grundlegenden Kennzahlen Kreisbahngeschwindigkeit, Umlaufzeit und
Fluchtgeschwindigkeit und visualisiert die Bahn um den Zentralkoerper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from trajecto.core import orbital_mechanics as om
from trajecto.core.bodies import BODIES, EARTH, get_body
from trajecto.modules.base import (
    ChoiceParameter,
    FloatParameter,
    Module,
    ModuleResult,
    Parameter,
    ResultItem,
)

#: Standard-Bahnradius: Erdradius + 500 km (niedrige Umlaufbahn).
_DEFAULT_RADIUS_SI = EARTH.mean_radius + 500_000.0


class CircularOrbitModule(Module):
    """Kreisbahn: Geschwindigkeit, Umlaufzeit, Fluchtgeschwindigkeit."""

    id = "circular_orbit"
    title = "Kreisbahn"
    subtitle = "Geschwindigkeit, Umlaufzeit und Fluchtgeschwindigkeit"

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(
                name="body",
                label="Zentralkoerper",
                choices=tuple(BODIES.keys()),
                default=EARTH.name,
            ),
            FloatParameter(
                name="radius",
                label="Bahnradius",
                default_si=_DEFAULT_RADIUS_SI,
                display_unit="km",
                minimum_si=1.0,
            ),
        ]

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        radius = float(values["radius"])

        # om.* validiert den Radius und wirft bei r <= 0 einen ValueError.
        v_circ = om.circular_orbit_velocity(body.mu, radius)
        period = om.circular_orbit_period(body.mu, radius)
        v_esc = om.escape_velocity(body.mu, radius)
        altitude = radius - body.mean_radius

        items = [
            ResultItem("Bahnradius", radius, "km"),
            ResultItem("Bahnhoehe ueber Oberflaeche", altitude, "km"),
            ResultItem("Kreisbahngeschwindigkeit", v_circ, "km/s"),
            ResultItem("Umlaufzeit", period, "min"),
            ResultItem("Fluchtgeschwindigkeit", v_esc, "km/s"),
        ]
        data = {
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "orbit_radius": radius,
        }
        return ModuleResult(items=items, data=data)

    def plot(self, figure: Figure, values: dict[str, Any], result: ModuleResult) -> None:
        figure.clear()
        ax = figure.add_subplot(111)

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
        # Satellitenposition (bei theta = 0) als Markierung.
        ax.plot([orbit_r_km], [0.0], "o", color="#222222", zorder=4)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Kreisbahn um {body_name}")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize="small")

    def explanation(self) -> str:
        return (
            "Kreisbahn\n"
            "=========\n\n"
            "Auf einer idealen Kreisbahn haelt die Gravitation als "
            "Zentripetalkraft den Koerper auf konstantem Radius r um den "
            "Zentralkoerper (Gravitationsparameter mu = G*M).\n\n"
            "- Kreisbahngeschwindigkeit:  v = sqrt(mu / r)\n"
            "- Umlaufzeit:                T = 2*pi*sqrt(r^3 / mu)\n"
            "- Fluchtgeschwindigkeit:     v_esc = sqrt(2*mu / r) = sqrt(2) * v\n\n"
            "Je groesser der Radius, desto kleiner die Bahngeschwindigkeit und "
            "desto laenger die Umlaufzeit. Die Fluchtgeschwindigkeit am selben "
            "Radius ist stets um den Faktor sqrt(2) groesser als die "
            "Kreisbahngeschwindigkeit."
        )
