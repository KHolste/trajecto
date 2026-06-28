"""Modul ``kepler_laws`` – die drei Keplerschen Gesetze anschaulich.

Didaktisches Grundlagenmodul:

* 1. Gesetz – elliptische Bahn mit Zentralkoerper in einem Brennpunkt.
* 2. Gesetz – gleiche Flaechen in gleichen Zeiten (zeitgetreu markierte Segmente).
* 3. Gesetz – Zusammenhang von grosser Halbachse a und Umlaufzeit T.

Die gesamte Fachlogik stammt aus ``core``; das Modul orchestriert nur Eingaben,
Ergebnisse, Visualisierung und Erklaerung.
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

#: Standardwerte: deutlich elliptische Erdbahn, kollisionsfrei.
_DEFAULT_A_SI = 24_000_000.0  # 24 000 km
_DEFAULT_E = 0.3


class KeplerLawsModule(Module):
    """Keplersche Gesetze an einer frei waehlbaren Ellipsenbahn."""

    id = "kepler_laws"
    title = "Keplersche Gesetze"
    subtitle = "Ellipsenbahn, Flaechensatz und 3. Gesetz"

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(
                name="body",
                label="Zentralkoerper",
                choices=tuple(BODIES.keys()),
                default=EARTH.name,
            ),
            FloatParameter(
                name="a",
                label="Grosse Halbachse a",
                default_si=_DEFAULT_A_SI,
                display_unit="km",
                minimum_si=1.0,
            ),
            FloatParameter(
                name="e",
                label="Exzentrizitaet e",
                default_si=_DEFAULT_E,
                display_unit="-",
                minimum_si=0.0,
                maximum_si=0.95,
            ),
            ChoiceParameter(
                name="segments",
                label="Flaechensegmente (2. Gesetz)",
                choices=("4", "6", "8", "12"),
                default="6",
            ),
        ]

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        a = float(values["a"])
        e = float(values["e"])
        segments = int(values.get("segments", 6))

        # core validiert a > 0 und 0 <= e < 1 (loest ValueError aus).
        r_peri = om.periapsis_radius(a, e)
        r_apo = om.apoapsis_radius(a, e)
        period = om.orbital_period(body.mu, a)
        v_peri = om.vis_viva_speed(body.mu, r_peri, a)
        v_apo = om.vis_viva_speed(body.mu, r_apo, a)
        speed_ratio = v_peri / v_apo

        items = [
            ResultItem("Periapsisradius", r_peri, "km"),
            ResultItem("Apoapsisradius", r_apo, "km"),
            ResultItem("Umlaufzeit", period, "min"),
            ResultItem("Geschwindigkeit im Periapsis", v_peri, "km/s"),
            ResultItem("Geschwindigkeit im Apoapsis", v_apo, "km/s"),
            ResultItem("Verhaeltnis v_peri / v_apo", speed_ratio, "-"),
        ]
        data = {
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "a": a,
            "e": e,
            "r_peri": r_peri,
            "r_apo": r_apo,
            "segments": segments,
        }
        return ModuleResult(items=items, data=data)

    # -- Visualisierung -------------------------------------------------------

    def plot(self, figure: Figure, values: dict[str, Any], result: ModuleResult) -> None:
        figure.clear()
        ax = figure.add_subplot(111)

        a = result.data["a"] / 1000.0  # km
        e = result.data["e"]
        body_r = result.data["body_radius"] / 1000.0
        body_name = result.data["body_name"]
        segments = result.data["segments"]

        c = a * e  # lineare Exzentrizitaet (Fokus-Abstand vom Mittelpunkt)
        b = a * np.sqrt(1.0 - e * e)  # kleine Halbachse

        # Bahnpunkt relativ zum Fokus (Zentralkoerper im Ursprung), aus
        # exzentrischer Anomalie E. Mittelpunkt der Ellipse liegt bei (-c, 0).
        def point(eccentric: np.ndarray | float):
            x = a * np.cos(eccentric) - c
            y = b * np.sin(eccentric)
            return x, y

        # 2. Gesetz: gleiche Flaeche je Zeitsegment ↔ gleiche
        # Mittelpunktsanomalie-Intervalle. Pro Segment einen Keil vom Fokus
        # zeichnen; gleiche Flaechen, aber unterschiedliche Winkelbreiten.
        colors = ["#f4a259", "#5b8e7d"]
        for i in range(segments):
            m0 = 2.0 * np.pi * i / segments
            m1 = 2.0 * np.pi * (i + 1) / segments
            m_samples = np.linspace(m0, m1, 24)
            ecc = np.array([om.solve_eccentric_anomaly(float(m), e) for m in m_samples])
            px, py = point(ecc)
            poly_x = np.concatenate(([0.0], px, [0.0]))
            poly_y = np.concatenate(([0.0], py, [0.0]))
            ax.fill(
                poly_x, poly_y,
                color=colors[i % 2], alpha=0.45, edgecolor="none", zorder=1,
            )

        # Vollstaendige Ellipsenbahn.
        ecc_full = np.linspace(0.0, 2.0 * np.pi, 361)
        ex, ey = point(ecc_full)
        ax.plot(ex, ey, color="#2b2d42", linewidth=1.6, zorder=3, label="Bahn")

        # Halbachsen dezent andeuten (a entlang Hauptachse, b durch Mittelpunkt).
        ax.plot(
            [-(a + c), a - c], [0.0, 0.0],
            color="#888888", linestyle="--", linewidth=0.8, zorder=2,
        )
        ax.plot(
            [-c, -c], [-b, b],
            color="#888888", linestyle=":", linewidth=0.8, zorder=2,
        )

        # Zentralkoerper im Brennpunkt (Ursprung).
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        ax.fill(
            body_r * np.cos(theta), body_r * np.sin(theta),
            color="#3a7bd5", alpha=0.9, zorder=4, label=body_name,
        )

        # Periapsis (+x) und Apoapsis (-x) markieren.
        r_peri = result.data["r_peri"] / 1000.0
        r_apo = result.data["r_apo"] / 1000.0
        ax.plot([r_peri], [0.0], "o", color="#c1121f", zorder=5)
        ax.annotate("Periapsis", (r_peri, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize="small", color="#c1121f")
        ax.plot([-r_apo], [0.0], "o", color="#005f73", zorder=5)
        ax.annotate("Apoapsis", (-r_apo, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize="small", color="#005f73")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Keplerbahn um {body_name}  (a={a:,.0f} km, e={e:.2f})")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize="small")

    def explanation(self) -> str:
        return (
            "Keplersche Gesetze\n"
            "==================\n\n"
            "1. Gesetz (Bahnform):\n"
            "   Jeder Planet bewegt sich auf einer Ellipse, in deren einem\n"
            "   Brennpunkt der Zentralkoerper steht – nicht im Mittelpunkt.\n\n"
            "2. Gesetz (Flaechensatz):\n"
            "   Die Verbindungslinie Koerper–Zentralkoerper ueberstreicht in\n"
            "   gleichen Zeiten gleiche Flaechen. Die farbigen Segmente im Plot\n"
            "   haben dieselbe Flaeche und stehen fuer gleiche Zeitabschnitte:\n"
            "   nahe der Periapsis ist die Bahn schnell, nahe der Apoapsis langsam.\n\n"
            "3. Gesetz (Periodengesetz):\n"
            "   T = 2*pi*sqrt(a^3 / mu).  Die Umlaufzeit haengt nur von der\n"
            "   grossen Halbachse a (und mu) ab, nicht von der Exzentrizitaet.\n\n"
            "Begriffe:\n"
            "  a  – grosse Halbachse (mittlerer Bahnradius)\n"
            "  e  – Exzentrizitaet (0 = Kreis, naeher 1 = stark gestreckt)\n"
            "  Periapsis  r_p = a*(1 - e):  naechster Punkt, hoechste Geschwindigkeit\n"
            "  Apoapsis   r_a = a*(1 + e):  fernster Punkt, kleinste Geschwindigkeit\n\n"
            "Daher ist die Geschwindigkeit im Periapsis stets groesser als im\n"
            "Apoapsis: v_peri / v_apo = (1 + e) / (1 - e)."
        )
