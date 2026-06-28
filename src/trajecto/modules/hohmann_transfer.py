"""Modul ``hohmann_transfer`` – Hohmann-Transfer zwischen zwei Kreisbahnen.

Didaktisches Transfermodul: der treibstoffguenstige Zwei-Impuls-Transfer
zwischen zwei koplanaren Kreisbahnen ueber eine Transferellipse, die beide
Bahnen tangential beruehrt. Unterstuetzt Transfers nach aussen (r2 > r1) und
nach innen (r2 < r1).

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

#: Standardfall: niedrige Erdumlaufbahn (~300 km) -> geostationaere Bahn.
_DEFAULT_R1_SI = 6_678_000.0   # ~300 km Hoehe (Radius vom Erdmittelpunkt)
_DEFAULT_R2_SI = 42_164_000.0  # geostationaerer Radius


def _direction_word(dv_signed: float) -> str:
    return "prograd" if dv_signed >= 0.0 else "retrograd"


class HohmannTransferModule(Module):
    """Hohmann-Transfer zwischen zwei koplanaren Kreisbahnen."""

    id = "hohmann_transfer"
    title = "Hohmann-Transfer"
    subtitle = "Zwei-Impuls-Transfer zwischen zwei Kreisbahnen"

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(
                name="body",
                label="Zentralkoerper",
                choices=tuple(BODIES.keys()),
                default=EARTH.name,
            ),
            FloatParameter(
                name="r1",
                label="Startradius r1",
                default_si=_DEFAULT_R1_SI,
                display_unit="km",
                minimum_si=1.0,
            ),
            FloatParameter(
                name="r2",
                label="Zielradius r2",
                default_si=_DEFAULT_R2_SI,
                display_unit="km",
                minimum_si=1.0,
            ),
        ]

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        r1 = float(values["r1"])
        r2 = float(values["r2"])

        # core validiert mu>0, r1>0, r2>0, r1 != r2 (loest ValueError aus).
        t = om.hohmann_transfer(body.mu, r1, r2)

        items = [
            ResultItem("Kreisbahngeschw. Startbahn", t.v_circ_1, "km/s"),
            ResultItem("Kreisbahngeschw. Zielbahn", t.v_circ_2, "km/s"),
            ResultItem("Transfergeschw. am Startpunkt", t.v_transfer_1, "km/s"),
            ResultItem("Transfergeschw. am Zielpunkt", t.v_transfer_2, "km/s"),
            ResultItem(f"Delta-v 1 ({_direction_word(t.dv1)})", abs(t.dv1), "km/s"),
            ResultItem(f"Delta-v 2 ({_direction_word(t.dv2)})", abs(t.dv2), "km/s"),
            ResultItem("Gesamt-Delta-v", t.dv_total, "km/s"),
            ResultItem("Transferzeit", t.transfer_time, "min"),
            ResultItem("Halbachse Transferellipse", t.a_transfer, "km"),
        ]

        notes: list[str] = [
            "Transferart: " + ("nach aussen (r2 > r1)" if t.outward else "nach innen (r2 < r1)"),
            "Radien sind vom Mittelpunkt des Zentralkoerpers gemessen, keine Hoehen.",
        ]
        # Plausibilitaetswarnung: Bahn innerhalb des Koerperradius.
        if r1 < body.mean_radius or r2 < body.mean_radius:
            notes.append(
                f"Warnung: r1 oder r2 liegt unterhalb des {body.name}-Radius "
                f"({body.mean_radius / 1000.0:,.0f} km) – physikalisch keine freie Bahn."
            )

        data = {
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "r1": r1,
            "r2": r2,
            "a_transfer": t.a_transfer,
            "dv1_prograde": t.dv1 >= 0.0,
            "dv2_prograde": t.dv2 >= 0.0,
        }
        return ModuleResult(items=items, notes=notes, data=data)

    # -- Visualisierung -------------------------------------------------------

    def plot(self, figure: Figure, values: dict[str, Any], result: ModuleResult) -> None:
        figure.clear()
        ax = figure.add_subplot(111)

        r1 = result.data["r1"] / 1000.0  # km
        r2 = result.data["r2"] / 1000.0
        body_r = result.data["body_radius"] / 1000.0
        body_name = result.data["body_name"]

        theta = np.linspace(0.0, 2.0 * np.pi, 361)

        # Start- und Zielkreisbahn.
        ax.plot(r1 * np.cos(theta), r1 * np.sin(theta),
                color="#3a7bd5", linewidth=1.4, zorder=2, label="Startbahn r1")
        ax.plot(r2 * np.cos(theta), r2 * np.sin(theta),
                color="#2a9d8f", linewidth=1.4, zorder=2, label="Zielbahn r2")

        # Transferellipse: Startpunkt auf +x (r1), Zielpunkt auf -x (r2),
        # Zentralkoerper im Fokus (Ursprung). Mittelpunkt liegt bei (r1-r2)/2.
        a = 0.5 * (r1 + r2)
        cx0 = 0.5 * (r1 - r2)
        b = np.sqrt(max(a * a - cx0 * cx0, 0.0))
        t_half = np.linspace(0.0, np.pi, 181)  # obere Haelfte = halbe Ellipse
        ax.plot(cx0 + a * np.cos(t_half), b * np.sin(t_half),
                color="#e76f51", linewidth=1.8, linestyle="--", zorder=3,
                label="Transferellipse")

        # Zentralkoerper im Fokus.
        ax.fill(body_r * np.cos(theta), body_r * np.sin(theta),
                color="#264653", alpha=0.9, zorder=4, label=body_name)

        # Start- und Zielpunkt markieren.
        ax.plot([r1], [0.0], "o", color="#1d3557", zorder=5)
        ax.annotate("Start (r1)", (r1, 0.0), textcoords="offset points",
                    xytext=(6, -12), fontsize="small", color="#1d3557")
        ax.plot([-r2], [0.0], "o", color="#006d77", zorder=5)
        ax.annotate("Ziel (r2)", (-r2, 0.0), textcoords="offset points",
                    xytext=(-10, 8), fontsize="small", color="#006d77")

        # Einfache Delta-v-Pfeile (tangential; Bewegung gegen den Uhrzeigersinn).
        arrow_len = 0.14 * max(r1, r2)
        # Burn 1 am Start (+x): Bahngeschwindigkeit zeigt nach +y.
        dy1 = arrow_len if result.data["dv1_prograde"] else -arrow_len
        ax.annotate("", xy=(r1, dy1), xytext=(r1, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
        # Burn 2 am Ziel (-x): Bahngeschwindigkeit zeigt nach -y.
        dy2 = -arrow_len if result.data["dv2_prograde"] else arrow_len
        ax.annotate("", xy=(-r2, dy2), xytext=(-r2, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Hohmann-Transfer um {body_name}")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize="small")

    def explanation(self) -> str:
        return (
            "Hohmann-Transfer\n"
            "================\n\n"
            "Der Hohmann-Transfer ist der treibstoffguenstigste Weg, um zwischen\n"
            "zwei koplanaren Kreisbahnen mit den Radien r1 und r2 zu wechseln.\n\n"
            "Zwei impulsive Manoever:\n"
            "  Delta-v 1 bringt das Raumfahrzeug von der Startkreisbahn auf eine\n"
            "  Transferellipse. Delta-v 2 macht am gegenueberliegenden Punkt aus\n"
            "  der Ellipse wieder eine Kreisbahn (die Zielbahn). Zwei getrennte\n"
            "  Schuebe, weil ein einzelner Impuls nie zwei verschiedene\n"
            "  Kreisbahnen verbinden kann.\n\n"
            "Warum eine Ellipse?\n"
            "  Ein einzelner Schub am Start aendert die Geschwindigkeit, nicht\n"
            "  sofort den Ort. Die resultierende gebundene Bahn ist nach Kepler\n"
            "  eine Ellipse, die die Startbahn (Periapsis bzw. Apoapsis) und die\n"
            "  Zielbahn am anderen Apsidenpunkt beruehrt. Ihre grosse Halbachse\n"
            "  ist a = (r1 + r2) / 2.\n\n"
            "Transferzeit:\n"
            "  Das Fahrzeug durchlaeuft genau die Haelfte der Transferellipse\n"
            "  (vom Start- zum Zielpunkt, 180 Grad). Daher ist die Transferzeit\n"
            "  die halbe Umlaufzeit der Transferellipse: t = T_transfer / 2.\n\n"
            "Richtung:\n"
            "  Transfer nach aussen (r2 > r1): beide Schuebe prograd.\n"
            "  Transfer nach innen (r2 < r1): beide Schuebe retrograd (abbremsen).\n"
            "  Die angezeigten Delta-v-Werte sind Betraege.\n\n"
            "Hinweis: r1 und r2 sind Radien vom Mittelpunkt des Zentralkoerpers,\n"
            "keine Hoehen ueber der Oberflaeche.\n\n"
            "Idealisierungen: koplanare Kreisbahnen, impulsive (augenblickliche)\n"
            "Manoever, reines Zwei-Koerper-Problem, keine Stoerungen."
        )
